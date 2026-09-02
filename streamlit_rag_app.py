import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import warnings

# Suppress noisy warnings
warnings.filterwarnings("ignore")

import streamlit as st
from dotenv import load_dotenv

# PDF & Image processing imports
from pypdf import PdfReader
import pypdfium2

# LangChain & AI imports
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_app")

# Ensure base upload & asset directories exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)

ASSETS_DIR = UPLOAD_DIR / "extracted_assets"
ASSETS_DIR.mkdir(exist_ok=True, parents=True)


def sanitize_key(key: Optional[str]) -> Optional[str]:
    """Sanitize API key strings by stripping whitespace and wrapping quotes."""
    if not key:
        return None
    cleaned = str(key).strip().strip('"').strip("'")
    return cleaned if cleaned else None


# ------------------------------------------------------------------------------
# Asset Extraction: Extract Embedded Images & Page Snapshots
# ------------------------------------------------------------------------------
def extract_pdf_assets(file_path: str) -> Dict[int, List[str]]:
    """
    Extract embedded raster images and high-resolution page snapshots from PDF.
    Returns a dictionary mapping page_number -> list of relative image paths.
    """
    pdf_name = Path(file_path).stem
    doc_assets_dir = ASSETS_DIR / pdf_name
    doc_assets_dir.mkdir(parents=True, exist_ok=True)
    
    extracted_images: Dict[int, List[str]] = {}
    
    # 1. Extract embedded raster images using PyPDF
    try:
        reader = PdfReader(file_path)
        for page_idx, page in enumerate(reader.pages, start=1):
            page_imgs = []
            for img_idx, img_obj in enumerate(page.images, start=1):
                try:
                    img_filename = f"page_{page_idx}_img_{img_idx}_{img_obj.name}"
                    save_path = doc_assets_dir / img_filename
                    with open(save_path, "wb") as f:
                        f.write(img_obj.data)
                    page_imgs.append(str(save_path))
                except Exception:
                    pass
            if page_imgs:
                extracted_images[page_idx] = page_imgs
    except Exception as e:
        logger.warning(f"PyPDF embedded image extraction notice: {e}")

    # 2. Render high-resolution page snapshots using pypdfium2
    try:
        pdf = pypdfium2.PdfDocument(file_path)
        for page_idx in range(len(pdf)):
            page_num = page_idx + 1
            snapshot_filename = f"page_{page_num}_snapshot.png"
            snapshot_path = doc_assets_dir / snapshot_filename
            
            if not snapshot_path.exists():
                page = pdf[page_idx]
                pil_image = page.render(scale=1.5).to_pil()
                pil_image.save(snapshot_path)
            
            if page_num not in extracted_images:
                extracted_images[page_num] = [str(snapshot_path)]
            elif str(snapshot_path) not in extracted_images[page_num]:
                extracted_images[page_num].append(str(snapshot_path))
    except Exception as e:
        logger.warning(f"pypdfium2 page snapshot notice: {e}")

    return extracted_images


# ------------------------------------------------------------------------------
# Helper Functions: PDF Parsing & Table Extraction with Unstructured
# ------------------------------------------------------------------------------
def process_pdf_with_unstructured(
    file_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    page_assets: Optional[Dict[int, List[str]]] = None
) -> List[Document]:
    """
    Parse PDF using Unstructured with table structure inference and associate images.
    Preserves tables (HTML + text), section titles, and page visual assets.
    """
    documents = []
    file_name = Path(file_path).name
    page_assets = page_assets or {}
    
    try:
        from unstructured.partition.pdf import partition_pdf
        from unstructured.chunking.title import chunk_by_title
        
        # Partition PDF with table structure inference
        elements = partition_pdf(
            filename=file_path,
            strategy="fast",
            infer_table_structure=True,
            extract_images_in_pdf=False
        )
        
        if elements:
            for el in elements:
                if el.category == "Table":
                    page_num = getattr(el.metadata, "page_number", 1) if hasattr(el, "metadata") else 1
                    table_html = getattr(el.metadata, "text_as_html", "") if hasattr(el, "metadata") else ""
                    table_text = str(el).strip()
                    
                    if table_text or table_html:
                        doc = Document(
                            page_content=f"[TABLE EXTRACTED FROM PAGE {page_num}]:\n{table_text}\n\nHTML Structure:\n{table_html}" if table_html else f"[TABLE EXTRACTED FROM PAGE {page_num}]:\n{table_text}",
                            metadata={
                                "source": file_name,
                                "file_path": str(file_path),
                                "page": int(page_num) if page_num else 1,
                                "chunk_id": len(documents) + 1,
                                "char_count": len(table_text),
                                "parser": "unstructured_table",
                                "is_table": True,
                                "table_html": table_html,
                                "images": page_assets.get(int(page_num) if page_num else 1, [])
                            }
                        )
                        documents.append(doc)

            # Chunk elements by title/section
            chunks = chunk_by_title(
                elements,
                max_characters=chunk_size,
                new_after_n_chars=max(100, chunk_size - chunk_overlap),
                combine_text_under_n_chars=150
            )
            
            for i, chunk in enumerate(chunks):
                text = str(chunk).strip()
                if not text:
                    continue
                
                page_num = getattr(chunk.metadata, "page_number", 1) if hasattr(chunk, "metadata") else 1
                table_html = getattr(chunk.metadata, "text_as_html", "") if hasattr(chunk, "metadata") else ""
                
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": file_name,
                        "file_path": str(file_path),
                        "page": int(page_num) if page_num else 1,
                        "chunk_id": len(documents) + 1,
                        "char_count": len(text),
                        "parser": "unstructured",
                        "is_table": bool(table_html),
                        "table_html": table_html,
                        "images": page_assets.get(int(page_num) if page_num else 1, [])
                    }
                )
                documents.append(doc)
                
    except Exception as e:
        logger.warning(f"Unstructured PDF partition notice: {e}. Falling back to PyPDF loader...")
    
    # Fallback to PyPDF if unstructured returned no chunks
    if not documents:
        try:
            from pypdf import PdfReader
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            
            reader = PdfReader(file_path)
            raw_docs = []
            for page_idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    raw_docs.append(Document(
                        page_content=text.strip(),
                        metadata={
                            "source": file_name,
                            "file_path": str(file_path),
                            "page": page_idx + 1,
                            "parser": "pypdf",
                            "is_table": False,
                            "table_html": "",
                            "images": page_assets.get(page_idx + 1, [])
                        }
                    ))
            
            if raw_docs:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separators=["\n\n", "\n", ". ", " ", ""]
                )
                split_docs = splitter.split_documents(raw_docs)
                for idx, doc in enumerate(split_docs):
                    doc.metadata["chunk_id"] = idx + 1
                    doc.metadata["char_count"] = len(doc.page_content)
                    doc.metadata["images"] = page_assets.get(doc.metadata["page"], [])
                    documents.append(doc)
        except Exception as e:
            raise RuntimeError(f"Failed to parse PDF file '{file_name}': {e}")

    return documents


# ------------------------------------------------------------------------------
# Qdrant Vector DB & Embeddings Helper Functions
# ------------------------------------------------------------------------------
def get_qdrant_client(url: str, api_key: Optional[str] = None) -> QdrantClient:
    """Initialize and validate Qdrant client connection."""
    if not url:
        raise ValueError("Qdrant URL must be provided in .env (QDRANT_URL).")
    
    clean_url = url.strip().rstrip("/")
    clean_key = sanitize_key(api_key) or sanitize_key(os.getenv("QDRANT_API_KEY"))
    
    client = QdrantClient(
        url=clean_url,
        api_key=clean_key,
        timeout=30,
        check_compatibility=False
    )
    return client


def get_collection_vector_size(col_info) -> Optional[int]:
    """Safely extract vector dimensionality from Qdrant collection info."""
    try:
        vectors_config = col_info.config.params.vectors
        if hasattr(vectors_config, "size"):
            return vectors_config.size
        elif isinstance(vectors_config, dict):
            if "size" in vectors_config:
                return vectors_config["size"]
            for v in vectors_config.values():
                if hasattr(v, "size"):
                    return v.size
                elif isinstance(v, dict) and "size" in v:
                    return v["size"]
    except Exception:
        pass
    return None


def ensure_qdrant_collection(client: QdrantClient, collection_name: str, vector_size: int = 3072):
    """Ensure the target collection exists in Qdrant server with matching dimension and payload index."""
    try:
        collections_res = client.get_collections()
        existing_collections = [c.name for c in collections_res.collections]
        
        if collection_name not in existing_collections:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(
                    size=vector_size,
                    distance=qmodels.Distance.COSINE
                )
            )
            logger.info(f"Created new Qdrant collection: '{collection_name}' (dim={vector_size})")
        else:
            col_info = client.get_collection(collection_name=collection_name)
            current_dim = get_collection_vector_size(col_info)
            if current_dim is not None and current_dim != vector_size:
                logger.warning(f"Collection '{collection_name}' dimension mismatch: current {current_dim} != expected {vector_size}. Recreating collection...")
                client.delete_collection(collection_name=collection_name)
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=vector_size,
                        distance=qmodels.Distance.COSINE
                    )
                )
        
        # Ensure payload index exists on metadata.source for filtering
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name="metadata.source",
                field_schema=qmodels.PayloadSchemaType.KEYWORD
            )
        except Exception:
            pass

    except Exception as e:
        if "403" in str(e) or "forbidden" in str(e).lower():
            raise PermissionError(
                "Qdrant returned 403 Forbidden. Please verify that your QDRANT_API_KEY in .env has write/admin permissions."
            )
        raise e


def get_indexed_sources(client: QdrantClient, collection_name: str) -> List[str]:
    """Retrieve distinct document filenames indexed in the Qdrant collection."""
    sources = set()
    try:
        if not client.collection_exists(collection_name):
            return []
        points, _ = client.scroll(
            collection_name=collection_name,
            limit=150,
            with_payload=True,
            with_vectors=False
        )
        for pt in points:
            src = pt.payload.get("metadata", {}).get("source") or pt.payload.get("source")
            if src:
                sources.add(src)
    except Exception as e:
        logger.warning(f"Error fetching sources from Qdrant: {e}")
    return sorted(list(sources))


def get_vector_store(
    client: QdrantClient,
    collection_name: str,
    gemini_api_key: str,
    embedding_model_name: str = "models/gemini-embedding-001"
) -> QdrantVectorStore:
    """Initialize LangChain QdrantVectorStore with Gemini Embeddings."""
    clean_gemini_key = sanitize_key(gemini_api_key) or sanitize_key(os.getenv("GEMINI_API_KEY")) or sanitize_key(os.getenv("GOOGLE_API_KEY"))
    if not clean_gemini_key:
        raise ValueError("GEMINI_API_KEY is required in .env file.")
        
    embeddings = GoogleGenerativeAIEmbeddings(
        model=embedding_model_name,
        google_api_key=clean_gemini_key
    )
    
    vector_size = 3072
    try:
        sample_vec = embeddings.embed_query("dimension check")
        if sample_vec:
            vector_size = len(sample_vec)
    except Exception:
        pass
    
    ensure_qdrant_collection(client, collection_name, vector_size=vector_size)
    
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings
    )
    return vectorstore


def index_documents_to_qdrant(
    documents: List[Document],
    vectorstore: QdrantVectorStore,
    batch_size: int = 15,
    progress_bar = None,
    status_text = None
) -> int:
    """Ingest document chunks into Qdrant server in batches with progress updates."""
    total_docs = len(documents)
    total_batches = (total_docs + batch_size - 1) // batch_size
    
    for i in range(0, total_docs, batch_size):
        batch = documents[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        
        if status_text:
            status_text.text(f"⏳ Embedding & indexing batch {batch_num}/{total_batches} ({min(i + batch_size, total_docs)}/{total_docs} chunks)...")
        
        retries = 3
        while retries > 0:
            try:
                vectorstore.add_documents(batch)
                break
            except Exception as e:
                retries -= 1
                if retries == 0:
                    if "403" in str(e) or "forbidden" in str(e).lower():
                        raise PermissionError("Qdrant returned 403 Forbidden during document upsert. Please check your QDRANT_API_KEY in .env.")
                    raise e
                time.sleep(3)
        
        if progress_bar:
            progress_bar.progress(min(1.0, (i + len(batch)) / total_docs))
            
    return total_docs


# ------------------------------------------------------------------------------
# RAG Query & Multi-Modal LLM Generation
# ------------------------------------------------------------------------------
def generate_rag_response(
    query: str,
    vectorstore: QdrantVectorStore,
    gemini_api_key: str,
    model_name: str = "gemini-3.6-flash",
    k: int = 4,
    doc_filter: Optional[str] = None
) -> Tuple[str, List[Tuple[Document, float]], List[Dict]]:
    """
    Retrieve relevant chunks from Qdrant and generate an answer with Gemini LLM.
    Supports optional document source filtering.
    """
    clean_gemini_key = sanitize_key(gemini_api_key) or sanitize_key(os.getenv("GEMINI_API_KEY")) or sanitize_key(os.getenv("GOOGLE_API_KEY"))
    if not clean_gemini_key:
        raise ValueError("GEMINI_API_KEY is required in .env file.")

    filter_obj = None
    if doc_filter and doc_filter != "All Documents":
        filter_obj = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="metadata.source",
                    match=qmodels.MatchValue(value=doc_filter)
                )
            ]
        )

    retrieved_results = vectorstore.similarity_search_with_score(
        query,
        k=k,
        filter=filter_obj
    )
    
    if not retrieved_results:
        filter_msg = f" in '{doc_filter}'" if doc_filter and doc_filter != "All Documents" else ""
        return f"No relevant context found{filter_msg} to answer this question.", [], []
    
    # Format context for strict grounding
    context_blocks = []
    matched_images = []
    seen_images = set()

    for idx, (doc, score) in enumerate(retrieved_results, start=1):
        source = doc.metadata.get("source", "Unknown Document")
        page = doc.metadata.get("page", 1)
        is_table = doc.metadata.get("is_table", False)
        table_html = doc.metadata.get("table_html", "")
        
        table_repr = f"\n[Table HTML Structure]:\n{table_html}" if (table_html and table_html not in doc.page_content) else ""
        context_blocks.append(
            f"=== Document Excerpt {idx} (File: {source} | Page: {page}) ===\n{doc.page_content.strip()}{table_repr}"
        )
        
        doc_images = doc.metadata.get("images", [])
        if isinstance(doc_images, list):
            for img_p in doc_images:
                if img_p and img_p not in seen_images and os.path.exists(img_p):
                    seen_images.add(img_p)
                    matched_images.append({
                        "path": img_p,
                        "source": source,
                        "page": page,
                        "score": score
                    })
    
    formatted_context = "\n\n--------------------\n\n".join(context_blocks)
    
    prompt_template = ChatPromptTemplate.from_template("""You are an intelligent, highly helpful AI document assistant analyzing content extracted from PDF documents (including text, data tables, metrics, bullet points, and figures).

Your goal is to answer the user's question accurately, thoroughly, and factually based on the provided Context.

GUIDELINES:
1. Carefully analyze all the provided Document Excerpts below. Extract all relevant facts, numbers, metrics, explanations, or data points that address the user's question.
2. IF THE ANSWER INVOLVES TABULAR DATA, METRICS, COMPARISONS, OR STRUCTURED VALUES:
   - Synthesize and present the information in a clean, beautifully formatted Markdown Table (`| Column 1 | Column 2 | ...`) with clear headers and aligned rows.
3. Cite the relevant source file and page numbers when stating facts (e.g., "[Page 1]").
4. Be comprehensive and direct: Even if the information is presented in tables, lists, or headers, interpret and explain it clearly to answer the user's question.
5. Only if the provided context contains absolutely no relevant information to answer the question, state:
   "Based on the provided PDF documents, I could not find information to answer this question."

Context:
{context}

User Question:
{question}

Answer:""")

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=clean_gemini_key,
        temperature=0.2
    )
    
    rag_chain = prompt_template | llm | StrOutputParser()
    answer = rag_chain.invoke({
        "context": formatted_context,
        "question": query
    })
    
    return answer, retrieved_results, matched_images


# ------------------------------------------------------------------------------
# Main Streamlit Application
# ------------------------------------------------------------------------------
def main():
    # Reload environment variables from .env
    load_dotenv(override=True)

    st.set_page_config(
        page_title="PDF Q&A Assistant",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom Styling
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(90deg, #4F46E5 0%, #06B6D4 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1rem;
            color: #6B7280;
            margin-bottom: 1.5rem;
        }
        .stat-card {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 12px 16px;
            margin-bottom: 10px;
        }
        .source-box {
            border-left: 3px solid #6366F1;
            background-color: #F8FAFC;
            padding: 10px 14px;
            border-radius: 0 8px 8px 0;
            margin-top: 8px;
            margin-bottom: 8px;
            font-size: 0.9rem;
        }
        .table-badge {
            background-color: #E0E7FF;
            color: #3730A3;
            font-weight: 600;
            font-size: 0.75rem;
            padding: 2px 6px;
            border-radius: 4px;
        }
    </style>
    """, unsafe_allow_html=True)

    # Read credentials & configuration parameters securely from .env
    gemini_api_key = sanitize_key(os.getenv("GEMINI_API_KEY")) or sanitize_key(os.getenv("GOOGLE_API_KEY"))
    qdrant_url = (os.getenv("QDRANT_URL") or "http://localhost:6333").strip().rstrip("/")
    qdrant_api_key = sanitize_key(os.getenv("QDRANT_API_KEY"))
    collection_name = (os.getenv("QDRANT_COLLECTION_NAME") or "pdf_rag_collection").strip()

    # Model & Retrieval Defaults (read from .env or optimized defaults)
    embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    top_k = int(os.getenv("TOP_K_CHUNKS", "6"))
    chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))

    # Session State Initialization
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "indexed_files" not in st.session_state:
        st.session_state.indexed_files = []
    if "last_retrieved_docs" not in st.session_state:
        st.session_state.last_retrieved_docs = []

    # --------------------------------------------------------------------------
    # Sidebar (Clean & Minimalist - No Configuration / Parameters Shown)
    # --------------------------------------------------------------------------
    with st.sidebar:
        st.image("https://img.icons8.com/clouds/100/database.png", width=70)
        st.title("📚 PDF Assistant")
        st.caption("AI-powered Document Intelligence with Text, Tables & Images.")
        
        st.markdown("---")
        st.subheader("📊 Document Database")
        if qdrant_url:
            try:
                q_client = get_qdrant_client(qdrant_url, qdrant_api_key)
                collections_info = q_client.get_collections()
                collection_names = [c.name for c in collections_info.collections]
                
                if collection_name in collection_names:
                    count_info = q_client.count(collection_name=collection_name)
                    st.success(f"🟢 **Database Connected**\n\n**{count_info.count}** document chunks indexed.")
                else:
                    st.info("🟡 **Database Ready**\n\nUpload your first PDF to begin.")
                    
                if st.button("🗑️ Reset Database", help="Deletes all indexed document data"):
                    if collection_name in collection_names:
                        q_client.delete_collection(collection_name=collection_name)
                        st.session_state.indexed_files = []
                        st.session_state.messages = []
                        st.session_state.last_retrieved_docs = []
                        st.success("Database reset successfully!")
                        st.rerun()
            except Exception as e:
                st.error(f"🔴 Database Connection Notice: {e}")

        st.markdown("---")
        st.markdown("""
        **💡 How it works:**
        1. **Upload** any PDF document in Tab 1.
        2. Text, tables & figures are automatically extracted and indexed.
        3. **Chat** and ask questions in Tab 2 to get grounded answers with citations.
        """)

    # --------------------------------------------------------------------------
    # Main App Header
    # --------------------------------------------------------------------------
    st.markdown('<div class="main-header">📚 PDF Q&A Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload your PDF documents, extract text, <b>tables</b> & <b>images</b>, and chat with AI.</div>', unsafe_allow_html=True)

    if not gemini_api_key:
        st.error("⚠️ **Configuration Required**: `GEMINI_API_KEY` is not set in your `.env` file. Please add your key to `.env`.")

    tab_upload, tab_chat = st.tabs(["📤 Upload & Index PDF", "💬 Chat & Ask Questions"])

    # --------------------------------------------------------------------------
    # TAB 1: PDF Upload, Asset Extraction & Ingestion
    # --------------------------------------------------------------------------
    with tab_upload:
        st.subheader("1. Upload and Ingest PDF Document")
        st.markdown("Select a PDF file. Text, tables, and images will be automatically extracted, embedded, and indexed for high-accuracy search.")
        
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"], key="pdf_uploader")
        
        if uploaded_file is not None:
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / (1024 * 1024):.2f} MB",
                "File type": uploaded_file.type
            }
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.json(file_details)
            
            with col2:
                st.info("✨ **Multi-Modal Document Extraction**\n\n• Structured Tables\n• Embedded Figures & Page Diagrams\n• Semantic Text Chunks")
            
            if st.button("🚀 Process & Ingest PDF", type="primary"):
                if not gemini_api_key:
                    st.error("❌ `GEMINI_API_KEY` is missing in your `.env` file.")
                elif not qdrant_url:
                    st.error("❌ `QDRANT_URL` is missing in your `.env` file.")
                else:
                    progress_container = st.container()
                    with progress_container:
                        status_placeholder = st.empty()
                        progress_bar = st.progress(0)
                        
                        try:
                            # Step 1: Save PDF to disk
                            status_placeholder.info("💾 Step 1/5: Saving PDF to local storage...")
                            save_path = UPLOAD_DIR / uploaded_file.name
                            with open(save_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            progress_bar.progress(0.15)
                            
                            # Step 2: Extract embedded images and page snapshots
                            status_placeholder.info("🖼️ Step 2/5: Extracting figures, diagrams & visual assets...")
                            extracted_assets = extract_pdf_assets(str(save_path))
                            total_images = sum(len(imgs) for imgs in extracted_assets.values())
                            progress_bar.progress(0.35)
                            
                            # Step 3: Parse and extract tables & chunks with Unstructured
                            status_placeholder.info("📄 Step 3/5: Parsing text and extracting table structures...")
                            start_time = time.time()
                            chunks = process_pdf_with_unstructured(
                                file_path=str(save_path),
                                chunk_size=chunk_size,
                                chunk_overlap=200,
                                page_assets=extracted_assets
                            )
                            chunk_time = time.time() - start_time
                            table_chunks_count = sum(1 for c in chunks if c.metadata.get("is_table"))
                            progress_bar.progress(0.55)
                            
                            if not chunks:
                                st.error("No text content could be extracted from this PDF.")
                            else:
                                st.success(f"✅ Extracted **{len(chunks)} chunks** (including **{table_chunks_count} table structures**) and **{total_images} visual figures** in {chunk_time:.2f}s!")
                                
                                # Step 4: Connect to Qdrant & initialize vector store
                                status_placeholder.info("🧠 Step 4/5: Initializing embeddings & document index...")
                                q_client = get_qdrant_client(qdrant_url, qdrant_api_key)
                                vectorstore = get_vector_store(
                                    q_client,
                                    collection_name,
                                    gemini_api_key,
                                    embedding_model_name=embedding_model
                                )
                                progress_bar.progress(0.70)
                                
                                # Step 5: Index chunks to Qdrant
                                status_placeholder.info("💾 Step 5/5: Generating embeddings and indexing...")
                                indexed_count = index_documents_to_qdrant(
                                    documents=chunks,
                                    vectorstore=vectorstore,
                                    batch_size=15,
                                    progress_bar=progress_bar,
                                    status_text=status_placeholder
                                )
                                
                                progress_bar.progress(1.0)
                                status_placeholder.success(f"🎉 **Success!** Successfully indexed **{indexed_count} chunks** from `{uploaded_file.name}`.")
                                
                                if uploaded_file.name not in st.session_state.indexed_files:
                                    st.session_state.indexed_files.append(uploaded_file.name)
                                
                                # Visual Asset Gallery
                                if total_images > 0:
                                    with st.expander(f"🖼️ View Extracted Figures & Page Snapshots ({total_images} images)", expanded=False):
                                        cols = st.columns(3)
                                        img_idx = 0
                                        for p_num, img_list in extracted_assets.items():
                                            for img_file in img_list:
                                                with cols[img_idx % 3]:
                                                    st.image(img_file, caption=f"Page {p_num}: {Path(img_file).name}", use_container_width=True)
                                                img_idx += 1
                                
                                # Chunk preview expander
                                with st.expander("🔍 Preview Sample Extracted Chunks & Tables", expanded=False):
                                    for idx, chk in enumerate(chunks[:4], 1):
                                        badge = " [TABLE CHUNK]" if chk.metadata.get("is_table") else ""
                                        st.markdown(f"**Chunk {idx} (Page {chk.metadata.get('page', 1)}){badge} — {chk.metadata.get('char_count', 0)} characters:**")
                                        st.code(chk.page_content, language="text")
                                        if chk.metadata.get("table_html"):
                                            st.markdown("**Rendered Table Preview:**")
                                            st.markdown(chk.metadata["table_html"], unsafe_allow_html=True)
                                        st.caption(f"Metadata: {chk.metadata}")
                                        st.divider()
                                        
                        except PermissionError as p_err:
                            st.error(f"🔒 **Authentication / Permission Error**: {str(p_err)}")
                        except Exception as err:
                            err_str = str(err)
                            if "403" in err_str or "forbidden" in err_str.lower():
                                st.error(
                                    "🔒 **Database 403 Forbidden Error**: Access denied. "
                                    "Please check your `QDRANT_API_KEY` in `.env`."
                                )
                            else:
                                st.error(f"❌ Ingestion Error: {err_str}")
                            logger.error(f"Ingestion failed: {err}", exc_info=True)

    # --------------------------------------------------------------------------
    # TAB 2: Q&A Chat & Similarity Search with Tables & Images
    # --------------------------------------------------------------------------
    with tab_chat:
        st.subheader("2. Ask Questions from Uploaded PDF")
        
        # Discover all indexed documents in the active Qdrant collection
        discovered_sources = []
        if qdrant_url:
            try:
                temp_client = get_qdrant_client(qdrant_url, qdrant_api_key)
                discovered_sources = get_indexed_sources(temp_client, collection_name)
            except Exception:
                pass
        
        col_scope, col_clear = st.columns([4, 1])
        with col_scope:
            scope_choices = ["🌐 All Documents (Search across all uploaded PDFs)"] + [f"📄 {s}" for s in discovered_sources]
            selected_scope = st.selectbox(
                "🎯 Search Scope",
                options=scope_choices,
                index=0,
                help="Choose whether to search across ALL uploaded PDFs or restrict your query to a specific document."
            )
            doc_filter = None if selected_scope.startswith("🌐") else selected_scope.replace("📄 ", "").strip()
        with col_clear:
            st.write("")
            if st.button("🧹 Clear Chat", key="clear_chat_btn", help="Clear chat messages"):
                st.session_state.messages = []
                st.session_state.last_retrieved_docs = []
                st.rerun()
        
        # Display Chat History
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Render attached images if assistant
                if message.get("images"):
                    st.markdown("**🖼️ Extracted Figures & Visuals from Context:**")
                    img_cols = st.columns(min(len(message["images"]), 3))
                    for idx, img_info in enumerate(message["images"]):
                        with img_cols[idx % 3]:
                            st.image(
                                img_info["path"],
                                caption=f"From {img_info.get('source')} (Page {img_info.get('page')})",
                                use_container_width=True
                            )
                
                if "sources" in message and message["sources"]:
                    with st.expander("📚 View Retrieved Sources & Tables", expanded=False):
                        for src in message["sources"]:
                            table_badge = '<span class="table-badge">📊 TABLE DATA</span> ' if src.get("is_table") else ''
                            st.markdown(f"""
                            <div class="source-box">
                                {table_badge}<b>Source:</b> {src.get('source')} | <b>Page:</b> {src.get('page')} | <b>Similarity Score:</b> {src.get('score', 0):.4f}<br>
                                <span style="color: #374151;">{src.get('content')}</span>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            if src.get("table_html"):
                                st.markdown("**Rendered Table Structure:**")
                                st.markdown(src["table_html"], unsafe_allow_html=True)

        # Chat Input Box
        scope_label = f"in '{doc_filter}'" if doc_filter else "across all PDFs"
        if user_prompt := st.chat_input(f"Ask any question {scope_label} (tables, metrics, images, summaries)..."):
            if not gemini_api_key:
                st.error("❌ `GEMINI_API_KEY` is not set in `.env`.")
            elif not qdrant_url:
                st.error("❌ `QDRANT_URL` is not set in `.env`.")
            else:
                st.session_state.messages.append({"role": "user", "content": user_prompt})
                with st.chat_message("user"):
                    st.markdown(user_prompt)
                
                with st.chat_message("assistant"):
                    with st.spinner(f"🔍 Searching {scope_label} & synthesizing answer..."):
                        try:
                            q_client = get_qdrant_client(qdrant_url, qdrant_api_key)
                            vectorstore = get_vector_store(
                                q_client,
                                collection_name,
                                gemini_api_key,
                                embedding_model_name=embedding_model
                            )
                            
                            answer, retrieved_results, matched_images = generate_rag_response(
                                query=user_prompt,
                                vectorstore=vectorstore,
                                gemini_api_key=gemini_api_key,
                                model_name=model_name,
                                k=top_k,
                                doc_filter=doc_filter
                            )
                            
                            # Render Answer (Markdown tables, lists, text)
                            st.markdown(answer)
                            
                            # Render Relevant Extracted Figures & Images
                            if matched_images:
                                st.markdown("**🖼️ Extracted Figures & Visuals from Matching Pages:**")
                                img_cols = st.columns(min(len(matched_images), 3))
                                for idx, img_info in enumerate(matched_images):
                                    with img_cols[idx % 3]:
                                        st.image(
                                            img_info["path"],
                                            caption=f"From {img_info.get('source')} (Page {img_info.get('page')})",
                                            use_container_width=True
                                        )
                            
                            # Format sources data
                            sources_data = []
                            for doc, score in retrieved_results:
                                sources_data.append({
                                    "source": doc.metadata.get("source", "Document"),
                                    "page": doc.metadata.get("page", "N/A"),
                                    "score": score,
                                    "is_table": doc.metadata.get("is_table", False),
                                    "table_html": doc.metadata.get("table_html", ""),
                                    "content": doc.page_content[:400] + ("..." if len(doc.page_content) > 400 else "")
                                })
                            
                            # Render Sources Expander
                            if sources_data:
                                with st.expander("📚 View Retrieved Sources & Tables", expanded=True):
                                    for src in sources_data:
                                        table_badge = '<span class="table-badge">📊 TABLE DATA</span> ' if src.get("is_table") else ''
                                        st.markdown(f"""
                                        <div class="source-box">
                                            {table_badge}<b>Source:</b> {src.get('source')} | <b>Page:</b> {src.get('page')} | <b>Similarity Score:</b> {src.get('score', 0):.4f}<br>
                                            <span style="color: #374151;">{src.get('content')}</span>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                        if src.get("table_html"):
                                            st.markdown("**Rendered Table Structure:**")
                                            st.markdown(src["table_html"], unsafe_allow_html=True)
                            
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": answer,
                                "sources": sources_data,
                                "images": matched_images
                            })
                            
                        except Exception as e:
                            err_str = str(e)
                            if "403" in err_str or "forbidden" in err_str.lower():
                                error_msg = "🔒 **Database 403 Forbidden Error**: Access denied. Please check your QDRANT_API_KEY in .env."
                            else:
                                error_msg = f"❌ Error answering question: {err_str}"
                            st.error(error_msg)
                            logger.error(f"Chat generation error: {e}", exc_info=True)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})


if __name__ == "__main__":
    main()
