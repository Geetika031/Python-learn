import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_voyageai import VoyageAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables (.env)
load_dotenv()

def load_vector_store(persist_directory="db/chroma_db"):
    """Load the persisted ChromaDB vector store using Voyage AI embeddings."""
    if not os.path.exists(persist_directory):
        raise FileNotFoundError(
            f"Vector store not found at '{persist_directory}'. "
            "Please run 'python ingestion_pipeline.py' first to ingest documents from the docs folder."
        )
    
    # Using Voyage AI embeddings
    embedding_model = VoyageAIEmbeddings(model="voyage-3")
    
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )
    return vectorstore

def format_docs(docs):
    """Format retrieved document chunks into a structured context string."""
    formatted_chunks = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unknown")
        formatted_chunks.append(
            f"[Document Chunk {i} | Source: {source}]\n{doc.page_content.strip()}"
        )
    return "\n\n".join(formatted_chunks)

def create_strict_rag_chain(vectorstore, model_name="openai/gpt-oss-120b", k=3):
    """
    Build a strictly grounded RAG chain that forces the Groq LLM to use ONLY
    the retrieved document context and prevents the use of prior knowledge.
    """
    # 1. Voyage AI-backed vector retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    
    # 2. Strict Grounding Prompt Template (No outside knowledge allowed)
    template = """You are a strict, closed-domain question-answering assistant.

CRITICAL INSTRUCTIONS:
1. Answer the user's question relying ONLY and EXCLUSIVELY on the factual information contained in the Context below.
2. DO NOT use any of your own prior knowledge, training data, outside information, or assumptions.
3. If the context does not explicitly contain enough information to answer the question, state exactly:
   "Based on the provided documents, I cannot answer this question as the required information is not found in the context."
4. Every fact, number, or statement in your answer must be directly supported by the text in the Context.

Context:
{context}

Question:
{question}

Accurate Answer:"""

    prompt = ChatPromptTemplate.from_template(template)
    
    # 3. Groq LLM with temperature=0 for deterministic, factual output
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key or groq_api_key == "your_groq_api_key_here":
        raise ValueError(
            "GROQ_API_KEY not set or still has placeholder value in .env file. "
            "Please add your valid Groq API key to .env."
        )
    
    llm = ChatGroq(model=model_name, temperature=0, groq_api_key=groq_api_key)
    
    # 4. Construct LCEL Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever

def answer_query(query: str, persist_directory="db/chroma_db", model_name="openai/gpt-oss-120b", k=3):
    """
    Retrieve document chunks from ChromaDB (using Voyage AI embeddings)
    and pass all retrieved data to Groq LLM to get a strictly grounded answer.
    """
    print("=" * 75)
    print(f"❓ User Query: {query}")
    print("=" * 75)
    
    vectorstore = load_vector_store(persist_directory)
    rag_chain, retriever = create_strict_rag_chain(vectorstore, model_name=model_name, k=k)
    
    # Step 1: Retrieve matching chunks
    print("\n🔍 Retrieving matching chunks from ChromaDB (via Voyage AI embeddings)...")
    retrieved_docs = retriever.invoke(query)
    
    if not retrieved_docs:
        print("❌ No matching documents found in vector store.")
        return "No documents found."
    
    for idx, doc in enumerate(retrieved_docs, start=1):
        source = doc.metadata.get("source", "Unknown file")
        print(f"\n--- Retrieved Chunk {idx} (Source: {source}) ---")
        print(doc.page_content.strip()[:300] + ("..." if len(doc.page_content) > 300 else ""))
    
    # Step 2: Pass all retrieved context to Groq LLM
    print("\n🤖 Sending retrieved context to Groq LLM for strictly grounded answer...")
    answer = rag_chain.invoke(query)
    
    print("\n" + "=" * 75)
    print("💡 Groq LLM Answer (Strictly grounded in docs):")
    print("=" * 75)
    print(answer)
    print("=" * 75 + "\n")
    
    return answer

def main():
    print("=== RAG Pipeline: Voyage AI Embeddings + Groq LLM QA ===\n")
    
    # ------------------------------------------------------------------
    # ✍️ WRITE YOUR QUESTION / QUERY HERE
    # ------------------------------------------------------------------
    # query = "What GPUs and AI chips does Nvidia produce?"
    
    # Other examples:
    # query = "What are the main products and services offered by Google?"
    query = "What are the key rockets and spacecraft built by SpaceX?"
    # query = "Tell me about Tesla electric vehicles and energy storage"
    # ------------------------------------------------------------------
    
    answer_query(query=query, k=3)

if __name__ == "__main__":
    main()
