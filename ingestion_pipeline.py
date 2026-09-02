import os
import time
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_voyageai import VoyageAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path="docs"):
    """Load all text files from the docs directory"""
    print(f"Loading documents from {docs_path}...")
    
    # Check if docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your company files.")
    
    # Load all .txt files from the docs directory
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader
    )
    
    documents = loader.load()
    
    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your company documents.")
    
    for i, doc in enumerate(documents[:2]):  # Show first 2 documents
        print(f"\nDocument {i+1}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:100]}...")
        print(f"  metadata: {doc.metadata}")

    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    """Split documents into smaller chunks with overlap"""
    print("Splitting documents into chunks...")
    
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    
    chunks = text_splitter.split_documents(documents)
    
    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content)
            print("-" * 50)
        
        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")
    
    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db", batch_size=20, delay_seconds=20):
    """Create and persist ChromaDB vector store using Voyage AI embeddings with batching and retry handling"""
    print("Creating embeddings with Voyage AI and storing in ChromaDB...")
        
    embedding_model = VoyageAIEmbeddings(model="voyage-3")
    
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    total_chunks = len(chunks)
    print(f"--- Ingesting {total_chunks} chunks in batches of {batch_size} ---")
    
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total_chunks + batch_size - 1) // batch_size
        print(f"Processing batch {batch_num}/{total_batches} (chunks {i+1} to {min(i+batch_size, total_chunks)})...")
        
        retries = 5
        while retries > 0:
            try:
                vectorstore.add_documents(batch)
                break
            except Exception as e:
                print(f"⚠️ Rate limit or error encountered: {e}")
                retries -= 1
                if retries == 0:
                    raise e
                wait_time = 30
                print(f"Retrying in {wait_time}s... ({retries} retries left)")
                time.sleep(wait_time)
        
        if i + batch_size < total_chunks and delay_seconds > 0:
            time.sleep(delay_seconds)
            
    print(f"--- Finished creating vector store ---")
    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore

def main():
    """Main ingestion pipeline"""
    print("=== RAG Document Ingestion Pipeline (Voyage AI) ===\n")
    
    # Define paths
    docs_path = "docs"
    persistent_directory = "db/chroma_db"
    
    # Check if vector store already exists
    if os.path.exists(persistent_directory):
        print("✅ Vector store already exists. Loading existing vector store...")
        
        embedding_model = VoyageAIEmbeddings(model="voyage-3")
        vectorstore = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_model, 
            collection_metadata={"hnsw:space": "cosine"}
        )
        print(f"Loaded existing vector store with {vectorstore._collection.count()} documents")
        return vectorstore
    
    print("Persistent directory does not exist. Initializing vector store...\n")
    
    # Step 1: Load documents
    documents = load_documents(docs_path)  

    # Step 2: Split into chunks
    chunks = split_documents(documents)
    
    # Step 3: Create vector store
    vectorstore = create_vector_store(chunks, persistent_directory)
    
    print("\n✅ Ingestion complete! Your documents are now ready for queries.")
    return vectorstore

if __name__ == "__main__":
    main()
