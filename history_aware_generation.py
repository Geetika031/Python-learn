import os
import sys
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_voyageai import VoyageAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables (.env)
load_dotenv()


def get_groq_llm(model_name="openai/gpt-oss-120b", temperature=0):
    """Initialize and return the Groq Chat model."""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key or groq_api_key == "your_groq_api_key_here":
        raise ValueError(
            "GROQ_API_KEY is not set or has a placeholder value in the .env file. "
            "Please add your valid Groq API key to .env."
        )
    return ChatGroq(model=model_name, temperature=temperature, groq_api_key=groq_api_key)


def load_vector_store(persist_directory="db/chroma_db"):
    """Load the persisted ChromaDB vector store using Voyage AI embeddings."""
    if not os.path.exists(persist_directory):
        raise FileNotFoundError(
            f"Vector store not found at '{persist_directory}'. "
            "Please run 'python ingestion_pipeline.py' first to ingest documents from the docs folder."
        )

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


def create_query_rewriter_chain(model_name="openai/gpt-oss-120b"):
    """
    Build a query rewriter chain that takes past user questions and a follow-up question,
    and reformulates the follow-up question into a standalone question.
    """
    rephrase_template = """You are an expert query contextualizer.
Given a list of past user questions and a new follow-up question, rewrite the follow-up question so that it becomes a standalone, fully self-contained question that can be understood without the conversation history.

Rules:
1. Replace pronouns (it, they, them, this, that, etc.) and implicit references with the specific entities or subjects mentioned in the past questions.
2. Preserve the original intent and meaning of the follow-up question.
3. Do NOT answer the question. Only rewrite it.
4. If the follow-up question is already self-contained or does not depend on past questions, return it unchanged.
5. Return ONLY the rewritten standalone question without any preamble, explanation, or quotes.

Past Question History:
{history}

Follow-up Question:
{question}

Standalone Rewritten Question:"""

    prompt = ChatPromptTemplate.from_template(rephrase_template)
    llm = get_groq_llm(model_name=model_name, temperature=0)
    rewriter_chain = prompt | llm | StrOutputParser()
    return rewriter_chain


def rewrite_query(query: str, question_history: list, rewriter_chain=None, model_name="openai/gpt-oss-120b") -> str:
    """
    Rewrite the user query using the history of previous questions.
    If the question history is empty, the original query is returned directly.
    """
    if not question_history:
        return query.strip()

    if rewriter_chain is None:
        rewriter_chain = create_query_rewriter_chain(model_name=model_name)

    # Format question history list as numbered lines
    formatted_history = "\n".join([f"{i+1}. {q}" for i, q in enumerate(question_history)])
    
    rewritten_query = rewriter_chain.invoke({
        "history": formatted_history,
        "question": query
    })
    
    return rewritten_query.strip()


def create_rag_chain(vectorstore, model_name="openai/gpt-oss-120b", k=3):
    """Build a grounded RAG QA chain with ChromaDB retriever and Groq LLM."""
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    answer_template = """You are a helpful and strict domain assistant.

CRITICAL INSTRUCTIONS:
1. Answer the question relying ONLY on the factual information contained in the Context below.
2. DO NOT use prior knowledge, outside information, or ungrounded assumptions.
3. If the context does not contain enough information to answer the question, state:
   "Based on the provided documents, I cannot answer this question as the required information is not found in the context."
4. Every statement in your answer must be supported by the text in the Context.

Context:
{context}

Question:
{question}

Accurate Answer:"""

    prompt = ChatPromptTemplate.from_template(answer_template)
    llm = get_groq_llm(model_name=model_name, temperature=0)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever


class HistoryAwareRAG:
    """
    Manages history of user questions, query rewriting, retrieval, and response generation.
    """
    def __init__(self, persist_directory="db/chroma_db", model_name="openai/gpt-oss-120b", k=3):
        self.question_history = []  # Array to keep track of user questions
        self.model_name = model_name
        self.k = k
        self.vectorstore = load_vector_store(persist_directory)
        self.rewriter_chain = create_query_rewriter_chain(model_name=model_name)
        self.rag_chain, self.retriever = create_rag_chain(self.vectorstore, model_name=model_name, k=k)

    def process_query(self, user_query: str) -> dict:
        """
        Processes a user query with the following steps:
        1. Contextualize / Rewrite the query based on the history array.
        2. Retrieve relevant document chunks using the rewritten query.
        3. Generate a grounded answer using the context.
        4. Append the original query to the history array.
        """
        print("\n" + "=" * 75)
        print(f"📥 Current User Query : {user_query}")
        print(f"📜 Question History   : {self.question_history}")
        print("=" * 75)

        # Step 1: Rewrite Query based on history array
        if self.question_history:
            rewritten_query = rewrite_query(
                query=user_query,
                question_history=self.question_history,
                rewriter_chain=self.rewriter_chain
            )
            print(f"\n🔄 Rewritten Query    : {rewritten_query}")
        else:
            rewritten_query = user_query
            print("\n🔄 Rewritten Query    : (First question, no rewrite needed)")

        # Step 2: Retrieve relevant documents using rewritten query
        print(f"\n🔍 Retrieving chunks for: '{rewritten_query}'...")
        retrieved_docs = self.retriever.invoke(rewritten_query)

        for idx, doc in enumerate(retrieved_docs, start=1):
            source = doc.metadata.get("source", "Unknown file")
            print(f"\n--- Retrieved Chunk {idx} (Source: {source}) ---")
            print(doc.page_content.strip()[:250] + ("..." if len(doc.page_content) > 250 else ""))

        # Step 3: Generate Answer
        print("\n🤖 Generating strictly grounded answer...")
        answer = self.rag_chain.invoke(rewritten_query)

        print("\n" + "-" * 75)
        print(f"💡 Final Answer:\n{answer}")
        print("-" * 75)

        # Step 4: Add original question to the history array
        self.question_history.append(user_query)
        print(f"✅ Updated Question History Array: {self.question_history}\n")

        return {
            "original_query": user_query,
            "rewritten_query": rewritten_query,
            "answer": answer,
            "history": list(self.question_history)
        }

    def clear_history(self):
        """Reset the question history array."""
        self.question_history.clear()
        print("🧹 Question history cleared.")


def run_demo():
    """Demonstrates the exact multi-turn question rewriting scenario requested."""
    print("\n" + "#" * 75)
    print("### RUNNING DEMO: HISTORY-AWARE QUERY REWRITING ###")
    print("#" * 75)

    rag = HistoryAwareRAG()

    # Turn 1
    query_1 = "when was the microsoft released windows."
    rag.process_query(query_1)

    # Turn 2: Follow-up question with pronoun 'it'
    query_2 = "what does it do."
    rag.process_query(query_2)

    # Turn 3: Another follow-up question
    query_3 = "who was the CEO at that time?"
    rag.process_query(query_3)


def run_interactive():
    """Interactive loop for user to test custom multi-turn questions in real-time."""
    print("\n" + "#" * 75)
    print("### INTERACTIVE HISTORY-AWARE RAG CHAT ###")
    print("Type your questions below. Type 'exit' or 'quit' to stop, 'clear' to reset history.")
    print("#" * 75)

    rag = HistoryAwareRAG()

    while True:
        try:
            user_input = input("\nEnter your question: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("Goodbye!")
                break
            if user_input.lower() == "clear":
                rag.clear_history()
                continue

            rag.process_query(user_input)
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break


def main():
    # If '--interactive' is passed in sys.argv, run interactive mode; otherwise run demo
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        run_interactive()
    else:
        run_demo()


if __name__ == "__main__":
    main()
