"""
Case 4: RAG (Retrieval-Augmented Generation) chatbot.

Concept:
- Load documents, split them into chunks, and create embeddings.
- Store embeddings in a vector store (e.g., FAISS, Chroma).
- At query time: retrieve relevant chunks and inject them into the LLM prompt.
- The LLM answers based on both its training and the retrieved context.

When to use:
- You have a knowledge base (docs, PDFs, code, etc.) that the LLM wasn't trained on.
- You want factual, grounded answers from your own data.
- You don't need the LLM to decide when to retrieve (it always retrieves).
"""

from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from config import get_llm_config, make_chat_llm


def build_rag_chatbot():
    """Build a RAG chatbot with vector store retrieval."""
    cfg = get_llm_config()

    # Load documents
    loader = TextLoader("data/sample_docs.txt", encoding="utf-8")
    documents = loader.load()

    # Split documents into chunks (for better retrieval granularity)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,  # Small chunks for demo
        chunk_overlap=50,
    )
    chunks = text_splitter.split_documents(documents)

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(api_key=cfg.api_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Create a retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})  # Top 2 chunks

    # Create the RAG chain
    llm = make_chat_llm(temperature=0.1)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff",  # Simple: stuff all retrieved docs into prompt
    )

    return qa_chain


def chat_loop() -> None:
    qa_chain = build_rag_chatbot()
    print("RAG chatbot. Type 'exit' or 'quit' to stop.")
    print("Ask questions about LangChain, LangGraph, RAG, or Agentic RAG!")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        # The chain automatically retrieves relevant chunks and generates answer
        result = qa_chain.invoke({"query": user_input})
        answer = result["result"]
        sources = result.get("source_documents", [])

        print(f"\nAssistant: {answer}")

        # Optionally show which chunks were retrieved
        if sources:
            print("\n[Retrieved from documents]")


if __name__ == "__main__":
    chat_loop()


