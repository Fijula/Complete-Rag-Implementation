"""
Case 7: Adaptive RAG chatbot.

Concept:
- Analyzes query complexity first.
- Simple queries: direct LLM answer (no retrieval).
- Complex queries: retrieve documents and answer.
- Optimizes retrieval strategy based on query type.

When to use:
- You have a mix of simple and complex queries.
- You want performance optimization.
- You want to reduce unnecessary retrievals.
"""

from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from config import get_llm_config, make_chat_llm


def build_adaptive_rag_chatbot():
    """Build an Adaptive RAG chatbot that analyzes query complexity."""
    cfg = get_llm_config()

    # Load and prepare documents
    loader = TextLoader("data/sample_docs.txt", encoding="utf-8")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
    )
    chunks = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(api_key=cfg.api_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = make_chat_llm(temperature=0.1)

    # Create RAG chain for complex queries
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff",
    )

    # Query classification prompt
    classification_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Classify the query complexity. Respond with one word: "
                "'SIMPLE' for general knowledge questions (e.g., 'What is the capital of France?'), "
                "'COMPLEX' for questions requiring specific documents or technical details "
                "(e.g., 'How does LangGraph work?', 'Explain RAG architecture').",
            ),
            ("human", "Query: {query}\n\nClassification (SIMPLE/COMPLEX):"),
        ]
    )

    return {
        "llm": llm,
        "qa_chain": qa_chain,
        "classification_prompt": classification_prompt,
    }


def classify_query(llm, classification_prompt, query: str) -> str:
    """Classify query as SIMPLE or COMPLEX."""
    response = llm.invoke(classification_prompt.format_messages(query=query))
    classification = response.content.strip().upper()
    return "COMPLEX" if "COMPLEX" in classification else "SIMPLE"


def chat_loop() -> None:
    components = build_adaptive_rag_chatbot()
    llm = components["llm"]
    qa_chain = components["qa_chain"]
    classification_prompt = components["classification_prompt"]

    print("Adaptive RAG chatbot. Type 'exit' or 'quit' to stop.")
    print("The model will analyze query complexity and adapt retrieval strategy!")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        # Step 1: Classify query complexity
        print("\n[Analyzing query complexity...]")
        query_type = classify_query(llm, classification_prompt, user_input)

        if query_type == "COMPLEX":
            # Step 2: Complex query - retrieve and answer
            print("[Complex query detected. Retrieving documents...]")
            result = qa_chain.invoke({"query": user_input})
            answer = result["result"]
            sources = result.get("source_documents", [])
            print(f"\nAssistant: {answer}")
            print("[Retrieved from knowledge base]")
        else:
            # Step 2: Simple query - answer directly
            print("[Simple query detected. Answering directly...]")
            direct_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a helpful assistant. Answer the question directly "
                        "using your general knowledge.",
                    ),
                    ("human", "{query}"),
                ]
            )
            response = llm.invoke(direct_prompt.format_messages(query=user_input))
            answer = response.content
            print(f"\nAssistant: {answer}")
            print("[Direct answer, no retrieval needed]")


if __name__ == "__main__":
    chat_loop()

