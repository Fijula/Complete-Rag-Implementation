"""
Case 8: RAG-Fusion chatbot.

Concept:
- Generates multiple query variations from the original query.
- Retrieves documents for each query variation.
- Ranks and combines results from all queries.
- Provides more comprehensive retrieval coverage.

When to use:
- You have ambiguous or complex queries.
- You need comprehensive document coverage.
- You want multiple perspectives on the same query.
- Quality over speed.
"""

from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from config import get_llm_config, make_chat_llm


def build_rag_fusion_chatbot():
    """Build a RAG-Fusion chatbot with multi-query retrieval."""
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

    # Query generation prompt
    query_generation_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Generate 3 different query variations for the given question. "
                "Each variation should approach the question from a different angle "
                "or use different keywords. Return only the queries, one per line.",
            ),
            ("human", "Original query: {query}\n\nGenerate 3 query variations:"),
        ]
    )

    # Final answer synthesis prompt
    synthesis_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Synthesize a comprehensive answer "
                "based on the retrieved information from multiple queries. "
                "Combine insights from all retrieved documents to provide a complete answer.",
            ),
            (
                "human",
                "Original query: {query}\n\n"
                "Retrieved information:\n{retrieved_info}\n\n"
                "Provide a comprehensive answer:",
            ),
        ]
    )

    return {
        "llm": llm,
        "retriever": retriever,
        "query_generation_prompt": query_generation_prompt,
        "synthesis_prompt": synthesis_prompt,
    }


def generate_query_variations(llm, query_generation_prompt, original_query: str) -> list[str]:
    """Generate multiple query variations."""
    response = llm.invoke(query_generation_prompt.format_messages(query=original_query))
    queries = [
        q.strip()
        for q in response.content.strip().split("\n")
        if q.strip() and not q.strip().startswith("#")
    ]
    # Include original query
    queries = [original_query] + queries[:3]  # Original + up to 3 variations
    return queries


def retrieve_for_queries(retriever, queries: list[str]) -> list:
    """Retrieve documents for each query and combine results."""
    all_docs = []
    seen_content = set()

    for query in queries:
        docs = retriever.get_relevant_documents(query)
        for doc in docs:
            # Deduplicate by content
            if doc.page_content not in seen_content:
                all_docs.append(doc)
                seen_content.add(doc.page_content)

    return all_docs


def chat_loop() -> None:
    components = build_rag_fusion_chatbot()
    llm = components["llm"]
    retriever = components["retriever"]
    query_generation_prompt = components["query_generation_prompt"]
    synthesis_prompt = components["synthesis_prompt"]

    print("RAG-Fusion chatbot. Type 'exit' or 'quit' to stop.")
    print("The model will generate multiple query variations for comprehensive retrieval!")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        # Step 1: Generate query variations
        print("\n[Generating query variations...]")
        queries = generate_query_variations(llm, query_generation_prompt, user_input)
        print(f"[Generated {len(queries)} query variations]")

        # Step 2: Retrieve documents for each query
        print("[Retrieving documents for each query variation...]")
        all_docs = retrieve_for_queries(retriever, queries)

        if not all_docs:
            print("[No documents found. Answering from general knowledge...]")
            direct_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "You are a helpful assistant."),
                    ("human", "{query}"),
                ]
            )
            response = llm.invoke(direct_prompt.format_messages(query=user_input))
            answer = response.content
            print(f"\nAssistant: {answer}")
            continue

        # Step 3: Combine retrieved information
        retrieved_info = "\n\n".join(
            [f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(all_docs[:5])]
        )

        # Step 4: Synthesize final answer
        print("[Synthesizing comprehensive answer from all retrieved documents...]")
        response = llm.invoke(
            synthesis_prompt.format_messages(
                query=user_input, retrieved_info=retrieved_info
            )
        )
        answer = response.content

        print(f"\nAssistant: {answer}")
        print(f"\n[Retrieved {len(all_docs)} unique documents from {len(queries)} query variations]")


if __name__ == "__main__":
    chat_loop()

