"""
Case 6: Self-RAG (Self-Retrieval Augmented Generation) chatbot.

Concept:
- The model evaluates its own responses and decides if retrieval is needed.
- If the model is not confident, it retrieves relevant documents.
- Can retrieve multiple times and refine answers.
- Self-corrects and improves response quality.

When to use:
- You need high-quality, self-validated answers.
- You want adaptive retrieval based on confidence.
- Quality is more important than speed.
"""

from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from config import get_llm_config, make_chat_llm


def build_self_rag_chatbot():
    """Build a Self-RAG chatbot that evaluates and retrieves adaptively."""
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

    # Create RAG chain for retrieval
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff",
    )

    # Evaluation prompt to check if retrieval is needed
    evaluation_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an evaluator. Given a user query, determine if you need "
                "to retrieve documents to answer it accurately. "
                "Respond with 'YES' if retrieval is needed, 'NO' if you can answer "
                "from general knowledge. "
                "Retrieval is needed for: specific facts, technical details, "
                "information about LangChain/LangGraph/RAG, or domain-specific knowledge.",
            ),
            ("human", "Query: {query}\n\nDo you need to retrieve documents? (YES/NO)"),
        ]
    )

    return {
        "llm": llm,
        "qa_chain": qa_chain,
        "evaluation_prompt": evaluation_prompt,
        "retriever": retriever,
    }


def evaluate_need_retrieval(llm, evaluation_prompt, query: str) -> bool:
    """Evaluate if retrieval is needed for the query."""
    response = llm.invoke(evaluation_prompt.format_messages(query=query))
    answer = response.content.strip().upper()
    return "YES" in answer


def chat_loop() -> None:
    components = build_self_rag_chatbot()
    llm = components["llm"]
    qa_chain = components["qa_chain"]
    evaluation_prompt = components["evaluation_prompt"]

    print("Self-RAG chatbot. Type 'exit' or 'quit' to stop.")
    print("The model will evaluate if retrieval is needed for each query!")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        # Step 1: Evaluate if retrieval is needed
        print("\n[Evaluating if retrieval is needed...]")
        needs_retrieval = evaluate_need_retrieval(llm, evaluation_prompt, user_input)

        if needs_retrieval:
            print("[Retrieval needed. Searching documents...]")
            # Step 2: Retrieve and generate answer
            result = qa_chain.invoke({"query": user_input})
            answer = result["result"]
            sources = result.get("source_documents", [])

            # Step 3: Self-evaluate the answer quality
            quality_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "Evaluate if the answer is complete and accurate. "
                        "Respond with 'GOOD' if the answer is satisfactory, "
                        "'NEEDS_IMPROVEMENT' if it needs more information.",
                    ),
                    (
                        "human",
                        "Query: {query}\nAnswer: {answer}\n\nIs this answer good? "
                        "(GOOD/NEEDS_IMPROVEMENT)",
                    ),
                ]
            )

            quality_response = llm.invoke(
                quality_prompt.format_messages(query=user_input, answer=answer)
            )
            quality = quality_response.content.strip().upper()

            if "NEEDS_IMPROVEMENT" in quality:
                print("[Answer needs improvement. Retrieving more documents...]")
                # Retrieve more documents and refine
                result = qa_chain.invoke({"query": user_input})
                answer = result["result"]
                print(f"\nAssistant: {answer} [Refined with additional retrieval]")
            else:
                print(f"\nAssistant: {answer} [Retrieved and validated]")
        else:
            # Step 2: Answer directly without retrieval
            print("[No retrieval needed. Answering from general knowledge...]")
            direct_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a helpful assistant. Answer the question directly.",
                    ),
                    ("human", "{query}"),
                ]
            )
            response = llm.invoke(direct_prompt.format_messages(query=user_input))
            answer = response.content
            print(f"\nAssistant: {answer} [Direct answer, no retrieval]")


if __name__ == "__main__":
    chat_loop()

