"""
Case 5: Agentic RAG chatbot.

Concept:
- The LLM acts as an *agent* that can decide when to retrieve documents.
- The agent has access to tools (e.g., a retriever tool, a web search tool).
- The agent can plan: "I need to search for X, then retrieve Y, then answer."
- More autonomous than basic RAG: the agent decides what to retrieve and when.

When to use:
- You want the LLM to decide when retrieval is needed vs. using general knowledge.
- You have multiple data sources or tools the agent should choose from.
- You need multi-step reasoning: retrieve → analyze → retrieve more → answer.
- You want the agent to handle complex queries that require planning.
"""

from typing import TypedDict

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.retriever import create_retriever_tool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from config import get_llm_config, make_chat_llm


def build_agentic_rag_chatbot():
    """Build an agentic RAG chatbot with a retriever tool."""
    cfg = get_llm_config()

    # Load and prepare documents (same as basic RAG)
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

    # Create a tool that the agent can call
    retriever_tool = create_retriever_tool(
        retriever,
        "knowledge_base_search",
        "Search the knowledge base for information about LangChain, LangGraph, RAG, or Agentic RAG. "
        "Use this when you need factual information from the documents.",
    )

    # The agent has access to this tool and decides when to use it
    tools = [retriever_tool]

    # Create the agent
    llm = make_chat_llm(temperature=0.1)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful agentic RAG chatbot. "
                "You have access to a knowledge base search tool. "
                "Decide when to search the knowledge base vs. using your general knowledge. "
                "If the user asks about LangChain, LangGraph, RAG, or Agentic RAG, "
                "you should search the knowledge base. "
                "For general questions, you can answer directly.",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,  # Set to False to hide agent reasoning steps
        max_iterations=3,  # Limit agent steps
    )

    return agent_executor


def chat_loop() -> None:
    agent_executor = build_agentic_rag_chatbot()
    print("Agentic RAG chatbot. Type 'exit' or 'quit' to stop.")
    print("The agent will decide when to search the knowledge base!")

    chat_history = []

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        # The agent decides whether to use the retriever tool
        result = agent_executor.invoke(
            {
                "input": user_input,
                "chat_history": chat_history,
            }
        )

        answer = result["output"]
        print(f"\nAssistant: {answer}")

        # Update chat history
        chat_history.append(("human", user_input))
        chat_history.append(("assistant", answer))


if __name__ == "__main__":
    chat_loop()


