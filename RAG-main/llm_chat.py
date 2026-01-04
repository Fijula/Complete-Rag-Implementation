"""
Case 3: Direct LLM chatbot (no LangChain / LangGraph / RAG).

Concept:
- You call the LLM API directly via LangChain's ChatOpenAI wrapper, but
  you do not use chains, graphs, or retrieval.
- You manually manage the conversation history and prompt.

When to use:
- Quick prototypes or scripts.
- When you don't need tools, retrieval, or complex control flow.
- Great for understanding the *bare minimum* needed to chat with an LLM.
"""

from typing import List, Dict

from langchain_openai import ChatOpenAI

from config import get_llm_config


def make_raw_llm() -> ChatOpenAI:
    cfg = get_llm_config()
    return ChatOpenAI(api_key=cfg.api_key, model=cfg.model, temperature=0.3)


def chat_loop() -> None:
    llm = make_raw_llm()

    # Conversation history is a simple list of dicts with "role" and "content".
    history: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": "You are a friendly assistant. Be concise and helpful.",
        }
    ]

    print("Direct LLM chat. Type 'exit' or 'quit' to stop.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        history.append({"role": "user", "content": user_input})

        # Call the model with the full conversation so far.
        response = llm.invoke(history)
        assistant_message = response.content

        print(f"Assistant: {assistant_message}")
        history.append({"role": "assistant", "content": assistant_message})


if __name__ == "__main__":
    chat_loop()



