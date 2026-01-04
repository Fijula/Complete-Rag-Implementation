"""
Case 1: LangChain-based chatbot.

Concept:
- You build a *chain* that wires together: system prompt + memory + LLM.
- LangChain manages prompt templating and conversation history for you.

When to use:
- You want composability (prompts, tools, memory) without managing low-level details.
- You may later add tools, RAG, or other components but don't need graphs yet.
"""

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from config import make_chat_llm


def build_langchain_chatbot() -> ConversationChain:
    llm = make_chat_llm(temperature=0.3)

    # Prompt template: system + conversation history + user input
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful LangChain-based chatbot. "
                "Explain your reasoning briefly and clearly.",
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    memory = ConversationBufferMemory(return_messages=True)

    chain = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=prompt,
        verbose=False,
    )
    return chain


def chat_loop() -> None:
    chain = build_langchain_chatbot()
    print("LangChain chatbot. Type 'exit' or 'quit' to stop.")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        # Here we call the chain with just the latest input; memory handles context.
        response = chain.invoke({"input": user_input})
        print(f"Assistant: {response['response']}")


if __name__ == "__main__":
    chat_loop()



