"""
Case 2: LangGraph-based chatbot.

Concept:
- You define a *graph* of nodes (functions) with shared state.
- Each turn, the graph updates the state and decides the next step.
- This example is a minimal 2-node graph: (user_input) -> (chatbot).

When to use:
- You need explicit control flow: branching, looping, tool-calling, retries.
- You want strongly-typed, inspectable state over multiple steps.
"""

from typing import TypedDict

from langgraph.graph import StateGraph, END

from config import make_chat_llm


class ChatState(TypedDict):
    """Shared state in the LangGraph flow."""

    history: str  # simple concatenated transcript for this demo
    user_input: str
    assistant_response: str


def chatbot_node(state: ChatState) -> ChatState:
    """Single chatbot step using the shared history + new user input."""
    llm = make_chat_llm(temperature=0.3)

    prompt = (
        "You are a LangGraph-based chatbot.\n"
        "Conversation so far:\n"
        f"{state['history']}\n\n"
        f"User: {state['user_input']}\n"
        "Assistant:"
    )

    response = llm.invoke(prompt)
    answer = response.content

    new_history = (
        state["history"]
        + f"\nUser: {state['user_input']}\nAssistant: {answer}"
    )

    return {
        "history": new_history,
        "user_input": state["user_input"],
        "assistant_response": answer,
    }


def build_langgraph_chatbot():
    """Build a minimal LangGraph with one chatbot node."""
    graph = StateGraph(ChatState)

    # Add a single node that processes the chat turn.
    graph.add_node("chatbot", chatbot_node)

    # Entry point: always start at the chatbot node.
    graph.set_entry_point("chatbot")

    # After chatbot node runs, we end the graph execution for this turn.
    graph.add_edge("chatbot", END)

    return graph.compile()


def chat_loop() -> None:
    app = build_langgraph_chatbot()

    print("LangGraph chatbot. Type 'exit' or 'quit' to stop.")

    state: ChatState = {
        "history": "",
        "user_input": "",
        "assistant_response": "",
    }

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        state["user_input"] = user_input

        # Execute the graph once for this turn.
        state = app.invoke(state)
        print(f"Assistant: {state['assistant_response']}")


if __name__ == "__main__":
    chat_loop()



