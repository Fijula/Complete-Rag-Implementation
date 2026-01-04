## Multi-style Chatbot Examples (LangChain, LangGraph, LLM, RAG Variants)

This project demonstrates **8 different chatbot implementations** using (mostly) the same LLM model:

### Core Chatbot Types (Non-RAG):
- **Case 1 – LangChain**: A composable chain-based chatbot with memory.
- **Case 2 – LangGraph**: A graph-based chatbot with explicit state and control flow.
- **Case 3 – Direct LLM**: A minimal, raw LLM chat loop without frameworks.

### RAG Variants (5 types):
- **Case 4 – Basic RAG**: A Retrieval-Augmented Generation chatbot that always retrieves documents.
- **Case 5 – Agentic RAG**: An LLM agent that can decide when and how to retrieve documents.
- **Case 6 – Self-RAG**: Self-evaluating RAG that adaptively retrieves based on confidence.
- **Case 7 – Adaptive RAG**: Analyzes query complexity and adapts retrieval strategy.
- **Case 8 – RAG-Fusion**: Generates multiple query variations for comprehensive retrieval.

📖 **See [COMPARISON.md](COMPARISON.md) for detailed differences and when to use each type.**

All examples use Python and OpenAI-compatible models (through `langchain-openai`).

---

### 1. Setup

1. **Create & activate a virtual environment (recommended)**:

```bash
cd /Users/fijula/observability
python -m venv .venv
source .venv/bin/activate  # on macOS / Linux
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Configure your LLM API key** (example: OpenAI-compatible endpoint):

- Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_real_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

You can adjust the model as needed.

---

### 2. Files Overview

**Core Files:**
- `config.py`: Shared helpers to load environment variables and construct LLM clients.
- `data/sample_docs.txt`: Simple text corpus used for RAG examples.
- `COMPARISON.md`: Comprehensive comparison of all chatbot types and RAG variants.

**Core Chatbot Types:**
- `llm_chat.py`: Case 3 – direct LLM chat (no LangChain/graph/RAG).
- `langchain_chat.py`: Case 1 – LangChain chain with conversation memory.
- `langgraph_chat.py`: Case 2 – LangGraph state-machine-style chatbot.

**RAG Variants:**
- `rag_chat.py`: Case 4 – basic RAG (always retrieves).
- `agentic_rag_chat.py`: Case 5 – agentic RAG (LLM decides when to retrieve).
- `self_rag_chat.py`: Case 6 – self-RAG (self-evaluates and adaptively retrieves).
- `adaptive_rag_chat.py`: Case 7 – adaptive RAG (analyzes query complexity).
- `rag_fusion_chat.py`: Case 8 – RAG-fusion (multiple query variations).

Each script can be run independently.

---

### 3. Running Each Scenario

All commands assume you are in the project root (`/Users/fijula/observability`) and your virtual environment is active.

- **Case 3 – Direct LLM**:

```bash
python llm_chat.py
```

- **Case 1 – LangChain**:

```bash
python langchain_chat.py
```

- **Case 2 – LangGraph**:

```bash
python langgraph_chat.py
```

- **Case 4 – RAG**:

```bash
python rag_chat.py
```

- **Case 5 – Agentic RAG**:

```bash
python agentic_rag_chat.py
```

- **Case 6 – Self-RAG**:

```bash
python self_rag_chat.py
```

- **Case 7 – Adaptive RAG**:

```bash
python adaptive_rag_chat.py
```

- **Case 8 – RAG-Fusion**:

```bash
python rag_fusion_chat.py
```

Each script will open a simple REPL-style loop; type `exit` or `quit` to stop.

---

## 4. Detailed Explanation of Each Case

### Case 1: LangChain (`langchain_chat.py`)

**What it is:**
- A chatbot built using LangChain's `ConversationChain`.
- Uses LangChain's built-in memory management (`ConversationBufferMemory`).
- Composes prompts, LLM, and memory into a reusable chain.

**How it works:**
1. Creates a prompt template with system message, conversation history, and user input.
2. LangChain automatically manages conversation history in memory.
3. Each turn, the chain invokes the LLM with the full context.

**Key features:**
- Automatic conversation history management
- Composable components (prompts, memory, LLM)
- Easy to extend with tools, RAG, or other components

**When to use:**
- You want a simple, maintainable chatbot with memory.
- You plan to add more components (tools, RAG) later but don't need complex control flow.
- You prefer LangChain's abstractions over managing prompts/history manually.

**Code highlights:**
```python
# LangChain handles memory automatically
chain = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory(return_messages=True),
    prompt=prompt,
)
response = chain.invoke({"input": user_input})  # Memory is managed internally
```

---

### Case 2: LangGraph (`langgraph_chat.py`)

**What it is:**
- A chatbot built using LangGraph's state machine model.
- Defines a graph of nodes with explicit state transitions.
- Each node is a function that updates shared state.

**How it works:**
1. Defines a `ChatState` TypedDict to hold conversation state.
2. Creates graph nodes (e.g., `chatbot_node`) that process the state.
3. Defines edges between nodes to control flow.
4. Each turn, the graph executes and updates state.

**Key features:**
- Explicit control flow (branching, looping, conditional steps)
- Strongly-typed state that you can inspect and modify
- Can add multiple nodes (e.g., retrieval → analysis → response)
- Better for complex workflows with multiple steps

**When to use:**
- You need explicit control flow (e.g., "if this, then that").
- You want to add multiple processing steps (retrieval, validation, response).
- You need to handle errors, retries, or conditional logic.
- You want to visualize or debug the execution flow.

**Code highlights:**
```python
# Define state structure
class ChatState(TypedDict):
    history: str
    user_input: str
    assistant_response: str

# Build graph with nodes and edges
graph = StateGraph(ChatState)
graph.add_node("chatbot", chatbot_node)
graph.set_entry_point("chatbot")
graph.add_edge("chatbot", END)
```

---

### Case 3: Direct LLM (`llm_chat.py`)

**What it is:**
- A minimal chatbot using the OpenAI API directly (via `openai` package).
- No frameworks, no abstractions—just API calls and manual prompt management.

**How it works:**
1. Manually constructs prompts with conversation history.
2. Calls the OpenAI API directly.
3. Manually appends responses to history for next turn.

**Key features:**
- Minimal dependencies (just `openai` and `python-dotenv`)
- Full control over prompts and API calls
- No framework overhead
- Simple to understand and debug

**When to use:**
- You want maximum control and minimal dependencies.
- You don't need advanced features (memory management, tools, RAG).
- You're building a simple prototype or learning how LLMs work.
- You want to avoid framework abstractions.

**Code highlights:**
```python
# Manual prompt construction
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": user_input}
]

# Direct API call
response = client.chat.completions.create(
    model=cfg.model,
    messages=messages
)
```

---

### Case 4: RAG (`rag_chat.py`)

**What it is:**
- Retrieval-Augmented Generation: combines an LLM with a vector store of documents.
- At query time, retrieves relevant document chunks and injects them into the prompt.
- The LLM answers based on both its training and the retrieved context.

**How it works:**
1. Loads documents and splits them into chunks.
2. Creates embeddings for each chunk and stores them in a vector database (FAISS).
3. For each query:
   - Retrieves the top-k most relevant chunks (by similarity).
   - Injects those chunks into the LLM prompt as context.
   - LLM generates an answer grounded in the retrieved documents.

**Key features:**
- Answers are grounded in your own documents (not just LLM training data).
- Always retrieves relevant chunks (no decision-making by the LLM).
- Good for factual queries over a knowledge base.

**When to use:**
- You have a knowledge base (docs, PDFs, code, etc.) that the LLM wasn't trained on.
- You want factual, accurate answers from your own data.
- You don't need the LLM to decide when to retrieve (it always retrieves).
- You want to reduce hallucinations by grounding answers in documents.

**Code highlights:**
```python
# Create vector store from documents
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# RAG chain automatically retrieves and uses context
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
)
```

---

### Case 5: Agentic RAG (`agentic_rag_chat.py`)

**What it is:**
- An LLM agent that can decide when and how to retrieve documents.
- The agent has access to tools (e.g., a retriever tool) and decides when to use them.
- More autonomous than basic RAG: the agent plans and chooses actions.

**How it works:**
1. Sets up a retriever tool that the agent can call.
2. Creates an agent (using LangChain's agent framework) with access to the tool.
3. For each query:
   - The agent decides: "Do I need to search? Or can I answer directly?"
   - If it decides to search, it calls the retriever tool.
   - The agent can make multiple tool calls and reason about the results.
   - Finally generates an answer.

**Key features:**
- The LLM decides when retrieval is needed vs. using general knowledge.
- Can handle multi-step reasoning (retrieve → analyze → retrieve more → answer).
- Can be extended with multiple tools (web search, calculator, database, etc.).
- More flexible than basic RAG but also more complex.

**When to use:**
- You want the LLM to decide when retrieval is needed (not always retrieve).
- You have multiple data sources or tools the agent should choose from.
- You need complex queries that require planning and multi-step reasoning.
- You want the agent to handle both general questions and document-specific queries.

**Code highlights:**
```python
# Create a tool the agent can call
retriever_tool = create_retriever_tool(
    retriever,
    "knowledge_base_search",
    "Search the knowledge base for information...",
)

# Agent decides when to use the tool
agent = create_openai_tools_agent(llm, tools=[retriever_tool], prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)
```

---

## 5. Comparison Table

| Feature | Direct LLM | LangChain | LangGraph | Basic RAG | Agentic RAG | Self-RAG | Adaptive RAG | RAG-Fusion |
|---------|-----------|-----------|-----------|-----------|-------------|----------|---------------|------------|
| **Complexity** | Low | Medium | Medium-High | Medium | High | High | Medium-High | High |
| **Dependencies** | Minimal | LangChain | LangGraph | LangChain + FAISS | LangChain + FAISS | LangChain + FAISS | LangChain + FAISS | LangChain + FAISS |
| **Memory** | Manual | Automatic | Manual (state) | Automatic | Automatic | Automatic | Automatic | Automatic |
| **Control Flow** | Linear | Linear | Graph-based | Linear | Agent-based | Self-evaluating | Conditional | Multi-query |
| **Retrieval** | No | No | No | Always | Conditional | Adaptive | Conditional | Multi-query |
| **Query Analysis** | No | No | No | No | No | Yes | Yes | Yes |
| **Best For** | Simple chats | Composable apps | Complex workflows | Knowledge Q&A | Multi-step reasoning | High-quality answers | Mixed queries | Comprehensive retrieval |

📖 **For detailed comparison, see [COMPARISON.md](COMPARISON.md)**

---

## 6. When to Use Each Case

### Use **Direct LLM** when:
- Building a simple prototype
- You want minimal dependencies
- You need full control over prompts
- Learning how LLMs work

### Use **LangChain** when:
- You want a chatbot with memory and easy extensibility
- You plan to add tools or RAG later
- You prefer framework abstractions
- Building a production app that needs composability

### Use **LangGraph** when:
- You need explicit control flow (branching, loops)
- You have multi-step workflows
- You want to visualize execution flow
- You need error handling and retries

### Use **RAG** when:
- You have a knowledge base to query
- You want factual answers from your documents
- You always want retrieval (no decision-making needed)
- Building a document Q&A system

### Use **Agentic RAG** when:
- You want the LLM to decide when to retrieve
- You have multiple tools/data sources
- You need complex, multi-step reasoning
- Building an intelligent assistant that can plan

### Use **Self-RAG** when:
- You need high-quality, self-validated answers
- You want adaptive retrieval based on confidence
- Quality is more important than speed
- You want the model to self-correct

### Use **Adaptive RAG** when:
- You have a mix of simple and complex queries
- You want performance optimization
- You want to reduce unnecessary retrievals
- You need efficient query handling

### Use **RAG-Fusion** when:
- You have ambiguous or complex queries
- You need comprehensive document coverage
- You want multiple perspectives on queries
- Quality over speed

---

## 7. Example Queries to Try

### For RAG and Agentic RAG:
- "What is LangChain?"
- "Explain the difference between RAG and Agentic RAG"
- "How does LangGraph work?"

### For all chatbots:
- "What is the capital of France?"
- "Tell me a joke"
- "Explain quantum computing in simple terms"

Notice how RAG/Agentic RAG can answer questions about LangChain/RAG from the documents, while the others rely on general knowledge.

---

## 8. Troubleshooting

**Error: "OPENAI_API_KEY is not set"**
- Make sure you have a `.env` file with your API key.

**Error: "Module not found"**
- Make sure you've installed dependencies: `pip install -r requirements.txt`
- Make sure your virtual environment is activated.

**FAISS installation issues:**
- On Apple Silicon (M1/M2), you might need: `pip install faiss-cpu` (already in requirements.txt)
- For GPU support: `pip install faiss-gpu` (but requires CUDA)

**Agentic RAG shows verbose output:**
- Set `verbose=False` in `AgentExecutor` to hide agent reasoning steps.

---

## 9. Next Steps

- **Extend RAG**: Add more documents, use different vector stores (Chroma, Pinecone), or add metadata filtering.
- **Extend Agentic RAG**: Add more tools (web search, calculator, database queries).
- **Extend LangGraph**: Add conditional edges, loops, or multiple processing nodes.
- **Add streaming**: Stream responses token-by-token for better UX.
- **Add evaluation**: Test accuracy with a question-answer dataset.

---

Happy coding! 🚀

# Agentic-AI_Analyzers
