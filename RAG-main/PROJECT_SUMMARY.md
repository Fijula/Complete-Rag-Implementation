# Project Summary: Multi-Style Chatbot Examples

## What's Included

This repository contains **8 different chatbot implementations** demonstrating various approaches to building LLM-powered chatbots:

### Core Chatbot Types (3):
1. **Direct LLM** (`llm_chat.py`) - Minimal, no frameworks
2. **LangChain** (`langchain_chat.py`) - Composable chains with memory
3. **LangGraph** (`langgraph_chat.py`) - Graph-based state machine

### RAG Variants (5):
4. **Basic RAG** (`rag_chat.py`) - Always retrieves documents
5. **Agentic RAG** (`agentic_rag_chat.py`) - LLM decides when to retrieve
6. **Self-RAG** (`self_rag_chat.py`) - Self-evaluates and adaptively retrieves
7. **Adaptive RAG** (`adaptive_rag_chat.py`) - Analyzes query complexity
8. **RAG-Fusion** (`rag_fusion_chat.py`) - Multiple query variations

## Key Differences

### Core Chatbots:
- **Direct LLM**: Simplest, manual prompt management
- **LangChain**: Automatic memory, composable components
- **LangGraph**: Explicit control flow, state management

### RAG Types:
- **Basic RAG**: Always retrieves (simplest RAG)
- **Agentic RAG**: LLM agent decides when to retrieve
- **Self-RAG**: Self-evaluates responses, adaptive retrieval
- **Adaptive RAG**: Query complexity-based retrieval
- **RAG-Fusion**: Multiple query perspectives

## Total RAG Types Covered: 5

We've implemented 5 different RAG variants:
1. Basic RAG (Naive RAG)
2. Agentic RAG
3. Self-RAG
4. Adaptive RAG
5. RAG-Fusion

## Files Structure

```
.
├── README.md                 # Main documentation
├── COMPARISON.md             # Detailed comparison of all types
├── PROJECT_SUMMARY.md        # This file
├── config.py                 # Shared configuration
├── requirements.txt          # Dependencies
├── .gitignore               # Git ignore rules
│
├── Core Chatbots:
│   ├── llm_chat.py          # Case 3: Direct LLM
│   ├── langchain_chat.py     # Case 1: LangChain
│   └── langgraph_chat.py     # Case 2: LangGraph
│
└── RAG Variants:
    ├── rag_chat.py           # Case 4: Basic RAG
    ├── agentic_rag_chat.py   # Case 5: Agentic RAG
    ├── self_rag_chat.py      # Case 6: Self-RAG
    ├── adaptive_rag_chat.py  # Case 7: Adaptive RAG
    └── rag_fusion_chat.py    # Case 8: RAG-Fusion
```

## Quick Start

1. **Setup environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure API key:**
   ```bash
   echo "OPENAI_API_KEY=your_key_here" > .env
   echo "OPENAI_MODEL=gpt-4o-mini" >> .env
   ```

3. **Run any chatbot:**
   ```bash
   python llm_chat.py           # Direct LLM
   python langchain_chat.py     # LangChain
   python langgraph_chat.py     # LangGraph
   python rag_chat.py           # Basic RAG
   python agentic_rag_chat.py   # Agentic RAG
   python self_rag_chat.py      # Self-RAG
   python adaptive_rag_chat.py  # Adaptive RAG
   python rag_fusion_chat.py    # RAG-Fusion
   ```

## Documentation

- **README.md**: Main documentation with setup and usage
- **COMPARISON.md**: Detailed comparison table and explanations
- **PROJECT_SUMMARY.md**: This summary file

## Next Steps for Git

To push to a remote repository:

```bash
# Create initial commit
git commit -m "Initial commit: 8 chatbot implementations with 5 RAG variants"

# Add remote (replace with your repo URL)
git remote add origin https://github.com/yourusername/your-repo.git

# Push to remote
git push -u origin main
```

## Testing All Implementations

To test all chatbots, run each script and try these queries:

**For RAG variants:**
- "What is LangChain?"
- "Explain the difference between RAG and Agentic RAG"
- "How does LangGraph work?"

**For all chatbots:**
- "What is the capital of France?"
- "Tell me a joke"
- "Explain quantum computing in simple terms"

Observe how different RAG types handle the same queries differently!

