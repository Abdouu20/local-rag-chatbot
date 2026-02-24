# 🤖 Local RAG Chatbot

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.41.1-FF4B4B)](https://streamlit.io/)

> **A production-ready, 100% local Retrieval-Augmented Generation (RAG) chatbot** that answers questions based on your documents. No cloud APIs, no data leaks, complete privacy.

![Demo](assets/demo-screenshot.png)

---

## 🚀 Features

| Feature | Description |
|---------|-------------|
| 🔒 **100% Local** | Runs entirely on your machine - no data leaves your computer |
| 📄 **Multi-Format Support** | PDF, TXT, DOCX document ingestion |
| 🎯 **Accurate Answers** | RAG-powered responses with source citations |
| 🛡️ **Security-First** | Input validation, prompt injection prevention |
| ⚡ **Fast Response** | Optimized for consumer GPUs (RTX 3060 Ti tested) |
| 📊 **Source Tracking** | Every answer includes document references |

---

## 🏗️ QUERY RESPONSE FLOW  
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    📤 QUERY RESPONSE FLOW                                              │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐
    │   💬 User    │
    │   Asks       │
    │   Question   │
    └──────┬───────┘
           │
           │ Natural Language Query
           ▼
    ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
    │  STEP 1: SECURITY VALIDATION (src/security/validator.py)                                        │
    │  ┌────────────────────────────────────────────────────────────────────────────────────────────┐  │
    │  │  • Max Length Check (2000 chars)                                                           │  │
    │  │  • Empty Input Check                                                                         │  │
    │  │  • Prompt Injection Detection (patterns)                                                     │  │
    │  │  • Input Sanitization                                                                        │  │
    │  └────────────────────────────────────────────────────────────────────────────────────────────┘  │
    │                                   │                                                              │
    │                                   ▼                                                              │
    │                  ┌────────────────────────────────┐                                              │
    │                  │  ✅ Validated Query            │                                              │
    │                  └────────────────┬───────────────┘                                              │
    └───────────────────────────────────┼────────────────────────────────────────────────────────────────┘
                                        │
                                        │ Validated Query String
                                        ▼
    ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
    │  STEP 2: VECTOR SEARCH (src/rag/vector_store.py)                                                │
    │  ┌────────────────────────────────────────────────────────────────────────────────────────────┐  │
    │  │  ChromaDB Similarity Search                                                                │  │
    │  │  • Query Embedding: Generate on-the-fly                                                    │  │
    │  │  • Similarity: Cosine Distance                                                             │  │
    │  │  • Top-K: 5 results                                                                        │  │
    │  │  • Filter: Collection = rag_documents                                                      │  │
    │  └────────────────────────────────────────────────────────────────────────────────────────────┘  │
    │                                   │                                                              │
    │                                   ▼                                                              │
    │                  ┌────────────────────────────────┐                                              │
    │                  │  Retrieved Documents           │                                              │
    │                  │  [Doc1, Doc2, Doc3...]         │                                              │
    │                  │  + Similarity Scores           │                                              │
    │                  └────────────────┬───────────────┘                                              │
    └───────────────────────────────────┼────────────────────────────────────────────────────────────────┘
                                        │
                                        │ Retrieved Context
                                        ▼
    ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
    │  STEP 3: PROMPT CONSTRUCTION (src/llm/prompts.py)                                               │
    │  ┌────────────────────────────────────────────────────────────────────────────────────────────┐  │
    │  │  RAG Prompt Template                                                                       │  │
    │  │  ┌──────────────────────────────────────────────────────────────────────────────────────┐  │  │
    │  │  │  You are a helpful AI assistant that answers questions based on context.             │  │  │
    │  │  │                                                                                      │  │  │
    │  │  │  IMPORTANT RULES:                                                                    │  │  │
    │  │  │  1. Only answer based on provided context                                            │  │  │
    │  │  │  2. Cite sources using [Source X] format                                             │  │  │
    │  │  │  3. Do not hallucinate                                                               │  │  │
    │  │  │                                                                                      │  │  │
    │  │  │  Context: {retrieved_documents}                                                      │  │  │
    │  │  │  Question: {user_query}                                                              │  │  │
    │  │  │  Answer:                                                                             │  │  │
    │  │  └──────────────────────────────────────────────────────────────────────────────────────┘  │  │
    │  └────────────────────────────────────────────────────────────────────────────────────────────┘  │
    │                                   │                                                              │
    │                                   ▼                                                              │
    │                  ┌────────────────────────────────┐                                              │
    │                  │  Final Prompt (~1500 tokens)   │                                              │
    │                  └────────────────┬───────────────┘                                              │
    └───────────────────────────────────┼────────────────────────────────────────────────────────────────┘
                                        │
                                        │ Prompt String
                                        ▼
    ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
    │  STEP 4: LLM GENERATION (src/llm/model.py)                                                      │
    │  ┌────────────────────────────────────────────────────────────────────────────────────────────┐  │
    │  │  Ollama + Llama 3.2:latest                                                                 │  │
    │  │  • Base URL: http://localhost:11434                                                        │  │
    │  │  • Temperature: 0.7                                                                        │  │
    │  │  • Context Window: 4096 tokens                                                             │  │
    │  │  • Inference: GPU Accelerated (RTX 3060 Ti)                                                │  │
    │  └────────────────────────────────────────────────────────────────────────────────────────────┘  │
    │                                   │                                                              │
    │                                   ▼                                                              │
    │                  ┌────────────────────────────────┐                                              │
    │                  │  Generated Response            │                                              │
    │                  │  (~200-500 tokens)             │                                              │
    │                  └────────────────┬───────────────┘                                              │
    └───────────────────────────────────┼────────────────────────────────────────────────────────────────┘
                                        │
                                        │ Response + Sources
                                        ▼
    ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
    │  STEP 5: RESPONSE DELIVERY (src/ui/app.py)                                                      │
    │  ┌────────────────────────────────────────────────────────────────────────────────────────────┐  │
    │  │  Streamlit UI                                                                              │  │
    │  │  • Display Answer                                                                          │  │
    │  │  • Show Source Citations                                                                   │  │
    │  │  • Expandable Source Details                                                               │  │
    │  │  • Chat History Saved                                                                      │  │
    │  └────────────────────────────────────────────────────────────────────────────────────────────┘  │
    │                                   │                                                              │
    │                                   ▼                                                              │
    │                  ┌────────────────────────────────┐                                              │
    │                  │  ✅ Answer Displayed           │                                              │
    │                  │  To User                       │                                              │
    │                  └────────────────────────────────┘                                              │
    └──────────────────────────────────────────────────────────────────────────────────────────────────┘
---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **LLM Runtime** | Ollama |
| **LLM Model** | Llama 3.2 3B (upgradeable to 8B) |
| **Embeddings** | BGE-M3 (Sentence Transformers) |
| **Vector DB** | ChromaDB |
| **Orchestration** | LangChain |
| **UI Framework** | Streamlit |
| **Language** | Python 3.10+ |

---

## 📋 Prerequisites

- **OS:** Windows 10/11, Linux, or macOS
- **GPU:** NVIDIA GPU with 8GB+ VRAM recommended (RTX 3060 Ti tested)
- **RAM:** 16GB minimum, 32GB recommended
- **Storage:** 10GB free space
- **Software:** 
  - [Ollama](https://ollama.ai/) (v0.16+)
  - [Python](https://www.python.org/) (3.10+)
  - [Git](https://git-scm.com/)

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/local-rag-chatbot.git
cd local-rag-chatbot


2. Install Ollama & Download Model
# Download Ollama from https://ollama.ai/

# Pull the LLM model
ollama pull llama3.2:latest

3. Setup Python Environment

# Create conda environment
conda create -n rag-chatbot python=3.10 -y
conda activate rag-chatbot

# Install dependencies
pip install -r requirements.txt


4. Configure Environment

# Copy environment template
copy .env.example .env

# Edit .env with your settings (optional - defaults work)

5. Launch the Application
# Ensure Ollama is running
ollama serve

# Start the Streamlit app
streamlit run src/ui/app.py

📖 Usage Guide
Upload Documents
Click the 📁 Upload Documents section in the sidebar
Select PDF, TXT, or DOCX files
Wait for ingestion confirmation ("X chunks added")
Ask Questions
Type your question in the chat input
Press Enter or click Send
View the answer with source citations
Click 📚 View Sources to see referenced documents
Example Questions
"What are the main products mentioned?"
"When was the company founded?"
"Summarize the key points"
"What contact information is provided?"

🧪 Testing
# Run verification tests
python tests/verification_test.py

# Test the pipeline directly
python src/main.py

📁 Project Structure

local-rag-chatbot/
├── src/
│   ├── config.py           # Configuration management
│   ├── main.py             # RAG pipeline orchestration
│   ├── rag/                # RAG components
│   │   ├── document_loader.py
│   │   ├── chunking.py
│   │   ├── embeddings.py
│   │   ├── vector_store.py
│   │   └── retriever.py
│   ├── llm/                # LLM components
│   │   ├── model.py
│   │   └── prompts.py
│   ├── security/           # Security measures
│   │   └── validator.py
│   └── ui/                 # User interface
│       └── app.py
├── data/
│   ├── sample_documents/   # Uploaded documents
│   └── vector_store/       # ChromaDB persistence
├── tests/                  # Test suite
├── docs/                   # Documentation
├── .env                    # Configuration (gitignored)
├── .env.example            # Configuration template
├── requirements.txt        # Dependencies
└── README.md               # This file
