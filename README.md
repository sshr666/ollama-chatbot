# Local RAG AI Chatbot

A fully local Retrieval-Augmented Generation (RAG) chatbot built using Streamlit, Ollama, ChromaDB, and open-source LLMs like Qwen and Llama.

## Features

- Fully local and offline AI chatbot
- PDF upload and document querying
- RAG pipeline with semantic search
- Real-time streaming responses
- Multiple local model support
- ChromaDB vector storage
- SentenceTransformer embeddings
- Streamlit-based interactive UI
- No OpenAI or cloud APIs

---

## Tech Stack

- Streamlit
- Ollama
- Qwen / Llama / Mistral
- ChromaDB
- SentenceTransformers
- LangChain
- PyPDF

---

## Project Structure

```bash
ollama-chatbot/
│
├── app.py
├── requirements.txt
│
├── config/
│   └── settings.py
│
├── components/
│   ├── chat_ui.py
│   └── sidebar.py
│
├── utils/
│   ├── ollama_client.py
│   └── rag.py
│
└── assets/
```

---

## Installation

### Clone the repository

```bash
git clone https://github.com/sshr666/ollama-chatbot.git
cd ollama-chatbot
```

### Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Install additional RAG dependencies

```bash
pip install chromadb sentence-transformers pypdf langchain langchain-community langchain-text-splitters torch
```

---

## Install Ollama

Download and install Ollama:

https://ollama.com

Pull models:

```bash
ollama pull qwen3:8b
ollama pull llama3
ollama pull mistral
```

Start Ollama:

```bash
ollama serve
```

---

## Run the chatbot

```bash
streamlit run app.py
```

---

## How It Works

1. Upload a PDF document
2. Text is extracted locally
3. Document is chunked into smaller sections
4. Chunks are converted into embeddings
5. Embeddings are stored in ChromaDB
6. User query retrieves relevant chunks semantically
7. Retrieved context is injected into the prompt
8. Ollama generates a grounded response

---

## Example Questions

```text
What is self-attention?
```

```text
What are the conclusions of the paper?
```

```text
Summarize the methodology section.
```

---

## Supported Models

- qwen3:8b
- llama3
- mistral
- phi3
- gemma2
- deepseek-r1

---

## Future Improvements

- Hybrid keyword + semantic retrieval
- Multi-document querying
- OCR for scanned PDFs
- Citation-aware responses
- Conversation memory
- Web search integration

---
