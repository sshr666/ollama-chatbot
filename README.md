# 🤖 Local AI Chatbot — Streamlit + Ollama

A fully local AI chatbot. No cloud APIs. No data leaves your machine.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Your Browser                       │
│         http://localhost:8501                        │
└─────────────────────┬───────────────────────────────┘
                      │  HTTP (Streamlit WebSocket)
┌─────────────────────▼───────────────────────────────┐
│              Streamlit App (app.py)                  │
│  ┌─────────────┐   ┌───────────────────────────┐    │
│  │  sidebar.py │   │       chat_ui.py           │    │
│  │ Model picker│   │  render_chat_history()     │    │
│  │ Temp slider │   │  render_chat_input()       │    │
│  │ Sys prompt  │   │  st.write_stream() ──────► │    │
│  └─────────────┘   └───────────┬───────────────┘    │
│                                │                     │
│  ┌─────────────────────────────▼─────────────────┐  │
│  │           utils/ollama_client.py               │  │
│  │    OllamaClient.stream_response() generator    │  │
│  └─────────────────────────────┬─────────────────┘  │
└────────────────────────────────┼────────────────────┘
                                 │  HTTP POST /api/chat
                                 │  (NDJSON streaming)
┌────────────────────────────────▼────────────────────┐
│            Ollama Server (localhost:11434)            │
│  ┌───────────────────────────────────────────────┐  │
│  │         Local LLM Runtime                     │  │
│  │   qwen3:8b  │  llama3  │  mistral  │  phi3    │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

### Session State Flow

```
User types message
      │
      ▼
st.chat_input() fires
      │
      ▼
session_state["messages"].append({"role":"user","content":"..."})
      │
      ▼
OllamaClient.stream_response()  ← generator, yields tokens one by one
      │
      ▼
st.write_stream(generator)  ← streams tokens into chat bubble live
      │
      ▼
full_reply returned when generator exhausted
      │
      ▼
session_state["messages"].append({"role":"assistant","content":full_reply})
      │
      ▼
Streamlit reruns → render_chat_history() redraws all messages
```

---

## Folder Structure

```
ollama-chatbot/
│
├── app.py                    ← Entry point. Run: streamlit run app.py
│
├── requirements.txt          ← pip dependencies
│
├── config/
│   ├── __init__.py
│   └── settings.py           ← All magic strings / defaults in one place
│
├── utils/
│   ├── __init__.py
│   └── ollama_client.py      ← Ollama REST API wrapper + streaming logic
│
├── components/
│   ├── __init__.py
│   ├── sidebar.py            ← Left panel: model picker, sliders, controls
│   └── chat_ui.py            ← Chat history + input + streaming loop
│
└── assets/                   ← Static files (CSS, images) — empty for now
```

**Why this structure?**
- `config/` — change settings without touching logic
- `utils/` — reusable helpers, easy to unit-test in isolation
- `components/` — each UI section is one file; easy to find and edit
- `assets/` — placeholder for CSS overrides, logos, etc.

---

## Setup & Installation

### Step 1 — Install Ollama

```bash
# Linux / macOS
curl -fsSL https://ollama.com/install.sh | sh

# Verify
ollama --version
```

### Step 2 — Pull models

```bash
# Pull at least one model before starting the app
ollama pull qwen3:8b      # ~5 GB, fast, great for chat
ollama pull llama3        # ~4.7 GB, Meta's flagship open model

# Optional extras
ollama pull mistral       # ~4.1 GB, great at instruction-following
ollama pull phi3          # ~2.3 GB, very fast on low-end hardware
```

### Step 3 — Start Ollama server

```bash
# Keep this terminal open while using the chatbot
ollama serve

# You should see:
# Ollama is running on http://localhost:11434
```

### Step 4 — Clone / create the project

```bash
# If starting fresh
mkdir ollama-chatbot && cd ollama-chatbot

# Create a virtual environment (recommended — keeps deps isolated)
python3 -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# Install Python dependencies
pip install -r requirements.txt
```

### Step 5 — Run the app

```bash
# Make sure you're in the project root (where app.py lives)
streamlit run app.py

# Streamlit will print:
#   Local URL:  http://localhost:8501
#   Network URL: http://192.168.x.x:8501
```

Open http://localhost:8501 in your browser. That's it!

---

## Streamlit Chat Components — Deep Dive

### `st.chat_message(role, avatar=...)`

```python
with st.chat_message("assistant", avatar="🤖"):
    st.markdown("Hello! How can I help?")
```

- Creates a styled message bubble
- `role` controls alignment + default avatar
- Everything inside the `with` block goes inside the bubble
- Accepts `"user"`, `"assistant"`, or any custom string

### `st.chat_input(placeholder)`

```python
if prompt := st.chat_input("Ask anything…"):
    # only runs when user hits Enter
    handle(prompt)
```

- Renders a sticky text box pinned to the bottom of the page
- Returns `None` when idle, the string when submitted
- The walrus operator `:=` assigns and tests in one line

### `st.write_stream(generator)`

```python
full_text = st.write_stream(token_generator)
```

- Accepts any Python generator that yields strings
- Appends each yielded chunk to the UI in real time
- Returns the complete concatenated string when done
- Must be called inside a `st.chat_message` context

---

## How Ollama Communication Works

Ollama exposes a REST API on `http://localhost:11434`.

### List models
```
GET /api/tags
→ {"models": [{"name": "llama3:latest", ...}]}
```

### Streaming chat
```
POST /api/chat
Body: {
  "model": "llama3",
  "messages": [{"role": "user", "content": "Hi!"}],
  "stream": true
}

Response (NDJSON — one JSON object per line):
{"message":{"role":"assistant","content":"Hello"},"done":false}
{"message":{"role":"assistant","content":"!"},"done":false}
{"message":{"role":"assistant","content":""},"done":true}
```

The Python `requests` library reads this line-by-line with `iter_lines()`,
and we `yield` each token to Streamlit's `write_stream()`.

---

## Common Debugging Issues

### "Cannot reach Ollama"
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not, start it:
ollama serve
```

### "Model not found"
```bash
# List installed models
ollama list

# Pull the model you need
ollama pull qwen3:8b
```

### "Streamlit not found"
```bash
# Make sure your venv is active
source .venv/bin/activate
pip install streamlit requests
```

### Slow responses
- Use a smaller model: `phi3` (~2.3 GB) is much faster than `qwen3:8b`
- Reduce `max_tokens` in the sidebar
- Make sure you have a GPU (Ollama auto-detects NVIDIA/AMD via `nvidia-smi`)

### Port already in use
```bash
# Run on a different port
streamlit run app.py --server.port 8502
```

### Hot-reload not working
```bash
# Force reload
streamlit run app.py --server.runOnSave true
```

---

## How to Extend This Project

### Add RAG (PDF chat)
```
pip install pypdf chromadb sentence-transformers langchain
```
1. Add a file uploader in `sidebar.py`
2. Create `utils/rag.py` with chunking + embedding logic
3. Retrieve relevant chunks before calling Ollama
4. Inject them into the system prompt

### Add tool calling
1. Define tools as JSON schemas
2. Pass `tools` to Ollama's `/api/chat` endpoint
3. Parse `tool_use` blocks in the response
4. Execute tools locally and return results

### Add vector memory
```
pip install chromadb
```
1. Embed each conversation turn
2. Store in ChromaDB
3. Retrieve semantically similar past turns before each reply

### Add a REST API layer
```
pip install fastapi uvicorn
```
1. Wrap `OllamaClient` in FastAPI endpoints
2. Point Streamlit at the FastAPI server instead of Ollama directly
3. Now you can have multiple frontends (mobile, CLI, web)

---

## VS Code Tips

Install the recommended extensions:
- **Python** (ms-python.python)
- **Pylance** (ms-python.vscode-pylance)
- **Ruff** (charliermarsh.ruff) — fast linter

Add to `.vscode/settings.json`:
```json
{
  "python.defaultInterpreterPath": ".venv/bin/python",
  "editor.formatOnSave": true,
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff"
  }
}
```

Run the app from VS Code terminal:
```bash
streamlit run app.py
```
