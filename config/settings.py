"""
config/settings.py — Central configuration (RAG edition)
=========================================================
Added RAG-specific settings below the existing ones.
Existing values are UNCHANGED so nothing breaks.
"""


class AppSettings:
    # ── Ollama connection ─────────────────────────────────────────────────────
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # ── Models ────────────────────────────────────────────────────────────────
    AVAILABLE_MODELS: list[str] = [
        "qwen3:8b",
        "llama3",
        "mistral",
        "phi3",
        "gemma2",
        "deepseek-r1:8b",
    ]
    DEFAULT_MODEL: str = "qwen3:8b"

    # ── Generation parameters ─────────────────────────────────────────────────
    DEFAULT_TEMPERATURE: float = 0.7
    DEFAULT_TOP_P: float = 0.9
    DEFAULT_MAX_TOKENS: int = 2048

    # ── UI ────────────────────────────────────────────────────────────────────
    APP_TITLE: str = "Local AI Chat"
    APP_ICON: str = "🤖"

    # ── System prompt ─────────────────────────────────────────────────────────
    DEFAULT_SYSTEM_PROMPT: str = (
        "You are a helpful, harmless, and honest AI assistant running entirely "
        "on the user's local machine. Be concise, clear, and friendly."
    )

    # ── RAG system prompt (used instead of DEFAULT when RAG is active) ────────
    # This is more directive — it tells the model to focus on the document.
    RAG_SYSTEM_PROMPT: str = (
        "You are a helpful document assistant. You answer questions based on "
        "the provided document context. Be precise, cite chunk numbers when "
        "relevant, and admit when information isn't in the document."
    )

    # ── Timeouts ──────────────────────────────────────────────────────────────
    REQUEST_TIMEOUT: int = 120

    # ── RAG configuration ─────────────────────────────────────────────────────

    # Where ChromaDB stores its vector database files on disk.
    # The directory is created automatically if it doesn't exist.
    CHROMA_PERSIST_DIR: str = "./chroma_db"

    # How many chunks to retrieve for each user query.
    # 4 is a good default: enough context, not too much prompt bloat.
    # The sidebar exposes this as a slider (1–8).
    RAG_TOP_K: int = 4

    # Maximum PDF file size accepted (in MB).
    # Larger PDFs take longer to embed on first upload.
    MAX_PDF_SIZE_MB: int = 50

    # Embedding model — runs locally via SentenceTransformers.
    # Downloaded once (~90 MB) to ~/.cache/huggingface/
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"