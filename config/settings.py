"""
config/settings.py — Central configuration for the chatbot
===========================================================
Having a single place for all magic strings / numbers means you only
need to change one file when, say, you add a new model or move Ollama
to a remote server.

Extend this later with:
  - pydantic-settings for .env file support
  - per-user config profiles
  - RAG / vector DB connection strings
"""


class AppSettings:
    # ── Ollama connection ─────────────────────────────────────────────────────
    # Ollama exposes a REST API on localhost:11434 by default.
    # Change this if you're running Ollama on another machine.
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # ── Models ────────────────────────────────────────────────────────────────
    # These must already be pulled locally: `ollama pull <model>`
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
    # These map 1-to-1 to Ollama's /api/chat options field.
    DEFAULT_TEMPERATURE: float = 0.7   # 0 = deterministic, 1+ = creative
    DEFAULT_TOP_P: float = 0.9         # nucleus sampling threshold
    DEFAULT_MAX_TOKENS: int = 2048     # max tokens in a single reply

    # ── UI ────────────────────────────────────────────────────────────────────
    APP_TITLE: str = "Local AI Chat"
    APP_ICON: str = "🤖"

    # ── System prompt ─────────────────────────────────────────────────────────
    # This is injected as the first "system" message in every conversation.
    # You can make this editable from the sidebar later.
    DEFAULT_SYSTEM_PROMPT: str = (
        "You are a helpful, harmless, and honest AI assistant running entirely "
        "on the user's local machine. Be concise, clear, and friendly."
    )

    # ── Timeouts ──────────────────────────────────────────────────────────────
    REQUEST_TIMEOUT: int = 120  # seconds; increase for slow hardware / big models
