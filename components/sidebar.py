"""
components/sidebar.py — Left-hand settings panel
=================================================
Everything in the sidebar is rendered here.  It writes directly to
`st.session_state` so that app.py and the chat components can read
the user's chosen settings without needing to pass values around.

This is the natural Streamlit pattern: session_state is the shared store.

Extend this file to add:
  - RAG toggle + document uploader
  - Vector DB connection settings
  - Conversation export button
  - Per-session memory management
"""

import streamlit as st

from config.settings import AppSettings
from utils.ollama_client import OllamaClient, OllamaConnectionError


def _check_ollama_status(client: OllamaClient) -> tuple[bool, list[str]]:
    """
    Returns (is_running, list_of_installed_model_names).
    Cached for 10 s so we don't hammer Ollama on every rerun.
    """
    try:
        models = client.list_models()
        return True, models
    except OllamaConnectionError:
        return False, []


def render_sidebar() -> None:
    """Draw the entire sidebar and sync selections into session_state."""
    with st.sidebar:
        st.title("⚙️ Settings")
        st.divider()

        # ── Ollama status indicator ───────────────────────────────────────────
        client: OllamaClient = st.session_state.ollama_client
        is_running, installed_models = _check_ollama_status(client)

        if is_running:
            st.success("✅ Ollama is running", icon="🟢")
        else:
            st.error(
                "❌ Ollama not found\n\n"
                "Run in a terminal:\n```\nollama serve\n```",
                icon="🔴",
            )

        st.divider()

        # ── Model selector ────────────────────────────────────────────────────
        st.subheader("🧠 Model")

        # Merge the config list with whatever is actually installed so users
        # always see installed models even if they're not in settings.py.
        all_models = sorted(
            set(AppSettings.AVAILABLE_MODELS) | set(installed_models)
        )

        if not all_models:
            st.warning("No models found. Pull one:\n```\nollama pull llama3\n```")
            all_models = AppSettings.AVAILABLE_MODELS  # show defaults anyway

        current_model = st.session_state.selected_model
        # If stored model was removed, fall back gracefully.
        if current_model not in all_models:
            current_model = all_models[0]

        st.session_state.selected_model = st.selectbox(
            "Choose model",
            options=all_models,
            index=all_models.index(current_model),
            help="Only models pulled via `ollama pull <name>` will work.",
        )

        # Quick-pull helper text
        if installed_models:
            st.caption(f"**Installed:** {', '.join(installed_models)}")
        else:
            st.caption("No local models detected.")

        st.divider()

        # ── Generation parameters ─────────────────────────────────────────────
        st.subheader("🎛️ Generation")

        st.session_state.temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=AppSettings.DEFAULT_TEMPERATURE,
            step=0.05,
            help="Higher = more creative/random. Lower = more focused/deterministic.",
        )

        st.session_state.top_p = st.slider(
            "Top-P",
            min_value=0.1,
            max_value=1.0,
            value=AppSettings.DEFAULT_TOP_P,
            step=0.05,
            help="Nucleus sampling threshold. Usually leave at 0.9.",
        )

        st.session_state.max_tokens = st.slider(
            "Max tokens",
            min_value=128,
            max_value=8192,
            value=AppSettings.DEFAULT_MAX_TOKENS,
            step=128,
            help="Maximum length of a single reply.",
        )

        st.divider()

        # ── System prompt ─────────────────────────────────────────────────────
        st.subheader("📝 System Prompt")
        st.session_state.system_prompt = st.text_area(
            "System prompt",
            value=st.session_state.system_prompt,
            height=120,
            help=(
                "This message is sent before every conversation. "
                "Use it to give the model a persona or constraints."
            ),
        )

        st.divider()

        # ── Conversation controls ─────────────────────────────────────────────
        st.subheader("💬 Conversation")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

        with col2:
            msg_count = len(st.session_state.get("messages", []))
            st.metric("Messages", msg_count)

        st.divider()

        # ── Future extensions placeholder ─────────────────────────────────────
        with st.expander("🚀 Coming soon", expanded=False):
            st.markdown(
                """
                - 📄 **PDF upload + RAG**
                - 🗄️ **Vector DB (ChromaDB / FAISS)**
                - 🧰 **Tool calling**
                - 🧠 **Long-term memory**
                - 📊 **Multi-modal (images)**
                """
            )

        # ── Footer ────────────────────────────────────────────────────────────
        st.caption("Built with Streamlit + Ollama · 100% local")
