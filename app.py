"""
app.py — Main entry point for the Ollama Local AI Chatbot (RAG edition)
=======================================================================
Changes from v1:
  - init_session_state() now bootstraps RAG-related state keys:
      rag_enabled          : bool  — master toggle for RAG mode
      rag_collection_name  : str   — which ChromaDB collection to query
      rag_filename         : str   — display name of the active PDF
      rag_chunk_count      : int   — how many chunks were indexed

Everything else (Streamlit config, sidebar, chat UI) is unchanged.
The RAG logic itself lives entirely in utils/rag.py and is called
from components/chat_ui.py — app.py stays thin.

Architecture (RAG mode ON):
  User types message
       ↓
  Streamlit captures input (st.chat_input)
       ↓
  chat_ui._handle_user_message()
       ↓
  rag.retrieve_context(query, collection_name)   ← NEW
       ↓
  rag.build_rag_prompt(query, chunks)            ← NEW
       ↓
  ollama_client.stream_response(augmented_prompt)
       ↓
  Tokens streamed into st.chat_message
       ↓
  Full reply saved to session_state["messages"]
"""

import streamlit as st

from config.settings import AppSettings
from utils.ollama_client import OllamaClient
from components.sidebar import render_sidebar
from components.chat_ui import render_chat_history, render_chat_input


st.set_page_config(
    page_title="Local AI Chat + RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state() -> None:
    """Bootstrap all session state keys on first load."""

    # ── Core chat state (unchanged from v1) ───────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "selected_model" not in st.session_state:
        st.session_state.selected_model = AppSettings.DEFAULT_MODEL

    if "ollama_client" not in st.session_state:
        st.session_state.ollama_client = OllamaClient(
            base_url=AppSettings.OLLAMA_BASE_URL
        )

    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = AppSettings.DEFAULT_SYSTEM_PROMPT

    # ── RAG state (new in v2) ─────────────────────────────────────────────────
    # rag_enabled: when True, every user message goes through the RAG pipeline
    # before being sent to Ollama.  The sidebar toggle controls this.
    if "rag_enabled" not in st.session_state:
        st.session_state.rag_enabled = False

    # rag_collection_name: the ChromaDB collection for the active PDF.
    # Set by sidebar.py after a PDF is uploaded and embedded.
    if "rag_collection_name" not in st.session_state:
        st.session_state.rag_collection_name = None

    # rag_filename: shown in the UI so the user knows which PDF is active.
    if "rag_filename" not in st.session_state:
        st.session_state.rag_filename = None

    # rag_chunk_count: shown in sidebar for transparency ("42 chunks indexed").
    if "rag_chunk_count" not in st.session_state:
        st.session_state.rag_chunk_count = 0

    # rag_top_k: how many chunks to retrieve per query (sidebar slider controls this)
    if "rag_top_k" not in st.session_state:
        st.session_state.rag_top_k = AppSettings.RAG_TOP_K


def main() -> None:
    """Application shell — thin orchestrator, no logic here."""
    init_session_state()

    render_sidebar()

    # ── Dynamic header ────────────────────────────────────────────────────────
    if st.session_state.rag_enabled and st.session_state.rag_filename:
        st.title("🤖 Local AI Chat  ·  📄 RAG Mode")
        st.caption(
            f"Model: **{st.session_state.selected_model}** via Ollama  ·  "
            f"Document: **{st.session_state.rag_filename}**  ·  "
            f"{st.session_state.rag_chunk_count} chunks indexed  ·  "
            "100% local — no data leaves your machine."
        )
    else:
        st.title("🤖 Local AI Chat")
        st.caption(
            f"Running **{st.session_state.selected_model}** locally via Ollama — "
            "no data leaves your machine."
        )

    st.divider()

    render_chat_history()
    render_chat_input()


if __name__ == "__main__":
    main()