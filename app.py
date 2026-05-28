"""
app.py — Main entry point for the Ollama Local AI Chatbot
=========================================================
This is the root file Streamlit runs. It wires together:
  - The sidebar (model selector, settings)
  - The chat UI (message history + input box)
  - The Ollama streaming client
  - Session state management (so history survives reruns)

Architecture:
  User types message
       ↓
  Streamlit captures input (st.chat_input)
       ↓
  app.py appends to session_state["messages"]
       ↓
  ollama_client.stream_response() hits Ollama REST API
       ↓
  Tokens arrive one-by-one → streamed into st.chat_message
       ↓
  Full reply saved back to session_state["messages"]
"""

import streamlit as st

# ── Local module imports ──────────────────────────────────────────────────────
# Each of these lives in a sub-folder so the project stays organised as it grows.
from config.settings import AppSettings
from utils.ollama_client import OllamaClient
from components.sidebar import render_sidebar
from components.chat_ui import render_chat_history, render_chat_input


# ── Page config (must be the FIRST Streamlit call) ───────────────────────────
st.set_page_config(
    page_title="Local AI Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state() -> None:
    """
    Bootstrap Streamlit session state on the very first run.

    session_state persists across reruns (e.g. each message send) within a
    single browser tab, but resets if the page is hard-refreshed.  This is
    the lightweight "memory" that keeps the conversation alive.
    """
    if "messages" not in st.session_state:
        # Each message is a dict: {"role": "user"|"assistant", "content": "..."}
        st.session_state.messages = []

    if "selected_model" not in st.session_state:
        st.session_state.selected_model = AppSettings.DEFAULT_MODEL

    if "ollama_client" not in st.session_state:
        # Instantiate once; reused for every message in the session.
        st.session_state.ollama_client = OllamaClient(
            base_url=AppSettings.OLLAMA_BASE_URL
        )

    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = AppSettings.DEFAULT_SYSTEM_PROMPT


def main() -> None:
    """
    Application shell.

    Streamlit re-runs this entire function from top to bottom every time:
      • the user sends a message
      • any widget changes value
      • the page is first loaded
    Session state is the glue that carries data between reruns.
    """
    init_session_state()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    # render_sidebar() draws the left panel and returns whatever settings the
    # user has picked.  We deliberately keep UI rendering separate from logic.
    render_sidebar()

    # ── Page header ───────────────────────────────────────────────────────────
    st.title("🤖 Local AI Chat")
    st.caption(
        f"Running **{st.session_state.selected_model}** locally via Ollama — "
        "no data leaves your machine."
    )

    st.divider()

    # ── Chat history ─────────────────────────────────────────────────────────
    # Renders all previous turns so the user sees the full conversation.
    render_chat_history()

    # ── Chat input + response ─────────────────────────────────────────────────
    # render_chat_input() blocks until the user submits a message, then drives
    # the streaming response cycle.
    render_chat_input()


if __name__ == "__main__":
    main()
