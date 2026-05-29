"""
components/sidebar.py — Settings panel (RAG edition)
=====================================================
New sections added (clearly marked with ── RAG ──):
  1. PDF uploader with progress feedback
  2. RAG enable/disable toggle
  3. Top-K retrieval slider
  4. Active document status display
  5. Clear document / re-upload controls

All existing sections (Ollama status, model picker, generation params,
system prompt, conversation controls) are PRESERVED exactly.
"""

import streamlit as st

from config.settings import AppSettings
from utils.ollama_client import OllamaClient, OllamaConnectionError


def _check_ollama_status(client: OllamaClient) -> tuple[bool, list[str]]:
    try:
        models = client.list_models()
        return True, models
    except OllamaConnectionError:
        return False, []


def _handle_pdf_upload(uploaded_file) -> None:
    """
    Process an uploaded PDF file through the full RAG ingestion pipeline.

    This runs synchronously in the sidebar.  For large PDFs (>20 pages)
    the embedding step can take 5–30 seconds on CPU — a spinner is shown.

    Steps:
      1. Read bytes from Streamlit's UploadedFile object
      2. Call rag.process_pdf_upload() → extract → chunk → embed → store
      3. Save collection_name and metadata to session_state
      4. Auto-enable RAG mode
    """
    from utils.rag import process_pdf_upload, collection_exists
    from config.settings import AppSettings

    filename = uploaded_file.name
    pdf_bytes = uploaded_file.read()

    # Check size limit
    size_mb = len(pdf_bytes) / (1024 * 1024)
    if size_mb > AppSettings.MAX_PDF_SIZE_MB:
        st.error(f"PDF is {size_mb:.1f} MB — maximum is {AppSettings.MAX_PDF_SIZE_MB} MB.")
        return

    # If this exact filename was already embedded, ask before re-embedding.
    # (collection_exists checks the ChromaDB directory for the collection name)
    already_embedded = collection_exists(filename, AppSettings.CHROMA_PERSIST_DIR)

    if already_embedded and st.session_state.get("rag_filename") == filename:
        st.info(f"**{filename}** is already indexed. Using existing embeddings.")
        st.session_state.rag_enabled = True
        return

    # Run the full pipeline with a progress indicator
    with st.spinner(f"Processing **{filename}**… (extract → chunk → embed → store)"):
        try:
            collection_name, chunk_count = process_pdf_upload(
                pdf_bytes=pdf_bytes,
                filename=filename,
                persist_dir=AppSettings.CHROMA_PERSIST_DIR,
            )

            # Store results in session_state so chat_ui.py can access them
            st.session_state.rag_collection_name = collection_name
            st.session_state.rag_filename = filename
            st.session_state.rag_chunk_count = chunk_count
            st.session_state.rag_enabled = True

            # Clear chat history so the new conversation context is clean
            st.session_state.messages = []

            st.success(
                f"✅ **{filename}** indexed successfully!\n\n"
                f"{chunk_count} chunks stored in ChromaDB."
            )

        except ValueError as e:
            # e.g. scanned PDF with no extractable text
            st.error(f"⚠️ Could not process PDF: {e}")
        except Exception as e:
            st.error(f"⚠️ Unexpected error during PDF processing: {e}")


def render_sidebar() -> None:
    """Draw the full sidebar including the new RAG section."""
    with st.sidebar:
        st.title("⚙️ Settings")
        st.divider()

        # ── Ollama status (unchanged) ─────────────────────────────────────────
        client: OllamaClient = st.session_state.ollama_client
        is_running, installed_models = _check_ollama_status(client)

        if is_running:
            st.success("✅ Ollama is running", icon="🟢")
        else:
            st.error(
                "❌ Ollama not found\n\nRun:\n```\nollama serve\n```",
                icon="🔴",
            )

        st.divider()

        # ── Model selector (unchanged) ────────────────────────────────────────
        st.subheader("🧠 Model")

        all_models = sorted(
            set(AppSettings.AVAILABLE_MODELS) | set(installed_models)
        )
        if not all_models:
            st.warning("No models found. Pull one:\n```\nollama pull llama3\n```")
            all_models = AppSettings.AVAILABLE_MODELS

        current_model = st.session_state.selected_model
        if current_model not in all_models:
            current_model = all_models[0]

        st.session_state.selected_model = st.selectbox(
            "Choose model",
            options=all_models,
            index=all_models.index(current_model),
            help="Only models pulled via `ollama pull <name>` will work.",
        )

        if installed_models:
            st.caption(f"**Installed:** {', '.join(installed_models)}")
        else:
            st.caption("No local models detected.")

        st.divider()

        # ══════════════════════════════════════════════════════════════════════
        # ── RAG: Document Upload ──────────────────────────────────────────────
        # ══════════════════════════════════════════════════════════════════════
        st.subheader("📄 Document (RAG)")

        # File uploader — only PDFs accepted
        # Streamlit re-runs when the user uploads, so _handle_pdf_upload runs once.
        uploaded_file = st.file_uploader(
            "Upload a PDF",
            type=["pdf"],
            help=(
                "PDF will be parsed, chunked, embedded locally with "
                "all-MiniLM-L6-v2, and stored in ChromaDB. "
                f"Max size: {AppSettings.MAX_PDF_SIZE_MB} MB."
            ),
            key="pdf_uploader",
        )

        # Only process when a NEW file has been uploaded
        if uploaded_file is not None:
            # Use the file size as a proxy for "is this a different upload"
            upload_key = f"{uploaded_file.name}_{uploaded_file.size}"
            if st.session_state.get("_last_upload_key") != upload_key:
                st.session_state["_last_upload_key"] = upload_key
                _handle_pdf_upload(uploaded_file)

        # ── Active document status ────────────────────────────────────────────
        if st.session_state.rag_filename:
            st.markdown(
                f"**Active document:**  \n"
                f"📎 `{st.session_state.rag_filename}`  \n"
                f"🗂️ {st.session_state.rag_chunk_count} chunks indexed"
            )

            # RAG enable/disable toggle
            # We use a dedicated key so Streamlit's widget state doesn't
            # overwrite rag_enabled on the same run that _handle_pdf_upload()
            # sets it to True (the toggle would silently reset it to False).
            toggled = st.toggle(
                "Enable RAG mode",
                value=st.session_state.rag_enabled,
                key="rag_toggle",
                help=(
                    "When ON: your question is enriched with relevant chunks "
                    "from the document before being sent to the model.  "
                    "When OFF: the model answers from its own knowledge only."
                ),
            )
            if toggled != st.session_state.rag_enabled:
                st.session_state.rag_enabled = toggled

            if st.session_state.rag_enabled:
                st.info("🔍 RAG active — responses grounded in your document.")
            else:
                st.warning("⚠️ RAG off — model using its own knowledge only.")

            # Top-K slider — how many chunks to retrieve per query
            st.session_state.rag_top_k = st.slider(
                "Chunks to retrieve (Top-K)",
                min_value=1,
                max_value=8,
                value=st.session_state.rag_top_k,
                step=1,
                help=(
                    "How many document chunks are injected into each prompt. "
                    "More = more context but slower and larger prompts."
                ),
            )

            # Clear document button
            if st.button("🗑️ Remove document", use_container_width=True):
                st.session_state.rag_collection_name = None
                st.session_state.rag_filename = None
                st.session_state.rag_chunk_count = 0
                st.session_state.rag_enabled = False
                st.session_state.messages = []
                st.session_state.pop("_last_upload_key", None)
                st.rerun()
        else:
            st.caption("Upload a PDF above to enable RAG mode.")

        st.divider()

        # ── Generation parameters (unchanged) ─────────────────────────────────
        st.subheader("🎛️ Generation")

        st.session_state.temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=AppSettings.DEFAULT_TEMPERATURE,
            step=0.05,
            help="Higher = more creative. Lower = more focused.",
        )

        st.session_state.top_p = st.slider(
            "Top-P",
            min_value=0.1,
            max_value=1.0,
            value=AppSettings.DEFAULT_TOP_P,
            step=0.05,
        )

        st.session_state.max_tokens = st.slider(
            "Max tokens",
            min_value=128,
            max_value=8192,
            value=AppSettings.DEFAULT_MAX_TOKENS,
            step=128,
        )

        st.divider()

        # ── System prompt (unchanged) ─────────────────────────────────────────
        st.subheader("📝 System Prompt")

        # Auto-swap system prompt based on RAG mode
        default_prompt = (
            AppSettings.RAG_SYSTEM_PROMPT
            if st.session_state.rag_enabled
            else AppSettings.DEFAULT_SYSTEM_PROMPT
        )

        st.session_state.system_prompt = st.text_area(
            "System prompt",
            value=default_prompt,
            height=120,
            help="Sent before every conversation. Changes when RAG mode toggles.",
        )

        st.divider()

        # ── Conversation controls (unchanged) ─────────────────────────────────
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
        st.caption("Built with Streamlit + Ollama + ChromaDB · 100% local")