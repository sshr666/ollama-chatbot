"""
components/chat_ui.py — Chat UI with RAG integration
======================================================
The ONLY change from v1 is inside _handle_user_message():

  BEFORE (v1):
    prompt → ollama_client.stream_response(prompt)

  AFTER (v2, RAG mode ON):
    prompt
      → rag.retrieve_context(prompt, collection_name)   # semantic search
      → rag.build_rag_prompt(prompt, chunks)            # inject context
      → ollama_client.stream_response(augmented_prompt) # stream as before

When RAG is OFF (rag_enabled = False), the code path is IDENTICAL to v1.
Streaming works exactly the same in both modes.

The user also sees which chunks were retrieved (expandable "Sources" section
below the assistant reply) so the system is transparent and debuggable.
"""

import streamlit as st

from utils.ollama_client import OllamaClient, OllamaConnectionError


USER_AVATAR = "🧑‍💻"
ASSISTANT_AVATAR = "🤖"


def render_chat_history() -> None:
    """
    Re-draw every previous message from session_state.
    Unchanged from v1 — history entries are plain dicts.
    """
    messages: list[dict] = st.session_state.get("messages", [])

    if not messages:
        # Show different empty state depending on RAG status
        if st.session_state.get("rag_enabled") and st.session_state.get("rag_filename"):
            empty_msg = (
                f"📄 **{st.session_state.rag_filename}** is loaded.  \n"
                "Ask anything about this document…"
            )
        else:
            empty_msg = "💬 Start a conversation below…"

        st.markdown(
            f"<div style='text-align:center; color: #888; margin-top: 3rem;'>"
            f"{empty_msg}</div>",
            unsafe_allow_html=True,
        )
        return

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        avatar = USER_AVATAR if role == "user" else ASSISTANT_AVATAR

        with st.chat_message(role, avatar=avatar):
            st.markdown(content)

            # Show retrieved sources under assistant messages (if stored)
            # This is purely informational — helps users understand RAG grounding.
            if role == "assistant" and msg.get("sources"):
                with st.expander("📚 Retrieved chunks (sources)", expanded=False):
                    for i, chunk in enumerate(msg["sources"], 1):
                        st.markdown(f"**Chunk {i}:**")
                        st.markdown(
                            f"<div style='background:#1e1e2e;padding:10px;"
                            f"border-radius:6px;font-size:0.85em;'>{chunk}</div>",
                            unsafe_allow_html=True,
                        )
                        if i < len(msg["sources"]):
                            st.divider()


def render_chat_input() -> None:
    """Render the sticky input and drive the request-response cycle."""
    placeholder = (
        "Ask about your document… (RAG active)"
        if st.session_state.get("rag_enabled") and st.session_state.get("rag_filename")
        else "Ask anything… (Shift+Enter for new line)"
    )

    if prompt := st.chat_input(placeholder=placeholder, key="chat_input"):
        _handle_user_message(prompt)


def _handle_user_message(prompt: str) -> None:
    """
    Core request-response loop with optional RAG enrichment.

    RAG path  (rag_enabled = True, collection exists):
      1. Retrieve top-K chunks from ChromaDB
      2. Build augmented prompt (context + question)
      3. Stream augmented prompt to Ollama
      4. Save reply + source chunks to session_state

    Plain path (rag_enabled = False):
      1. Stream plain prompt to Ollama  ← identical to v1
    """
    # ── 1. Persist user turn ──────────────────────────────────────────────────
    st.session_state.messages.append({"role": "user", "content": prompt})

    # ── 2. Render user bubble ─────────────────────────────────────────────────
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    # ── 3. RAG retrieval (only when enabled + a document is loaded) ───────────
    retrieved_chunks: list[str] = []
    prompt_to_send: str = prompt  # default: send the raw query

    rag_active = (
        st.session_state.get("rag_enabled", False)
        and st.session_state.get("rag_collection_name") is not None
    )

    if rag_active:
        from utils.rag import retrieve_context, build_rag_prompt
        from config.settings import AppSettings

        try:
            # Show a brief status while retrieving (retrieval is near-instant)
            with st.spinner("🔍 Searching document…"):
                retrieved_chunks = retrieve_context(
                    query=prompt,
                    collection_name=st.session_state.rag_collection_name,
                    top_k=st.session_state.get("rag_top_k", AppSettings.RAG_TOP_K),
                    persist_dir=AppSettings.CHROMA_PERSIST_DIR,
                )

            if retrieved_chunks:
                # Augment the prompt with retrieved context
                prompt_to_send = build_rag_prompt(prompt, retrieved_chunks)
            else:
                # No relevant chunks found — warn user but continue with plain prompt
                st.warning(
                    "⚠️ No relevant chunks found in the document for this query. "
                    "Answering from model knowledge instead.",
                    icon="🔍",
                )

        except Exception as e:
            st.error(f"⚠️ RAG retrieval failed: {e}. Falling back to plain mode.")
            prompt_to_send = prompt  # safe fallback
            retrieved_chunks = []

    # ── 4. Build messages list for Ollama ────────────────────────────────────
    # We send the full conversation history BUT replace the last user message
    # with the augmented prompt (the plain prompt is already in session_state).
    # This preserves multi-turn context while injecting RAG context.
    messages_for_ollama = st.session_state.messages[:-1]  # all but the last
    messages_for_ollama.append({"role": "user", "content": prompt_to_send})

    # ── 5. Stream assistant reply ─────────────────────────────────────────────
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        client: OllamaClient = st.session_state.ollama_client

        try:
            token_generator = client.stream_response(
                model=st.session_state.selected_model,
                messages=messages_for_ollama,
                temperature=st.session_state.get("temperature", 0.7),
                top_p=st.session_state.get("top_p", 0.9),
                max_tokens=st.session_state.get("max_tokens", 2048),
                system_prompt=st.session_state.get("system_prompt"),
            )

            # Streaming is IDENTICAL to v1 — RAG only changes what goes IN.
            full_reply: str = st.write_stream(token_generator)

        except OllamaConnectionError as exc:
            full_reply = f"⚠️ **Connection error:** {exc}"
            st.error(full_reply)

        except Exception as exc:
            full_reply = f"⚠️ **Unexpected error:** {exc}"
            st.error(full_reply)

        # Show retrieved sources inline (collapsible) right after the reply
        if retrieved_chunks:
            with st.expander("📚 Retrieved chunks (sources)", expanded=False):
                for i, chunk in enumerate(retrieved_chunks, 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.markdown(
                        f"<div style='background:#1e1e2e;padding:10px;"
                        f"border-radius:6px;font-size:0.85em;'>{chunk}</div>",
                        unsafe_allow_html=True,
                    )
                    if i < len(retrieved_chunks):
                        st.divider()

    # ── 6. Persist assistant turn (with sources for history display) ──────────
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_reply,
        "sources": retrieved_chunks,  # stored so render_chat_history() can show them
    })