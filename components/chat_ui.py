"""
components/chat_ui.py — Chat history display + input handling
=============================================================
This module owns two things:

  1. render_chat_history()  — loops over session_state["messages"] and
                              draws each turn with the correct avatar.

  2. render_chat_input()    — renders the text box at the bottom, waits
                              for a submission, then drives the full
                              request → stream → display cycle.

KEY STREAMLIT CHAT PRIMITIVES (explained):
─────────────────────────────────────────
  st.chat_message(role)
      Context manager that wraps everything inside a chat bubble.
      role = "user"      → right-aligned, human avatar  👤
      role = "assistant" → left-aligned,  bot avatar    🤖
      You can also pass a custom avatar: st.chat_message("assistant", avatar="🦙")

  st.chat_input(placeholder)
      Renders a sticky input box pinned to the bottom of the page.
      Returns None when empty, returns the string when submitted.
      Importantly: it does NOT rerun unless the user actually submits,
      so you can safely check `if prompt := st.chat_input(...)`.

  st.write_stream(generator)
      Takes any Python generator that yields strings and streams them
      into the chat bubble token-by-token.  Returns the full concatenated
      string when the generator is exhausted — very handy for saving the
      complete reply to session_state.

HOW STREAMING WORKS END-TO-END:
────────────────────────────────
  1. User submits message → appended to session_state["messages"]
  2. User bubble rendered immediately (instant feedback)
  3. Assistant bubble opened with st.chat_message("assistant")
  4. st.write_stream() is called with ollama_client.stream_response()
     which is a Python generator yielding one token per yield.
  5. As each token arrives from Ollama, Streamlit appends it to the
     bubble in real time — the user sees words appear progressively.
  6. When the generator finishes, st.write_stream() returns the full text.
  7. Full reply saved to session_state["messages"] for the next rerun.
"""

import streamlit as st

from utils.ollama_client import OllamaClient, OllamaConnectionError


# ── Avatars ───────────────────────────────────────────────────────────────────
USER_AVATAR = "🧑‍💻"
ASSISTANT_AVATAR = "🤖"


def render_chat_history() -> None:
    """
    Re-draw every previous message from session_state.

    WHY we redraw every time:
      Streamlit's execution model re-runs the entire script on each
      interaction.  There's no persistent DOM between runs.  So we must
      re-render the full history from session_state on every rerun.
      This is fast because Streamlit diffs the virtual DOM internally.
    """
    messages: list[dict] = st.session_state.get("messages", [])

    if not messages:
        # Friendly empty-state prompt
        st.markdown(
            "<div style='text-align:center; color: #888; margin-top: 3rem;'>"
            "💬 Start a conversation below…"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        avatar = USER_AVATAR if role == "user" else ASSISTANT_AVATAR

        with st.chat_message(role, avatar=avatar):
            # st.markdown lets assistant replies contain **bold**, tables, code
            # blocks, etc. without any extra work.
            st.markdown(content)


def render_chat_input() -> None:
    """
    Render the sticky input box and handle the full request-response cycle.

    This function returns immediately when no message is submitted.
    When a message IS submitted, it:
      1. Appends the user turn to session_state
      2. Streams the assistant reply
      3. Appends the assistant turn to session_state
    """
    # st.chat_input pins the box to the bottom of the viewport.
    # The walrus operator (:=) assigns AND tests in one expression.
    if prompt := st.chat_input(
        placeholder="Ask anything… (Shift+Enter for new line)",
        key="chat_input",
    ):
        _handle_user_message(prompt)


def _handle_user_message(prompt: str) -> None:
    """
    Core loop: user message in → streamed assistant reply out.

    Kept private (leading underscore) because only render_chat_input
    should call it.
    """
    # ── 1. Persist the user turn ──────────────────────────────────────────────
    user_msg = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_msg)

    # ── 2. Render the user bubble immediately ─────────────────────────────────
    # Users expect instant visual feedback before the model even starts.
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    # ── 3. Open the assistant bubble and stream into it ───────────────────────
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        client: OllamaClient = st.session_state.ollama_client

        try:
            # Build the token generator — this does NOT block yet.
            # The generator only advances when iterated (lazy evaluation).
            token_generator = client.stream_response(
                model=st.session_state.selected_model,
                messages=st.session_state.messages,
                temperature=st.session_state.get("temperature", 0.7),
                top_p=st.session_state.get("top_p", 0.9),
                max_tokens=st.session_state.get("max_tokens", 2048),
                system_prompt=st.session_state.get("system_prompt"),
            )

            # st.write_stream() drives the generator, appending each yielded
            # token to the chat bubble in real time.
            # It also returns the full text once the generator is exhausted.
            full_reply: str = st.write_stream(token_generator)

        except OllamaConnectionError as exc:
            # Surface a friendly error inside the assistant bubble.
            full_reply = f"⚠️ **Connection error:** {exc}"
            st.error(full_reply)

        except Exception as exc:
            # Catch-all so a bug never leaves the UI in a broken state.
            full_reply = f"⚠️ **Unexpected error:** {exc}"
            st.error(full_reply)

    # ── 4. Persist the assistant turn ────────────────────────────────────────
    # Must happen AFTER streaming so we store the complete reply, not an empty
    # string.  On the next rerun, render_chat_history() will show this turn.
    assistant_msg = {"role": "assistant", "content": full_reply}
    st.session_state.messages.append(assistant_msg)
