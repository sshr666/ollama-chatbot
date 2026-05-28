"""
utils/ollama_client.py — Low-level Ollama API communication layer
=================================================================
Ollama exposes a local REST API (default: http://localhost:11434).
The two endpoints we use here are:

  GET  /api/tags          → list installed models
  POST /api/chat          → multi-turn chat with optional streaming

WHY a dedicated client class?
  Keeping all HTTP logic here means app.py and UI components never
  import `requests` directly.  When you later want to:
    • swap Ollama for LM Studio
    • add retry logic / circuit breakers
    • log latency metrics
  …you change ONLY this file.

Streaming explained:
  When `stream=True` is sent in the JSON body, Ollama does NOT wait for
  the full reply before responding.  Instead it sends a series of newline-
  delimited JSON objects (NDJSON), each containing one token:

    {"model":"llama3","message":{"role":"assistant","content":"Hi"},"done":false}
    {"model":"llama3","message":{"role":"assistant","content":" there"},"done":false}
    …
    {"model":"llama3","message":{"role":"assistant","content":"!"},"done":true}

  We use `requests` with `stream=True` to read this line-by-line and
  `yield` each token.  Streamlit's `st.write_stream()` consumes the
  generator and updates the UI in real time.
"""

from __future__ import annotations

import json
import logging
from typing import Generator, Any

import requests

logger = logging.getLogger(__name__)


class OllamaConnectionError(Exception):
    """Raised when we can't reach the Ollama server at all."""


class OllamaClient:
    """
    Thin wrapper around the Ollama REST API.

    Parameters
    ----------
    base_url : str
        Root URL of the Ollama instance, e.g. "http://localhost:11434"
    timeout : int
        Seconds to wait before giving up on a request.
    """

    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ── Public helpers ────────────────────────────────────────────────────────

    def is_running(self) -> bool:
        """
        Quick health-check: returns True if Ollama is reachable.
        Used by the sidebar to show a green/red status indicator.
        """
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except requests.exceptions.ConnectionError:
            return False

    def list_models(self) -> list[str]:
        """
        Returns the names of every model currently pulled on this machine.
        Example return value: ["llama3:latest", "qwen3:8b", "mistral:latest"]
        """
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            # The API returns {"models": [{"name": "...", ...}, ...]}
            return [m["name"] for m in data.get("models", [])]
        except requests.exceptions.ConnectionError as exc:
            raise OllamaConnectionError(
                "Cannot reach Ollama. Is it running? Try: ollama serve"
            ) from exc
        except Exception as exc:
            logger.error("Failed to list models: %s", exc)
            return []

    def stream_response(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 2048,
        system_prompt: str | None = None,
    ) -> Generator[str, None, None]:
        """
        Stream a chat completion token by token.

        Parameters
        ----------
        model : str
            Ollama model name, e.g. "llama3" or "qwen3:8b"
        messages : list[dict]
            Full conversation so far in OpenAI-style format:
            [{"role": "user", "content": "..."}, ...]
        temperature, top_p, max_tokens : generation knobs
        system_prompt : str | None
            If provided, prepended as a {"role": "system", ...} message.

        Yields
        ------
        str
            Individual text tokens as they arrive from the model.

        Raises
        ------
        OllamaConnectionError
            If Ollama isn't running.
        requests.HTTPError
            If Ollama returns a non-2xx status (e.g. model not found).
        """
        # Build the full message list, optionally prepending a system turn.
        full_messages: list[dict[str, str]] = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        payload: dict[str, Any] = {
            "model": model,
            "messages": full_messages,
            "stream": True,          # ← key flag: enables NDJSON streaming
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
            },
        }

        try:
            # `stream=True` on the requests side means we read the response
            # body incrementally instead of buffering the whole thing.
            with requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=self.timeout,
            ) as response:
                response.raise_for_status()  # blow up early on 4xx/5xx

                # iter_lines() yields each newline-delimited chunk.
                for raw_line in response.iter_lines():
                    if not raw_line:
                        continue  # skip keep-alive blank lines

                    try:
                        chunk = json.loads(raw_line)
                    except json.JSONDecodeError:
                        logger.warning("Could not parse chunk: %s", raw_line)
                        continue

                    # Each chunk has "message": {"content": "<token>"}
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield token

                    # Ollama sets "done": true on the final chunk.
                    if chunk.get("done", False):
                        break

        except requests.exceptions.ConnectionError as exc:
            raise OllamaConnectionError(
                "Lost connection to Ollama mid-stream. "
                "Make sure `ollama serve` is still running."
            ) from exc
