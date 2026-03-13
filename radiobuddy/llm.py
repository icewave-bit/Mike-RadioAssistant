from __future__ import annotations

from typing import List, Optional

from openai import OpenAI

from .config import LlmConfig


class GptNanoClient:
    def __init__(self, cfg: LlmConfig) -> None:
        client_kwargs: dict[str, object] = {"api_key": cfg.api_key}
        if cfg.base_url:
            client_kwargs["base_url"] = cfg.base_url
        self._client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
        self._cfg = cfg
        # Alternating user/assistant chat history (system prompt is injected per call).
        self._history: List[dict[str, str]] = []

    def reset_history(self) -> None:
        """Clear conversation history for this client instance."""
        self._history.clear()

    def _trim_history(self, max_messages: int = 10) -> None:
        """Keep only the most recent messages to avoid unbounded growth."""
        if len(self._history) > max_messages:
            # Drop oldest messages while preserving order.
            self._history = self._history[-max_messages:]

    def chat(self, user_message: str) -> str:
        user_message = user_message.strip()
        if not user_message:
            return ""

        # Build full message list with system prompt and prior conversation.
        messages = [
            {"role": "system", "content": self._cfg.system_prompt},
            *self._history,
            {"role": "user", "content": user_message},
        ]

        resp = self._client.chat.completions.create(
            model=self._cfg.model,
            messages=messages,
        )
        choice = resp.choices[0]
        content: Optional[str] = choice.message.content if choice and choice.message else None
        reply = content or ""

        if reply:
            # Update history with this turn and trim.
            self._history.append({"role": "user", "content": user_message})
            self._history.append({"role": "assistant", "content": reply})
            self._trim_history()

        return reply


class DummyLlmClient:
    """
    Offline LLM stub used when no GPT5_NANO_API_KEY is configured.
    """

    def __init__(self) -> None:
        self._history: List[dict[str, str]] = []

    def reset_history(self) -> None:
        self._history.clear()

    def chat(self, user_message: str) -> str:
        user_message = user_message.strip()
        if not user_message:
            return "I received an empty message."

        # Simple echo-style response while still updating history
        reply = f"(dummy reply) You said: {user_message}"
        self._history.append({"role": "user", "content": user_message})
        self._history.append({"role": "assistant", "content": reply})
        if len(self._history) > 10:
            self._history = self._history[-10:]
        return reply


def build_llm_client(cfg: LlmConfig, mode: str = "ai"):
    """
    Build an LLM client based on configuration and app mode.

    In "dummy" mode we always return the offline dummy client, regardless of
    whether an API key is configured, so that the pipeline remains fully offline.
    """
    if mode == "dummy":
        return DummyLlmClient()

    # If no API key is set, fall back to a dummy LLM client so the
    # pipeline can be tested without external services.
    if not cfg.api_key:
        return DummyLlmClient()

    return GptNanoClient(cfg)

