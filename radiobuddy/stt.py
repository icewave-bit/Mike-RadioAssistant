from __future__ import annotations

import tempfile
import time
from typing import Optional
import numpy as np
import soundfile as sf
from openai import OpenAI

from .config import SttConfig


class WhisperSttClient:
    def __init__(self, cfg: SttConfig) -> None:
        self._cfg = cfg
        client_kwargs: dict[str, object] = {"api_key": cfg.api_key}
        if cfg.base_url:
            client_kwargs["base_url"] = cfg.base_url
        self._client = OpenAI(**client_kwargs)
        # Basic network robustness settings, driven by configuration.
        self._timeout_seconds = cfg.timeout_seconds
        self._max_retries = cfg.max_retries
        self._backoff_base_seconds = cfg.backoff_base_seconds

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """
        Transcribe a mono float32 numpy array using Whisper.
        """
        if audio.size == 0:
            return ""

        # Write to a temporary WAV file for the Whisper API.
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            sf.write(tmp.name, audio, sample_rate)
            last_error: Optional[Exception] = None
            for attempt in range(1, self._max_retries + 1):
                try:
                    with open(tmp.name, "rb") as f:
                        resp = self._client.audio.transcriptions.create(
                            model=self._cfg.model,
                            file=f,
                            language=self._cfg.language,
                            timeout=self._timeout_seconds,
                        )
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    if attempt >= self._max_retries:
                        raise
                    sleep_for = self._backoff_base_seconds * (2 ** (attempt - 1))
                    time.sleep(sleep_for)
        text: Optional[str] = getattr(resp, "text", None)
        return text or ""


class DummySttClient:
    """
    Offline STT stub used when no WHISPER_API_KEY is configured.
    """

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:  # noqa: ARG002
        if audio.size == 0:
            return ""
        # Simple fixed transcription just to exercise the pipeline.
        return "пробное сообщение по радио"


def build_stt_client(cfg: SttConfig, mode: str = "ai"):
    # For now we only support Whisper; provider switch is future work.
    if cfg.provider.lower() != "whisper":
        raise ValueError(f"Unsupported STT provider: {cfg.provider}")

    # In "dummy" mode we always return the offline dummy client, regardless of
    # whether an API key is configured, so that the pipeline remains fully offline.
    if mode == "dummy":
        return DummySttClient()

    # If no API key is set, fall back to a dummy STT client so the
    # pipeline can be tested without external services.
    if not cfg.api_key:
        return DummySttClient()

    return WhisperSttClient(cfg)

