from __future__ import annotations

import tempfile
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
        self._client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """
        Transcribe a mono float32 numpy array using Whisper.
        """
        if audio.size == 0:
            return ""

        # Write to a temporary WAV file for the Whisper API.
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            sf.write(tmp.name, audio, sample_rate)
            with open(tmp.name, "rb") as f:
                resp = self._client.audio.transcriptions.create(
                    model=self._cfg.model,
                    file=f,
                    language=self._cfg.language,
                )
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
        return "test message over radio"


def build_stt_client(cfg: SttConfig):
    # For now we only support Whisper; provider switch is future work.
    if cfg.provider.lower() != "whisper":
        raise ValueError(f"Unsupported STT provider: {cfg.provider}")

    # If no API key is set, fall back to a dummy STT client so the
    # pipeline can be tested without external services.
    if not cfg.api_key:
        return DummySttClient()

    return WhisperSttClient(cfg)

