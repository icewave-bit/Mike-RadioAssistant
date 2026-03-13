from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


@dataclass
class AudioConfig:
    input_device: Optional[str]
    output_device: Optional[str]
    sample_rate: int
    chunk_seconds: float


@dataclass
class VoxConfig:
    threshold_db: float
    min_duration_ms: int
    silence_timeout_sec: float


@dataclass
class SttConfig:
    provider: str
    api_key: str
    model: str
    base_url: Optional[str]
    language: Optional[str]


@dataclass
class LlmConfig:
    api_key: str
    base_url: Optional[str]
    model: str
    system_prompt: str


@dataclass
class TtsConfig:
    voice: Optional[str]
    piper_model_path: Optional[str]


@dataclass
class AppConfig:
    audio: AudioConfig
    vox: VoxConfig
    stt: SttConfig
    llm: LlmConfig
    tts: TtsConfig
    mode: str


def load_config() -> AppConfig:
    load_dotenv()

    audio = AudioConfig(
        input_device=os.getenv("AUDIO_INPUT_DEVICE") or None,
        output_device=os.getenv("AUDIO_OUTPUT_DEVICE") or None,
        sample_rate=int(os.getenv("SAMPLE_RATE", "16000")),
        chunk_seconds=float(os.getenv("CHUNK_SECONDS", "3")),
    )

    vox = VoxConfig(
        threshold_db=float(os.getenv("VOX_THRESHOLD_DB", "-35")),
        min_duration_ms=int(os.getenv("VOX_MIN_DURATION_MS", "300")),
        silence_timeout_sec=float(os.getenv("VOX_SILENCE_TIMEOUT_SEC", "5")),
    )

    stt = SttConfig(
        provider=os.getenv("STT_PROVIDER", "whisper"),
        api_key=os.getenv("WHISPER_API_KEY", ""),
        model=os.getenv("STT_MODEL", "whisper-1"),
        base_url=os.getenv("WHISPER_BASE_URL") or os.getenv("GPT5_NANO_BASE_URL") or None,
        language=os.getenv("STT_LANGUAGE") or None,
    )

    llm = LlmConfig(
        api_key=os.getenv("GPT5_NANO_API_KEY", ""),
        base_url=os.getenv("GPT5_NANO_BASE_URL") or None,
        model=os.getenv("GPT5_NANO_MODEL", "gpt-5-nano"),
        system_prompt=os.getenv(
            "AI_SYSTEM_PROMPT",
            "You are a concise VHF radio assistant. Reply in short, clear sentences.",
        ),
    )

    tts = TtsConfig(
        voice=os.getenv("MACOS_TTS_VOICE") or None,
        piper_model_path=os.getenv("PIPER_MODEL_PATH") or None,
    )

    mode = os.getenv("MODE", "ai").strip().lower()
    if mode not in ("ai", "dummy", "repeater"):
        mode = "ai"

    return AppConfig(
        audio=audio,
        vox=vox,
        stt=stt,
        llm=llm,
        tts=tts,
        mode=mode,
    )

