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
    timeout_seconds: float
    max_retries: int
    backoff_base_seconds: float


@dataclass
class LlmConfig:
    api_key: str
    base_url: Optional[str]
    model: str
    system_prompt: str
    timeout_seconds: float
    max_retries: int
    backoff_base_seconds: float


@dataclass
class TtsConfig:
    voice: Optional[str]
    piper_model_path: Optional[str]
    rate: Optional[int]


@dataclass
class AppConfig:
    audio: AudioConfig
    vox: VoxConfig
    stt: SttConfig
    llm: LlmConfig
    tts: TtsConfig
    mode: str
    reset_phrases: list[str]
    dtmf_enabled: bool
    dtmf_secret: str
    dtmf_command_timeout_sec: float
    dtmf_digit_gap_timeout_sec: float
    dtmf_min_tone_ms: int
    dtmf_energy_gate_db: float
    dtmf_peak_ratio: float
    dtmf_debug: bool
    dtmf_frame_ms: int
    dtmf_hop_ms: int
    dtmf_bandpass_enabled: bool


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
        timeout_seconds=float(os.getenv("STT_TIMEOUT_SECONDS", "20")),
        max_retries=int(os.getenv("STT_MAX_RETRIES", "3")),
        backoff_base_seconds=float(os.getenv("STT_BACKOFF_BASE_SECONDS", "1")),
    )

    llm = LlmConfig(
        api_key=os.getenv("GPT5_NANO_API_KEY", ""),
        base_url=os.getenv("GPT5_NANO_BASE_URL") or None,
        model=os.getenv("GPT5_NANO_MODEL", "gpt-5-nano"),
        system_prompt=os.getenv(
            "AI_SYSTEM_PROMPT",
            "You are a concise VHF radio assistant. Reply in short, clear sentences.",
        ),
        timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "20")),
        max_retries=int(os.getenv("LLM_MAX_RETRIES", "3")),
        backoff_base_seconds=float(os.getenv("LLM_BACKOFF_BASE_SECONDS", "1")),
    )

    tts = TtsConfig(
        voice=os.getenv("MACOS_TTS_VOICE") or None,
        piper_model_path=os.getenv("PIPER_MODEL_PATH") or None,
        rate=int(os.getenv("MACOS_TTS_RATE")) if os.getenv("MACOS_TTS_RATE") else None,
    )

    mode = os.getenv("MODE", "ai").strip().lower()
    if mode not in ("ai", "dummy", "repeater"):
        mode = "ai"

    # Voice commands that reset the LLM conversation history.
    # Use a delimiter unlikely to appear in phrases themselves.
    default_reset_phrases = "|".join(
        [
            "reset radiobuddy",
            "reset radio buddy",
            "clear history",
            "reset conversation",
            "reset brain",
            "start fresh",
            "wipe brains",
            "history delete",
            "delete history",
            "forget history",
            "clear brain",
            "forget brain",
            "forget everything",
            "forget all",
            "mike, forget",
            "mike, forget history",
            "mike, forget brain",
            "mike, forget everything",
            "mike, forget all",
            "mike, reset",
            "mike, reset conversation",
            "mike, reset brain",
            "mike, start fresh",
            "mike, wipe brains",
            "mike, history delete",
            "mike, delete history",
            "mike, clear brain",
            "new mike",
        ]
    )
    reset_raw = os.getenv("RADIOBUDDY_RESET_PHRASES", default_reset_phrases)
    reset_phrases = [p.strip().lower() for p in reset_raw.split("|") if p.strip()]

    dtmf_enabled = (os.getenv("DTMF_ENABLED", "1").strip().lower() not in ("0", "false", "no", "off"))
    dtmf_secret = (os.getenv("DTMF_SECRET", "0909").strip() or "0909")
    dtmf_command_timeout_sec = float(os.getenv("DTMF_COMMAND_TIMEOUT_SEC", "12"))
    dtmf_digit_gap_timeout_sec = float(os.getenv("DTMF_DIGIT_GAP_TIMEOUT_SEC", "2.5"))
    dtmf_frame_ms = int(os.getenv("DTMF_FRAME_MS", "80"))
    dtmf_hop_ms = int(os.getenv("DTMF_HOP_MS", "40"))
    dtmf_min_tone_ms = int(os.getenv("DTMF_MIN_TONE_MS", "120"))
    dtmf_energy_gate_db = float(os.getenv("DTMF_ENERGY_GATE_DB", "-50"))
    dtmf_peak_ratio = float(os.getenv("DTMF_PEAK_RATIO", "2.8"))
    dtmf_bandpass_enabled = (os.getenv("DTMF_BANDPASS_ENABLED", "1").strip().lower() not in ("0", "false", "no", "off"))
    dtmf_debug = (os.getenv("DTMF_DEBUG", "0").strip().lower() in ("1", "true", "yes", "on"))

    return AppConfig(
        audio=audio,
        vox=vox,
        stt=stt,
        llm=llm,
        tts=tts,
        mode=mode,
        reset_phrases=reset_phrases,
        dtmf_enabled=dtmf_enabled,
        dtmf_secret=dtmf_secret,
        dtmf_command_timeout_sec=dtmf_command_timeout_sec,
        dtmf_digit_gap_timeout_sec=dtmf_digit_gap_timeout_sec,
        dtmf_min_tone_ms=dtmf_min_tone_ms,
        dtmf_energy_gate_db=dtmf_energy_gate_db,
        dtmf_peak_ratio=dtmf_peak_ratio,
        dtmf_debug=dtmf_debug,
        dtmf_frame_ms=dtmf_frame_ms,
        dtmf_hop_ms=dtmf_hop_ms,
        dtmf_bandpass_enabled=dtmf_bandpass_enabled,
    )

