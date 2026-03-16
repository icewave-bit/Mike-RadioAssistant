from __future__ import annotations

import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import soundfile as sf

from .config import TtsConfig


class MacTts:
    def __init__(self, cfg: TtsConfig) -> None:
        self._cfg = cfg

    def synthesize_to_array(self, text: str) -> Tuple[np.ndarray, int]:
        """
        Use macOS 'say' to synthesize text to a temp audio file, then load into a numpy array.
        """
        if not text.strip():
            return np.zeros(0, dtype="float32"), 16000

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "tts.aiff"
            # Ensure voice option is applied before output option so macOS 'say' respects it.
            cmd = ["say"]
            if self._cfg.voice:
                cmd.extend(["-v", self._cfg.voice])
            if getattr(self._cfg, "rate", None):
                cmd.extend(["-r", str(self._cfg.rate)])
            cmd.extend(["-o", str(out_path), text])
            subprocess.run(cmd, check=True)

            data, sr = sf.read(str(out_path), dtype="float32")
            if data.ndim == 2:
                data = data.mean(axis=1)
            return data.astype("float32"), int(sr)


class PiperTts:
    """
    Local TTS on Linux using Piper (piper-tts). Uses an .onnx voice model; no network.
    """

    def __init__(self, cfg: TtsConfig) -> None:
        if not cfg.piper_model_path:
            raise ValueError(
                "PIPER_MODEL_PATH is required on Linux. "
                "Set it to the path of a Piper .onnx voice model (e.g. from piper-tts download)."
            )
        self._cfg = cfg
        from piper import PiperVoice

        self._voice = PiperVoice.load(cfg.piper_model_path)

    def synthesize_to_array(self, text: str) -> Tuple[np.ndarray, int]:
        if not text.strip():
            return np.zeros(0, dtype="float32"), 16000

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
        try:
            with wave.open(wav_path, "wb") as wav_file:
                self._voice.synthesize_wav(text, wav_file)
            data, sr = sf.read(wav_path, dtype="float32")
            if data.ndim == 2:
                data = data.mean(axis=1)
            return data.astype("float32"), int(sr)
        finally:
            Path(wav_path).unlink(missing_ok=True)


def build_tts_client(cfg: TtsConfig) -> Union[MacTts, PiperTts]:
    if sys.platform == "darwin":
        return MacTts(cfg)
    if sys.platform == "linux" or sys.platform.startswith("linux"):
        try:
            import piper  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "Piper TTS is required on Linux. Install with: pip install piper-tts"
            ) from e
        return PiperTts(cfg)
    raise RuntimeError(f"TTS is not supported on this platform: {sys.platform}")
