from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


DTMF_LOW = (697.0, 770.0, 852.0, 941.0)
DTMF_HIGH = (1209.0, 1336.0, 1477.0, 1633.0)

DTMF_MAP = {
    (697.0, 1209.0): "1",
    (697.0, 1336.0): "2",
    (697.0, 1477.0): "3",
    (697.0, 1633.0): "A",
    (770.0, 1209.0): "4",
    (770.0, 1336.0): "5",
    (770.0, 1477.0): "6",
    (770.0, 1633.0): "B",
    (852.0, 1209.0): "7",
    (852.0, 1336.0): "8",
    (852.0, 1477.0): "9",
    (852.0, 1633.0): "C",
    (941.0, 1209.0): "*",
    (941.0, 1336.0): "0",
    (941.0, 1477.0): "#",
    (941.0, 1633.0): "D",
}


def _goertzel_power(x: np.ndarray, sample_rate: int, freq_hz: float) -> float:
    """
    Goertzel power estimate for a single tone frequency.
    x must be 1D float32/float64.
    """
    n = int(x.size)
    if n <= 0:
        return 0.0
    # Use the exact target frequency (not an FFT bin) to reduce bin-mismatch issues
    # for short frames and arbitrary sample rates.
    w = 2.0 * math.pi * (float(freq_hz) / float(sample_rate))
    coeff = 2.0 * math.cos(w)

    s_prev = 0.0
    s_prev2 = 0.0
    for v in x:
        s = float(v) + coeff * s_prev - s_prev2
        s_prev2 = s_prev
        s_prev = s
    return s_prev2 * s_prev2 + s_prev * s_prev - coeff * s_prev * s_prev2


@dataclass(frozen=True)
class DtmfDecoderConfig:
    sample_rate: int
    # Radio audio is often distorted/noisy; use slightly longer frames by default.
    frame_ms: int = 80
    hop_ms: int = 40
    min_tone_ms: int = 120
    min_silence_ms: int = 80
    energy_gate_db: float = -50.0
    peak_ratio: float = 2.8
    bandpass_enabled: bool = True


class DtmfDecoder:
    """
    Streaming DTMF decoder.

    Feed float32 mono samples via process(), get digits when stable.
    """

    def __init__(self, cfg: DtmfDecoderConfig) -> None:
        self._cfg = cfg
        self._frame_len = max(64, int(cfg.sample_rate * (cfg.frame_ms / 1000.0)))
        self._hop_len = max(32, int(cfg.sample_rate * (cfg.hop_ms / 1000.0)))
        self._buf = np.zeros(0, dtype="float32")

        self._active_symbol: Optional[str] = None
        self._active_ms = 0.0
        self._silence_ms = 0.0
        # Latch ensures "one digit per keypress": after we emit a symbol, we won't emit
        # the same symbol again until we observe enough silence.
        self._latched_symbol: Optional[str] = None
        self._hp_state = 0.0
        self._lp_state = 0.0

    def process(self, samples: np.ndarray) -> list[str]:
        if samples.size == 0:
            return []
        if samples.dtype != np.float32:
            samples = samples.astype("float32")

        self._buf = np.concatenate([self._buf, samples])
        out: list[str] = []

        while self._buf.size >= self._frame_len:
            frame = self._buf[: self._frame_len]
            self._buf = self._buf[self._hop_len :] if self._buf.size > self._hop_len else np.zeros(0, dtype="float32")

            if self._cfg.bandpass_enabled:
                frame = self._bandpass_650_1700(frame)

            # Energy gate (skip very quiet frames)
            rms = float(np.sqrt(np.mean(frame.astype("float64") ** 2))) if frame.size else 0.0
            rms_db = 20.0 * math.log10(rms + 1e-12)
            if rms_db < self._cfg.energy_gate_db:
                out.extend(self._update_symbol(None))
                continue

            low_pows = [(f, _goertzel_power(frame, self._cfg.sample_rate, f)) for f in DTMF_LOW]
            high_pows = [(f, _goertzel_power(frame, self._cfg.sample_rate, f)) for f in DTMF_HIGH]

            low_pows.sort(key=lambda t: t[1], reverse=True)
            high_pows.sort(key=lambda t: t[1], reverse=True)

            low_f, low_1 = low_pows[0]
            high_f, high_1 = high_pows[0]
            low_2 = low_pows[1][1]
            high_2 = high_pows[1][1]

            # Require a clear winner in each band to reduce false positives.
            if low_2 <= 0.0 or high_2 <= 0.0:
                out.extend(self._update_symbol(None))
                continue
            if (low_1 / low_2) < self._cfg.peak_ratio or (high_1 / high_2) < self._cfg.peak_ratio:
                out.extend(self._update_symbol(None))
                continue

            sym = DTMF_MAP.get((low_f, high_f))
            out.extend(self._update_symbol(sym))

        return out

    def _update_symbol(self, sym: Optional[str]) -> list[str]:
        frame_ms = float(self._cfg.hop_ms)
        emitted: list[str] = []

        if sym is None:
            self._silence_ms += frame_ms
            self._active_ms = 0.0
            if self._silence_ms >= self._cfg.min_silence_ms:
                self._active_symbol = None
                self._latched_symbol = None
            return emitted

        self._silence_ms = 0.0
        if self._latched_symbol == sym:
            # Same key still held; wait for silence to clear the latch.
            return emitted
        if sym != self._active_symbol:
            self._active_symbol = sym
            self._active_ms = 0.0
            return emitted

        self._active_ms += frame_ms
        if self._active_ms >= self._cfg.min_tone_ms:
            # Emit once per tone "press", then wait for silence reset.
            emitted.append(sym)
            self._active_ms = 0.0
            self._latched_symbol = sym

        return emitted

    def _bandpass_650_1700(self, x: np.ndarray) -> np.ndarray:
        """
        Cheap IIR band-pass (HP @ ~650 Hz then LP @ ~1700 Hz).
        Not audiophile-grade; just enough to suppress speech/rumble/hiss outside DTMF bands.
        """
        sr = float(self._cfg.sample_rate)
        # 1st order HP: y[n] = a*(y[n-1] + x[n] - x[n-1])
        hp_fc = 650.0
        hp_a = math.exp(-2.0 * math.pi * hp_fc / sr)
        y = np.empty_like(x, dtype="float32")
        prev_x = 0.0
        prev_y = float(self._hp_state)
        for i, v in enumerate(x):
            yv = (1.0 - hp_a) * (prev_y + float(v) - prev_x)
            prev_x = float(v)
            prev_y = yv
            y[i] = float(yv)
        self._hp_state = prev_y

        # 1st order LP: y[n] = (1-a)*x[n] + a*y[n-1]
        lp_fc = 1700.0
        lp_a = math.exp(-2.0 * math.pi * lp_fc / sr)
        z = np.empty_like(y, dtype="float32")
        prev = float(self._lp_state)
        for i, v in enumerate(y):
            prev = (1.0 - lp_a) * float(v) + lp_a * prev
            z[i] = float(prev)
        self._lp_state = prev
        return z


def synthesize_dtmf(
    digits: str,
    sample_rate: int = 8000,
    tone_ms: int = 140,
    gap_ms: int = 70,
    amplitude: float = 0.6,
) -> np.ndarray:
    """
    Generate a clean DTMF waveform for testing/recording.
    Returns float32 mono samples in [-1, 1].
    """
    # Map symbol -> (low, high)
    inv: dict[str, tuple[float, float]] = {v: k for k, v in DTMF_MAP.items()}

    def tone(sym: str) -> np.ndarray:
        pair = inv.get(sym)
        if pair is None:
            raise ValueError(f"Unsupported DTMF symbol: {sym}")
        f_lo, f_hi = pair
        n = int(sample_rate * (tone_ms / 1000.0))
        t = (np.arange(n, dtype="float32") / float(sample_rate)).astype("float32")
        x = (np.sin(2.0 * math.pi * f_lo * t) + np.sin(2.0 * math.pi * f_hi * t)) * 0.5
        # Short fade to reduce clicks
        fade = int(max(1, min(n // 10, sample_rate * 0.005)))
        w = np.ones(n, dtype="float32")
        ramp = np.linspace(0.0, 1.0, fade, dtype="float32")
        w[:fade] = ramp
        w[-fade:] = ramp[::-1]
        return (x * w * float(amplitude)).astype("float32")

    gap = np.zeros(int(sample_rate * (gap_ms / 1000.0)), dtype="float32")
    parts: list[np.ndarray] = []
    for ch in digits:
        if ch.strip() == "":
            continue
        parts.append(tone(ch))
        parts.append(gap)
    return np.concatenate(parts) if parts else np.zeros(0, dtype="float32")

