from __future__ import annotations

import math
from dataclasses import dataclass
import threading
from typing import Callable, Optional, Tuple

import numpy as np
import sounddevice as sd


@dataclass
class AudioDevices:
    input_name: Optional[str]
    output_name: Optional[str]


def list_devices() -> None:
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        print(f"{idx}: {dev['name']} (in={dev['max_input_channels']}, out={dev['max_output_channels']})")


def _print_devices_filtered(kind: str) -> None:
    devices = sd.query_devices()
    print(f"Available {kind} devices:")
    for idx, dev in enumerate(devices):
        has_in = dev["max_input_channels"] > 0
        has_out = dev["max_output_channels"] > 0
        if kind == "input" and not has_in:
            continue
        if kind == "output" and not has_out:
            continue
        print(f"  {idx}: {dev['name']} (in={dev['max_input_channels']}, out={dev['max_output_channels']})")


def interactive_select_device(kind: str) -> Optional[str]:
    """
    Simple CLI helper to choose an input or output device.

    Returns the chosen index as a string, or None to keep the system default.
    """
    assert kind in ("input", "output")

    _print_devices_filtered(kind)
    prompt = (
        f"Enter {kind} device index (or press Enter for system default): "
    )
    while True:
        choice = input(prompt).strip()
        if choice == "":
            return None
        try:
            idx = int(choice)
        except ValueError:
            print("Please enter a numeric index or press Enter.")
            continue

        devices = sd.query_devices()
        if not (0 <= idx < len(devices)):
            print("Index out of range, try again.")
            continue

        dev = devices[idx]
        if kind == "input" and dev["max_input_channels"] <= 0:
            print("That device has no input channels; choose another.")
            continue
        if kind == "output" and dev["max_output_channels"] <= 0:
            print("That device has no output channels; choose another.")
            continue

        return choice


def resolve_device(name_or_index: Optional[str], kind: str) -> Optional[int]:
    if name_or_index is None or name_or_index == "":
        return None

    devices = sd.query_devices()

    # If it's an integer index, return directly if valid.
    try:
        idx = int(name_or_index)
        if 0 <= idx < len(devices):
            return idx
    except ValueError:
        pass

    # Otherwise treat as name substring.
    lowered = name_or_index.lower()
    candidates = [
        i
        for i, dev in enumerate(devices)
        if lowered in str(dev["name"]).lower()
        and ((kind == "input" and dev["max_input_channels"] > 0) or (kind == "output" and dev["max_output_channels"] > 0))
    ]
    return candidates[0] if candidates else None


def get_device_display_name(device_spec: Optional[str], kind: str) -> str:
    """
    Return a human-readable label for the current device (e.g. "USB Audio (index 2)" or "system default").
    """
    idx = resolve_device(device_spec, kind)
    if idx is None:
        return "system default"
    devices = sd.query_devices()
    if 0 <= idx < len(devices):
        return f"{devices[idx]['name']} (index {idx})"
    return "system default"


def rms_db(samples: np.ndarray) -> float:
    if samples.size == 0:
        return -120.0
    rms = math.sqrt(float(np.mean(np.square(samples), dtype=np.float64)))
    if rms <= 1e-9:
        return -120.0
    return 20.0 * math.log10(rms)


def record_segment_vox(
    input_device: Optional[int],
    sample_rate: int,
    chunk_seconds: float,
    threshold_db: float,
    min_duration_ms: int,
    silence_timeout_sec: float = 5.0,
    on_level: Optional[Callable[[float], None]] = None,
    on_started: Optional[Callable[[], None]] = None,
    on_chunk: Optional[Callable[[np.ndarray], None]] = None,
    stop_event: Optional[threading.Event] = None,
) -> Optional[np.ndarray]:
    """
    Block until a segment of audio above the VOX threshold is captured, then return it.

    After speech starts, a pause (below threshold) does not end the segment immediately:
    we keep recording and wait for silence_timeout_sec of consecutive silence. If speech
    resumes before that, the timer resets. So one segment can include multiple phrases
    with pauses up to silence_timeout_sec between them.

    If on_level is provided, it is called with the current RMS level in dB for each chunk.

    Returns None if recording is interrupted by user (Ctrl+C).
    """
    level_block_frames = max(256, int(sample_rate * 0.1))
    above_threshold_frames_needed = int(sample_rate * (min_duration_ms / 1000.0))
    silence_frames_timeout = int(sample_rate * silence_timeout_sec)

    buffer: list[np.ndarray] = []
    above_threshold_frames = 0
    started = False
    consecutive_silence_frames = 0

    try:
        with sd.InputStream(
            device=input_device,
            channels=1,
            samplerate=sample_rate,
            blocksize=level_block_frames,
            dtype="float32",
        ) as stream:
            while True:
                if stop_event is not None and stop_event.is_set():
                    return None
                data, _ = stream.read(level_block_frames)
                data = data.astype("float32").reshape(-1)
                if on_chunk is not None:
                    on_chunk(data)
                level_db = rms_db(data)
                if on_level is not None:
                    on_level(level_db)

                if level_db >= threshold_db:
                    buffer.append(data)
                    above_threshold_frames += data.size
                    consecutive_silence_frames = 0
                    if not started and above_threshold_frames >= above_threshold_frames_needed:
                        started = True
                        if on_started is not None:
                            on_started()
                else:
                    if started:
                        buffer.append(data)
                        consecutive_silence_frames += data.size
                        if consecutive_silence_frames >= silence_frames_timeout:
                            break
                    else:
                        buffer.clear()
                        above_threshold_frames = 0
    except KeyboardInterrupt:
        raise

    if not buffer:
        return None

    segment = np.concatenate(buffer)
    # Trim part of the trailing silence so we don't send the full
    # silence_timeout_sec to STT/repeater, but keep the last ~2 seconds
    # for a more natural tail.
    keep_silence_frames = int(sample_rate * 2.0)
    max_trim = max(0, silence_frames_timeout - keep_silence_frames)
    trim = min(len(segment), max_trim)
    if trim > 0:
        segment = segment[:-trim]
    if segment.size == 0:
        return None
    return segment


def play_audio(
    audio: np.ndarray,
    sample_rate: int,
    output_device: Optional[int],
) -> None:
    if audio.size == 0:
        return
    # Normalize to avoid clipping; simple peak normalization.
    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = (audio.astype("float64") / peak * 0.8).astype("float32")
    else:
        audio = audio.astype("float32")

    sd.play(audio, samplerate=sample_rate, device=output_device)
    sd.wait()

