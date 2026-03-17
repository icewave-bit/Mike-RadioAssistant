from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ToneControlConfig:
    enabled: bool = True
    secret: str = "0909"
    command_len: int = 2
    # Human-friendly timing: allow pauses between presses over radio.
    digit_gap_timeout_sec: float = 2.5
    command_timeout_sec: float = 12.0
    debounce_sec: float = 0.8


@dataclass(frozen=True)
class ToneEvent:
    kind: str  # "open" | "accepted" | "mode_set"
    command: str  # "0" or "01"/"02"/"03"
    program_mode: Optional[int] = None


class ToneController:
    """
    Interprets DTMF digit stream into:
      - secret (e.g. 0909) -> "open"
      - next 2 digits -> command ("01", "02", ...)

    Emits events; no audio/TTS side effects here.
    """

    def __init__(self, cfg: ToneControlConfig) -> None:
        self._cfg = cfg
        self._recent: str = ""
        self._last_digit_at: Optional[float] = None

        self._awaiting_cmd: bool = False
        self._cmd_started_at: Optional[float] = None
        self._cmd_digits: str = ""
        self._last_event_at: Optional[float] = None

    def feed_digit(self, digit: str, now: Optional[float] = None) -> list[ToneEvent]:
        if not self._cfg.enabled:
            return []
        now = time.monotonic() if now is None else now
        events: list[ToneEvent] = []

        # If operator paused too long between digits, reset current state.
        if self._last_digit_at is not None and (now - self._last_digit_at) > self._cfg.digit_gap_timeout_sec:
            self._reset()
        self._last_digit_at = now

        # Ignore non-numeric digits for this feature.
        if digit not in "0123456789":
            return events

        # If we're awaiting command, collect it.
        if self._awaiting_cmd:
            self._cmd_digits += digit
            if len(self._cmd_digits) >= self._cfg.command_len:
                cmd = self._cmd_digits[: self._cfg.command_len]
                events.extend(self._emit_accepted(cmd, now))
                self._reset()
            return events

        # Otherwise keep a rolling window for secret detection.
        self._recent = (self._recent + digit)[-max(16, len(self._cfg.secret)) :]
        if self._recent.endswith(self._cfg.secret):
            self._awaiting_cmd = True
            self._cmd_started_at = now
            self._cmd_digits = ""
            # Emit an "open" event so the pipeline can stop forwarding audio into STT/LLM
            # as soon as the secret is recognized (before command digits arrive).
            events.append(ToneEvent(kind="open", command=""))
        return events

    def tick(self, now: Optional[float] = None) -> list[ToneEvent]:
        """
        Call periodically (or on each audio chunk) to handle timeouts.
        """
        if not self._cfg.enabled:
            return []
        now = time.monotonic() if now is None else now
        events: list[ToneEvent] = []

        if self._awaiting_cmd and self._cmd_started_at is not None:
            if (now - self._cmd_started_at) >= self._cfg.command_timeout_sec:
                events.extend(self._emit_accepted("0", now))
                self._reset()

        return events

    def mode_set(self, program_mode: int, command: str) -> list[ToneEvent]:
        if not self._cfg.enabled:
            return []
        return [ToneEvent(kind="mode_set", command=command, program_mode=program_mode)]

    def _emit_accepted(self, cmd: str, now: float) -> list[ToneEvent]:
        if self._last_event_at is not None and (now - self._last_event_at) < self._cfg.debounce_sec:
            return []
        self._last_event_at = now
        return [ToneEvent(kind="accepted", command=cmd)]

    def _reset(self) -> None:
        self._recent = ""
        self._last_digit_at = None
        self._awaiting_cmd = False
        self._cmd_started_at = None
        self._cmd_digits = ""

