from __future__ import annotations

import logging
import queue
import threading
import time
from datetime import datetime
from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from .audio_io import play_audio, record_segment_vox, resolve_device
from .config import AppConfig
from .llm import GptNanoClient
from .stt import WhisperSttClient
from .tts import MacTts, PiperTts
from .dtmf import DtmfDecoder, DtmfDecoderConfig
from .tone_control import ToneController, ToneControlConfig, ToneEvent

if TYPE_CHECKING:
    from .console_ui import RadioBuddyUI

logger = logging.getLogger(__name__)


class RadioBuddyPipeline:
    def __init__(
        self,
        cfg: AppConfig,
        stt_client: WhisperSttClient,
        llm_client: GptNanoClient,
        tts_client: Union[MacTts, PiperTts],
        ui: Optional["RadioBuddyUI"] = None,
    ) -> None:
        self._cfg = cfg
        self._stt = stt_client
        self._llm = llm_client
        self._tts = tts_client
        self._ui = ui

        self._input_device_index: Optional[int] = resolve_device(cfg.audio.input_device, "input")
        self._output_device_index: Optional[int] = resolve_device(cfg.audio.output_device, "output")

        self._stop_event = threading.Event()
        self._tone_events: "queue.Queue[ToneEvent]" = queue.Queue()
        self._dtmf_debug = bool(getattr(cfg, "dtmf_debug", False))
        self._dtmf = DtmfDecoder(
            DtmfDecoderConfig(
                sample_rate=cfg.audio.sample_rate,
                frame_ms=int(getattr(cfg, "dtmf_frame_ms", 80)),
                hop_ms=int(getattr(cfg, "dtmf_hop_ms", 40)),
                min_tone_ms=int(getattr(cfg, "dtmf_min_tone_ms", 80)),
                energy_gate_db=float(getattr(cfg, "dtmf_energy_gate_db", -38.0)),
                peak_ratio=float(getattr(cfg, "dtmf_peak_ratio", 6.0)),
                bandpass_enabled=bool(getattr(cfg, "dtmf_bandpass_enabled", True)),
            )
        )
        self._tone = ToneController(
            ToneControlConfig(
                enabled=cfg.dtmf_enabled,
                secret=cfg.dtmf_secret,
                command_timeout_sec=cfg.dtmf_command_timeout_sec,
                digit_gap_timeout_sec=cfg.dtmf_digit_gap_timeout_sec,
            )
        )
        self._program_mode: int = 0
        self._dtmf_seq: str = ""
        self._last_dtmf_at: Optional[float] = None

        logger.info(
            "Using devices input=%s output=%s",
            self._input_device_index if self._input_device_index is not None else "default",
            self._output_device_index if self._output_device_index is not None else "default",
        )

    def _resample_to_config_rate(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        """Resample audio to the configured sample rate if needed."""
        target_sr = self._cfg.audio.sample_rate
        if sr == target_sr:
            return audio, sr

        factor = target_sr / sr
        new_len = int(len(audio) * factor)
        if new_len <= 0 or len(audio) == 0:
            return audio.astype("float32"), target_sr

        x_old = np.arange(len(audio))
        x_new = np.linspace(0, len(audio) - 1, new_len)
        resampled = np.interp(x_new, x_old, audio).astype("float32")
        return resampled, target_sr

    def _record_segment(self):
        return record_segment_vox(
            input_device=self._input_device_index,
            sample_rate=self._cfg.audio.sample_rate,
            chunk_seconds=self._cfg.audio.chunk_seconds,
            threshold_db=self._cfg.vox.threshold_db,
            min_duration_ms=self._cfg.vox.min_duration_ms,
            silence_timeout_sec=self._cfg.vox.silence_timeout_sec,
            on_chunk=lambda chunk: self._on_audio_chunk(chunk, self._stop_event),
            stop_event=self._stop_event,
        )

    def _speak_phrase(self, phrase: str) -> None:
        """Synthesize a short phrase and play it to the radio output."""
        if not phrase.strip():
            return

        audio, sr = self._tts.synthesize_to_array(phrase)
        logger.info(
            "Synthesized phrase TTS audio of %d samples at %d Hz",
            len(audio),
            sr,
        )

        audio, sr = self._resample_to_config_rate(audio, sr)

        logger.info("Playing phrase TTS to radio output")
        play_audio(audio, sr, self._output_device_index)

    def _on_audio_chunk(self, chunk: np.ndarray, stop_event: threading.Event) -> None:
        # Drive the tone controller even when no tone is detected.
        for ev in self._tone.tick():
            self._tone_events.put(ev)
            self._stop_event.set()
            stop_event.set()

        digits = self._dtmf.process(chunk)
        if digits:
            # Any detected DTMF is treated as a control-channel signal; do not forward
            # this audio into STT/LLM regardless of app mode.
            now = time.monotonic()
            if self._last_dtmf_at is None or (now - self._last_dtmf_at) > self._cfg.dtmf_digit_gap_timeout_sec:
                self._dtmf_seq = ""
            self._last_dtmf_at = now
            self._dtmf_seq = (self._dtmf_seq + "".join(digits))[-32:]
            logger.info("DTMF digits detected: %s", self._dtmf_seq)
            if self._ui is not None:
                self._ui.set_last_dtmf(self._dtmf_seq)
        for digit in digits:
            for ev in self._tone.feed_digit(digit):
                logger.info("DTMF control event: kind=%s command=%s", ev.kind, ev.command)
                if self._ui is not None:
                    self._ui.set_last_dtmf_event(f"{ev.kind} {ev.command}".strip())
                self._tone_events.put(ev)
                self._stop_event.set()
                stop_event.set()

    def _apply_program_mode(self, program_mode: int) -> None:
        """
        Program modes map to app operating modes:
          1 -> ai
          2 -> repeater
          3 -> dummy
        """
        mode_by_program = {1: "ai", 2: "repeater", 3: "dummy"}
        new_mode = mode_by_program.get(program_mode)
        if not new_mode:
            return
        if getattr(self._cfg, "mode", "ai") == new_mode:
            self._program_mode = program_mode
            return

        self._program_mode = program_mode
        self._cfg.mode = new_mode  # runtime switch (config is used as a state bag here)

        # Rebuild STT/LLM clients so the pipeline behavior actually changes.
        from .llm import build_llm_client
        from .stt import build_stt_client

        self._stt = build_stt_client(self._cfg.stt, self._cfg.mode)
        self._llm = build_llm_client(self._cfg.llm, self._cfg.mode)

    def _drain_tone_events(self) -> bool:
        """
        Handle pending tone events (TTS + mode switching).
        Returns True if at least one event was handled.
        """
        handled = False
        while True:
            try:
                ev = self._tone_events.get_nowait()
            except queue.Empty:
                break
            handled = True
            if ev.kind == "open":
                # Secret code recognized; do not speak yet. We'll wait for a command or timeout.
                continue
            if ev.kind == "accepted":
                cmd = ev.command
                self._speak_phrase(f"Кира передает - Код принят - команда {cmd}")

                cmd_to_mode = {"01": 1, "02": 2, "03": 3}
                program_mode = cmd_to_mode.get(cmd)
                if program_mode is not None:
                    self._apply_program_mode(program_mode)
                    for follow in self._tone.mode_set(program_mode=program_mode, command=cmd):
                        self._tone_events.put(follow)
                    continue

                if cmd == "11":
                    now = datetime.now()
                    hhmm = now.strftime("%H:%M")
                    self._speak_phrase(f"Кира передает - Время {hhmm}")
                    continue

            elif ev.kind == "mode_set":
                # Speak the *command* number as operator-friendly two digits ("01", "02", "03").
                self._speak_phrase(f"Кира передает - Режим {ev.command}")

        return handled

    def run_forever(self) -> None:
        logger.info("Starting RadioBuddy VOX loop")
        while True:
            if self._stop_event.is_set():
                self._stop_event.clear()
                if self._drain_tone_events():
                    continue

            if self._ui is not None:
                segment = self._ui.run_listening_until_segment(
                    lambda on_level, on_started, stop_event: record_segment_vox(
                        input_device=self._input_device_index,
                        sample_rate=self._cfg.audio.sample_rate,
                        chunk_seconds=self._cfg.audio.chunk_seconds,
                        threshold_db=self._cfg.vox.threshold_db,
                        min_duration_ms=self._cfg.vox.min_duration_ms,
                        silence_timeout_sec=self._cfg.vox.silence_timeout_sec,
                        on_level=on_level,
                        on_started=on_started,
                        on_chunk=lambda chunk: self._on_audio_chunk(chunk, stop_event),
                        stop_event=stop_event,
                    )
                )
            else:
                segment = self._record_segment()

            # If we aborted listening due to a DTMF control event, handle it now.
            if self._stop_event.is_set():
                self._stop_event.clear()
                if self._drain_tone_events():
                    continue

            if segment is None:
                logger.info("No segment captured (interrupted or quiet). Continuing.")
                continue

            logger.info("Captured segment of %.2f seconds", len(segment) / self._cfg.audio.sample_rate)

            # Repeater mode: no STT/LLM/TTS, just play back the captured audio.
            if getattr(self._cfg, "mode", "ai") == "repeater":
                if self._ui is not None:
                    self._ui.set_status("Repeating")
                logger.info("Repeater mode: playing back captured segment")
                play_audio(segment, self._cfg.audio.sample_rate, self._output_device_index)
                if self._ui is not None:
                    self._ui.set_status("Stand By")
                continue

            if self._ui is not None:
                self._ui.set_status("STT processing")
            try:
                text = self._stt.transcribe(segment, self._cfg.audio.sample_rate)
            except Exception:
                logger.exception("STT transcription failed.")
                if self._ui is not None:
                    self._ui.set_status("Error")
                try:
                    self._speak_phrase("Mike had a problem understanding you. Please try again.")
                except Exception:
                    logger.exception("Failed to synthesize STT error message.")
                continue
            if not text.strip():
                logger.info("STT returned empty transcription; skipping.")
                continue

            logger.info("Heard: %s", text)
            if self._ui is not None:
                self._ui.set_last_heard(text.strip())

            # Voice command to reset the AI conversation history.
            normalized = text.strip().lower()
            if any(phrase in normalized for phrase in self._cfg.reset_phrases):
                logger.info("Received history reset voice command; clearing LLM history.")
                reset_fn = getattr(self._llm, "reset_history", None)
                if callable(reset_fn):
                    reset_fn()
                if self._ui is not None:
                    self._ui.set_status("History reset.")

                # Speak an audible confirmation so the operator knows context was cleared.
                confirmation = "Hi, I am new Mike, ready to help!"
                if self._ui is not None:
                    self._ui.set_status("Transmitting")
                try:
                    self._speak_phrase(confirmation)
                except Exception:
                    logger.exception("Failed to synthesize reset confirmation message.")

                if self._ui is not None:
                    self._ui.set_status("Stand By")

                # Do not send this command text to the LLM.
                continue

            if self._ui is not None:
                self._ui.set_status("AI replying")
            try:
                reply = self._llm.chat(text)
            except Exception:
                logger.exception("LLM chat call failed.")
                if self._ui is not None:
                    self._ui.set_status("Error")
                try:
                    self._speak_phrase("Mike's brain does not brain. Please try again.")
                except Exception:
                    logger.exception("Failed to synthesize LLM error message.")
                continue
            if not reply.strip():
                logger.info("LLM returned empty reply.")
                if self._ui is not None:
                    self._ui.set_status("No reply")
                try:
                    self._speak_phrase("Mike has nothing to tell you.")
                except Exception:
                    logger.exception("Failed to synthesize empty-reply message.")
                continue

            logger.info("AI reply: %s", reply)
            if self._ui is not None:
                self._ui.set_last_reply(reply.strip())

            if self._ui is not None:
                self._ui.set_status("Transmitting")
            try:
                audio, sr = self._tts.synthesize_to_array(reply)
                logger.info("Synthesized TTS audio of %d samples at %d Hz", len(audio), sr)
                audio, sr = self._resample_to_config_rate(audio, sr)

                logger.info("Playing TTS to radio output")
                play_audio(audio, sr, self._output_device_index)
            except Exception:
                logger.exception("TTS synthesis or playback failed.")
                if self._ui is not None:
                    self._ui.set_status("Error")
                try:
                    # Best effort: speak via TTS about TTS failure is unlikely to help if TTS is broken,
                    # but we keep this for cases where playback failed after synthesis.
                    self._speak_phrase("Mike had a problem speaking the reply.")
                except Exception:
                    logger.exception("Failed to synthesize TTS error message.")

            if self._ui is not None:
                self._ui.set_status("Stand By")

