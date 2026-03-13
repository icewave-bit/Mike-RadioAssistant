from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from .audio_io import play_audio, record_segment_vox, resolve_device
from .config import AppConfig
from .llm import GptNanoClient
from .stt import WhisperSttClient
from .tts import MacTts, PiperTts

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

        logger.info(
            "Using devices input=%s output=%s",
            self._input_device_index if self._input_device_index is not None else "default",
            self._output_device_index if self._output_device_index is not None else "default",
        )

    def _record_segment(self):
        return record_segment_vox(
            input_device=self._input_device_index,
            sample_rate=self._cfg.audio.sample_rate,
            chunk_seconds=self._cfg.audio.chunk_seconds,
            threshold_db=self._cfg.vox.threshold_db,
            min_duration_ms=self._cfg.vox.min_duration_ms,
            silence_timeout_sec=self._cfg.vox.silence_timeout_sec,
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

        if sr != self._cfg.audio.sample_rate:
            factor = self._cfg.audio.sample_rate / sr
            indices = (np.arange(int(len(audio) * factor)) / factor).astype(int)
            indices = np.clip(indices, 0, len(audio) - 1)
            audio = audio[indices]
            sr = self._cfg.audio.sample_rate

        logger.info("Playing phrase TTS to radio output")
        play_audio(audio, sr, self._output_device_index)

    def run_forever(self) -> None:
        logger.info("Starting RadioBuddy VOX loop")
        while True:
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
                        stop_event=stop_event,
                    )
                )
            else:
                segment = self._record_segment()

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
            if any(
                phrase in normalized
                for phrase in (
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
                )
            ):
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

                if sr != self._cfg.audio.sample_rate:
                    factor = self._cfg.audio.sample_rate / sr
                    indices = (np.arange(int(len(audio) * factor)) / factor).astype(int)
                    indices = np.clip(indices, 0, len(audio) - 1)
                    audio = audio[indices]
                    sr = self._cfg.audio.sample_rate

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

