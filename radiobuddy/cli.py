from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from typing import Any, Optional


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="radiobuddy",
        description="RadioBuddy: VOX-driven VHF radio ↔ voice AI pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser(
        "run",
        help="Run the VOX loop (record → STT → LLM → TTS → play).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_run.add_argument(
        "--no-ui",
        action="store_true",
        help="Disable Rich live UI (prints logs only).",
    )
    p_run.add_argument(
        "--interactive-devices",
        action="store_true",
        help="Prompt to select audio devices on startup.",
    )
    p_run.add_argument(
        "--interactive-settings",
        action="store_true",
        help="Prompt to adjust VOX settings on startup.",
    )
    p_run.add_argument(
        "--input-device",
        default=None,
        help="Override AUDIO_INPUT_DEVICE (index or name substring).",
    )
    p_run.add_argument(
        "--output-device",
        default=None,
        help="Override AUDIO_OUTPUT_DEVICE (index or name substring).",
    )
    p_run.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help="Override SAMPLE_RATE (Hz).",
    )
    p_run.add_argument(
        "--chunk-seconds",
        type=float,
        default=None,
        help="Override CHUNK_SECONDS (VOX chunk size, seconds).",
    )
    p_run.add_argument(
        "--vox-threshold-db",
        type=float,
        default=None,
        help="Override VOX_THRESHOLD_DB.",
    )
    p_run.add_argument(
        "--vox-min-duration-ms",
        type=int,
        default=None,
        help="Override VOX_MIN_DURATION_MS.",
    )
    p_run.add_argument(
        "--vox-silence-timeout-sec",
        type=float,
        default=None,
        help="Override VOX_SILENCE_TIMEOUT_SEC.",
    )

    p_devices = sub.add_parser(
        "list-devices",
        help="List audio devices (input/output) for configuring .env.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_devices.add_argument(
        "--raw",
        action="store_true",
        help="Print only device list (no extra text).",
    )

    p_dtmf = sub.add_parser(
        "dtmf-test",
        help="Decode DTMF digits from a WAV/AIFF/FLAC file (offline).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_dtmf.add_argument(
        "--wav",
        required=True,
        help="Path to an audio file containing DTMF tones.",
    )
    p_dtmf.add_argument(
        "--stream-ms",
        type=int,
        default=50,
        help="Chunk size for streaming decode (ms).",
    )

    p_gen = sub.add_parser(
        "dtmf-generate",
        help="Generate a clean DTMF WAV (for testing your audio path).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_gen.add_argument("--digits", default="0909", help="Digit string to generate.")
    p_gen.add_argument("--out", required=True, help="Output WAV path.")
    p_gen.add_argument("--sr", type=int, default=8000, help="Sample rate for output WAV.")
    p_gen.add_argument("--tone-ms", type=int, default=140, help="Tone duration per digit (ms).")
    p_gen.add_argument("--gap-ms", type=int, default=70, help="Silence gap between digits (ms).")

    return parser


def _apply_overrides(cfg: Any, args: argparse.Namespace) -> Any:
    audio = cfg.audio
    vox = cfg.vox

    if getattr(args, "input_device", None) is not None:
        audio = replace(audio, input_device=args.input_device)
    if getattr(args, "output_device", None) is not None:
        audio = replace(audio, output_device=args.output_device)
    if getattr(args, "sample_rate", None) is not None:
        audio = replace(audio, sample_rate=args.sample_rate)
    if getattr(args, "chunk_seconds", None) is not None:
        audio = replace(audio, chunk_seconds=args.chunk_seconds)

    if getattr(args, "vox_threshold_db", None) is not None:
        vox = replace(vox, threshold_db=args.vox_threshold_db)
    if getattr(args, "vox_min_duration_ms", None) is not None:
        vox = replace(vox, min_duration_ms=args.vox_min_duration_ms)
    if getattr(args, "vox_silence_timeout_sec", None) is not None:
        vox = replace(vox, silence_timeout_sec=args.vox_silence_timeout_sec)

    return replace(cfg, audio=audio, vox=vox)


def _maybe_choose_devices(cfg: Any, ui: Any) -> Any:
    from .audio_io import get_device_display_name, interactive_select_device

    input_name = get_device_display_name(cfg.audio.input_device, "input")
    output_name = get_device_display_name(cfg.audio.output_device, "output")

    if ui is not None:
        ui.show_devices_panel(input_name, output_name)
        if not ui.ask_change_devices():
            return cfg
        ui.console.print("\n[bold]Choose input device[/]")
        chosen_in = interactive_select_device("input")
        ui.console.print("\n[bold]Choose output device[/]")
        chosen_out = interactive_select_device("output")
    else:
        print(f"Current input device:  {input_name}")
        print(f"Current output device: {output_name}")
        choice = input("Change audio devices? [y/N]: ").strip().lower()
        if choice not in ("y", "yes"):
            return cfg
        print("\nChoose input device")
        chosen_in = interactive_select_device("input")
        print("\nChoose output device")
        chosen_out = interactive_select_device("output")

    audio = cfg.audio
    if chosen_in is not None:
        audio = replace(audio, input_device=chosen_in)
    if chosen_out is not None:
        audio = replace(audio, output_device=chosen_out)
    return replace(cfg, audio=audio)


def _maybe_adjust_vox_settings(cfg: Any, ui: Any) -> Any:
    if ui is None:
        return cfg

    from .config import VoxConfig

    ui.show_vox_panel(
        threshold_db=cfg.vox.threshold_db,
        min_duration_ms=cfg.vox.min_duration_ms,
        silence_timeout_sec=cfg.vox.silence_timeout_sec,
    )

    from rich.prompt import Confirm

    if not Confirm.ask(
        "[dim]Change VOX settings (threshold/duration/silence)?[/]",
        default=False,
        console=ui.console,
    ):
        return cfg

    new_threshold, new_min_duration, new_silence = ui.edit_vox_settings(
        threshold_db=cfg.vox.threshold_db,
        min_duration_ms=cfg.vox.min_duration_ms,
        silence_timeout_sec=cfg.vox.silence_timeout_sec,
    )

    vox = VoxConfig(
        threshold_db=new_threshold,
        min_duration_ms=new_min_duration,
        silence_timeout_sec=new_silence,
    )
    return replace(cfg, vox=vox)


def _startup_menu(cfg: Any, ui: Any) -> Any:
    """
    Show a simple start menu:
      1) Start with current settings
      2) Configure and start (audio devices + VOX)
      3) Select mode and start
    """
    if ui is None:
        return cfg

    from rich.prompt import Prompt

    ui.console.print(
        "\n[bold]Startup options[/]\n"
        "  [cyan]1[/]: Start with current settings\n"
        "  [cyan]2[/]: Configure audio / VOX, then start\n"
        "  [cyan]3[/]: Select mode, then start\n"
    )

    while True:
        choice = Prompt.ask(
            "Select option",
            choices=["1", "2", "3"],
            default="1",
            console=ui.console,
        )
        if choice == "1":
            return cfg
        if choice == "2":
            cfg = _maybe_choose_devices(cfg, ui)
            cfg = _maybe_adjust_vox_settings(cfg, ui)
            cfg = _maybe_choose_mode(cfg, ui)
            return cfg
        if choice == "3":
            cfg = _maybe_choose_mode(cfg, ui)
            return cfg


def _maybe_choose_mode(cfg: Any, ui: Any) -> Any:
    if ui is None:
        return cfg

    from rich.prompt import Prompt

    current_mode = getattr(cfg, "mode", "ai")
    ui.console.print(
        "\n[bold]Mode settings[/]\n"
        "  [cyan]1[/]: ai       - full STT + AI + TTS\n"
        "  [cyan]2[/]: dummy    - offline dummy STT + dummy AI reply\n"
        "  [cyan]3[/]: repeater - no STT/AI/TTS, just record and repeat\n"
    )
    ui.console.print(f"Current mode: [cyan]{current_mode}[/]")

    default_choice = {
        "ai": "1",
        "dummy": "2",
        "repeater": "3",
    }.get(current_mode, "1")

    choice = Prompt.ask(
        "Select mode",
        choices=["1", "2", "3"],
        default=default_choice,
        console=ui.console,
    )

    mode_by_choice = {"1": "ai", "2": "dummy", "3": "repeater"}
    mode = mode_by_choice.get(choice, "ai")
    return replace(cfg, mode=mode)


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.command == "list-devices":
        from .audio_io import list_devices

        if not args.raw:
            print("Available audio devices:")
        list_devices()
        return

    if args.command == "dtmf-test":
        import soundfile as sf

        from .dtmf import DtmfDecoder, DtmfDecoderConfig

        data, sr = sf.read(str(args.wav), dtype="float32")
        if data.ndim == 2:
            data = data.mean(axis=1)
        data = data.astype("float32")

        dec = DtmfDecoder(DtmfDecoderConfig(sample_rate=int(sr)))
        chunk = max(64, int(int(sr) * (float(args.stream_ms) / 1000.0)))
        out: list[str] = []
        for i in range(0, data.size, chunk):
            out.extend(dec.process(data[i : i + chunk]))
        print("".join(out))
        return

    if args.command == "dtmf-generate":
        import soundfile as sf

        from .dtmf import synthesize_dtmf

        wav = synthesize_dtmf(
            digits=str(args.digits),
            sample_rate=int(args.sr),
            tone_ms=int(args.tone_ms),
            gap_ms=int(args.gap_ms),
        )
        sf.write(str(args.out), wav, int(args.sr))
        print(str(args.out))
        return

    if args.command != "run":
        parser.error(f"Unknown command: {args.command}")

    try:
        from .config import load_config

        cfg = load_config()
    except Exception as exc:  # noqa: BLE001
        logging.error("Failed to load configuration: %s", exc)
        raise SystemExit(1) from exc

    cfg = _apply_overrides(cfg, args)

    ui: Any
    if args.no_ui:
        ui = None
    else:
        from .console_ui import RadioBuddyUI

        ui = RadioBuddyUI(threshold_db=cfg.vox.threshold_db)
        ui.print_banner()

    # Show startup menu when UI is enabled. For non-UI runs, the old
    # behavior with CLI overrides only is preserved.
    if ui is not None:
        cfg = _startup_menu(cfg, ui)

    from .llm import build_llm_client
    from .pipeline import RadioBuddyPipeline
    from .stt import build_stt_client
    from .tts import build_tts_client

    stt_client = build_stt_client(cfg.stt, cfg.mode)
    llm_client = build_llm_client(cfg.llm, cfg.mode)
    tts_client = build_tts_client(cfg.tts)

    pipeline = RadioBuddyPipeline(cfg, stt_client, llm_client, tts_client, ui=ui)
    try:
        pipeline.run_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        raise SystemExit(0) from None

