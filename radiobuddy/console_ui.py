"""
Rich-based console UI for RadioBuddy: level meter, status, and styled panels.
"""
from __future__ import annotations

import queue
import threading
from typing import Callable, Optional

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# Theme: radio / terminal aesthetic
RADIO_THEME = Theme(
    {
        "info": "dim cyan",
        "success": "green",
        "warning": "yellow",
        "error": "bold red",
        "level": "bright_blue",
        "status": "bold",
        "banner": "bold cyan",
    }
)

# dB range for the level meter (typical speech)
LEVEL_MIN_DB = -50.0
LEVEL_MAX_DB = -10.0


def _level_to_fraction(level_db: float) -> float:
    """Map level_db to 0..1 for progress bar."""
    if level_db <= LEVEL_MIN_DB:
        return 0.0
    if level_db >= LEVEL_MAX_DB:
        return 1.0
    return (level_db - LEVEL_MIN_DB) / (LEVEL_MAX_DB - LEVEL_MIN_DB)


def _level_color(level_db: float, threshold_db: float) -> str:
    if level_db >= threshold_db:
        return "green"
    if level_db >= LEVEL_MIN_DB + (LEVEL_MAX_DB - LEVEL_MIN_DB) * 0.5:
        return "yellow"
    return "blue"


class RadioBuddyUI:
    """
    Console UI: banner, device panel, live level meter, status, and message panels.
    """

    def __init__(self, threshold_db: float) -> None:
        self._console = Console(theme=RADIO_THEME)
        self._threshold_db = threshold_db
        self._level_db: float = LEVEL_MIN_DB
        self._status = "Stand By"
        self._last_heard = "—"
        self._last_reply = "—"
        self._last_dtmf = "—"
        self._last_dtmf_event = "—"
        self._speech_detected = False

    def _build_level_meter(self) -> Panel:
        frac = _level_to_fraction(self._level_db)
        color = _level_color(self._level_db, self._threshold_db)
        bar_width = 40
        filled = int(bar_width * frac)
        bar = "█" * filled + "░" * (bar_width - filled)
        threshold_frac = _level_to_fraction(self._threshold_db)
        threshold_pos = int(bar_width * threshold_frac)
        threshold_marker = "│" if 0 <= threshold_pos <= bar_width else ""
        line = Text()
        line.append("  ")
        line.append(bar[:threshold_pos], style=f"bold {color}")
        line.append(threshold_marker, style="red")
        line.append(bar[threshold_pos:], style=color)
        line.append(f"  {self._level_db:+.1f} dB", style="level")
        if self._speech_detected:
            line.append("  ● REC", style="bold green")
        return Panel(
            line,
            title="[bold]Input level[/]",
            border_style="blue",
            padding=(0, 1),
        )

    def _build_status_panel(self) -> Panel:
        return Panel(
            f"[status]{self._status}[/]",
            title="Status",
            border_style="cyan",
            padding=(0, 1),
        )

    def _build_messages_panel(self) -> Panel:
        table = Table.grid(padding=(0, 2))
        table.add_row(
            Text("Heard:", style="bold cyan"),
            Text(self._last_heard, style="white"),
        )
        table.add_row(
            Text("Reply:", style="bold green"),
            Text(self._last_reply, style="white"),
        )
        table.add_row(
            Text("DTMF:", style="bold magenta"),
            Text(self._last_dtmf, style="white"),
        )
        table.add_row(
            Text("DTMF evt:", style="bold magenta"),
            Text(self._last_dtmf_event, style="white"),
        )
        return Panel(
            table,
            title="Last exchange",
            border_style="dim blue",
            padding=(0, 1),
        )

    def _build_live_layout(self) -> Panel:
        """Single panel containing level + status + messages for Live display."""
        from rich.columns import Columns

        level = self._build_level_meter()
        status = self._build_status_panel()
        msgs = self._build_messages_panel()
        top = Columns([level, status], equal=False, expand=True)
        return Panel(
            Group(top, msgs),
            title="",
            border_style="dim",
            padding=(0, 1),
        )

    def print_banner(self) -> None:
        banner = Text()
        banner.append("  ◉ ", style="banner")
        banner.append("R A D I O B U D D Y", style="banner")
        banner.append("  ◉\n", style="banner")
        banner.append("  VHF ↔ AI voice pipeline\n", style="dim")
        self._console.print(Panel(banner, border_style="cyan", padding=(0, 2)))
        self._console.print()

    def show_devices_panel(
        self,
        input_name: str,
        output_name: str,
    ) -> None:
        table = Table.grid(padding=(0, 2))
        table.add_row(Text("Input:", style="bold"), Text(input_name, style="cyan"))
        table.add_row(Text("Output:", style="bold"), Text(output_name, style="cyan"))
        self._console.print(
            Panel(
                table,
                title="[bold]Audio devices[/]",
                border_style="blue",
                padding=(0, 1),
            )
        )

    def ask_change_devices(self) -> bool:
        from rich.prompt import Confirm
        return Confirm.ask(
            "[dim]Change audio devices?[/]",
            default=False,
            console=self._console,
        )

    def show_vox_panel(
        self,
        threshold_db: float,
        min_duration_ms: int,
        silence_timeout_sec: float,
    ) -> None:
        table = Table.grid(padding=(0, 2))
        table.add_row(Text("VOX threshold (dB):", style="bold"), Text(f"{threshold_db}", style="cyan"))
        table.add_row(Text("Min duration (ms):", style="bold"), Text(f"{min_duration_ms}", style="cyan"))
        table.add_row(
            Text("Silence timeout (sec):", style="bold"),
            Text(f"{silence_timeout_sec}", style="cyan"),
        )
        self._console.print(
            Panel(
                table,
                title="[bold]VOX settings[/]",
                border_style="blue",
                padding=(0, 1),
            )
        )

    def edit_vox_settings(
        self,
        threshold_db: float,
        min_duration_ms: int,
        silence_timeout_sec: float,
    ) -> tuple[float, int, float]:
        from rich.prompt import Prompt

        self._console.print("\n[dim]Press Enter to keep current value.[/]\n")

        thresh_str = Prompt.ask(
            "VOX threshold dB",
            default=f"{threshold_db}",
            console=self._console,
        )
        dur_str = Prompt.ask(
            "Min duration ms",
            default=f"{min_duration_ms}",
            console=self._console,
        )
        silence_str = Prompt.ask(
            "Silence timeout sec",
            default=f"{silence_timeout_sec}",
            console=self._console,
        )

        try:
            new_threshold = float(thresh_str)
        except ValueError:
            new_threshold = threshold_db

        try:
            new_min_duration = int(dur_str)
        except ValueError:
            new_min_duration = min_duration_ms

        try:
            new_silence = float(silence_str)
        except ValueError:
            new_silence = silence_timeout_sec

        return new_threshold, new_min_duration, new_silence

    def set_status(self, status: str) -> None:
        self._status = status

    def set_last_heard(self, text: str) -> None:
        self._last_heard = text or "—"

    def set_last_reply(self, text: str) -> None:
        self._last_reply = text or "—"

    def set_last_dtmf(self, digits: str) -> None:
        self._last_dtmf = digits or "—"

    def set_last_dtmf_event(self, text: str) -> None:
        self._last_dtmf_event = text or "—"

    def set_speech_detected(self, detected: bool) -> None:
        self._speech_detected = detected

    def run_listening_until_segment(
        self,
        record_fn: Callable[
            [Callable[[float], None], Callable[[], None], threading.Event], Optional[object]
        ],
    ) -> Optional[object]:
        """
        Run record_fn in a background thread with an on_level callback.
        Main thread runs a Live display showing input level until record_fn returns.
        Returns the segment (or None) from record_fn.
        """
        result_queue: queue.Queue = queue.Queue()
        self._speech_detected = False
        stop_event = threading.Event()

        def on_level(level_db: float) -> None:
            self._level_db = level_db

        def on_started() -> None:
            # Recording has officially started (after min_duration_ms above threshold).
            self._speech_detected = True
            self.set_status("Recording")

        def run_record() -> None:
            try:
                seg = record_fn(on_level, on_started, stop_event)
                result_queue.put(seg)
            except Exception as e:
                result_queue.put(e)

        self._status = "Stand By"
        thread = threading.Thread(target=run_record, daemon=False)
        thread.start()

        try:
            with Live(
                self._build_live_layout(),
                console=self._console,
                refresh_per_second=20,
                transient=False,
            ) as live:
                while thread.is_alive():
                    live.update(self._build_live_layout())
                    thread.join(timeout=0.05)
        except KeyboardInterrupt:
            stop_event.set()
            thread.join(timeout=2.0)
            raise

        # One final update with last level
        self._level_db = LEVEL_MIN_DB
        self._speech_detected = False
        self._status = "Stand By"
        result = result_queue.get_nowait()
        if isinstance(result, Exception):
            raise result
        return result

    def update_level(self, level_db: float) -> None:
        """For use from callback; updates current level (thread-safe for float)."""
        self._level_db = level_db

    def print_message(self, message: str, style: str = "info") -> None:
        self._console.print(f"  [{style}]{message}[/]")

    @property
    def console(self) -> Console:
        return self._console
