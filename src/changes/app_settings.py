"""Application settings model and persistence for EUB Changes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

LIBRARY_PATH = Path.home() / "EUBChanges" / "library"
# Settings are stored in a fixed user-level location independent of library_path,
# so changing the library folder does not orphan the settings file.
SETTINGS_PATH = Path.home() / ".eub_changes_settings.json"


@dataclass
class AppSettings:
    library_path: str = field(default_factory=lambda: str(LIBRARY_PATH))

    # Cloud — one track assignment per voice (None = don't send)
    cloud_trigger_policy: str = "hold_until_change"  # hold_until_change | retrigger
    cloud_center_midi: int = 60   # C4
    cloud_spread_min: int = 14
    cloud_spread_max: int = 16
    cloud_average_tolerance: int = 2
    cloud_tracks: list[int | None] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])

    # Bass — single track assignment (None = don't send)
    bass_trigger_policy: str = "hold_until_change"
    bass_center_midi: int = 36    # C2
    bass_track: int | None = 7

    # Chord — single track for all 6 chord notes (None = don't send)
    chord_trigger_policy: str = "retrigger"
    chord_center_midi: int = 60   # C4
    chord_track: int | None = 8

    # LPC Lead Layer (v0.3.0 beta) — Tracks 9–16
    lpc_lead_layer_enabled: bool = False
    lpc_lead_layer_range_root: int = 0   # pitch class 0=C … 11=B
    lpc_lead_layer_octave: int = 4        # e.g. 4 → range C4–B4

    # Safety
    confirm_before_hardware_write: bool = True
    pattern_change_policy: str = "auto_song_mode"  # auto_song_mode | off

    # Display
    note_accidental: str = "flat"  # "flat" | "sharp"
    song_display_mode: str = "chord_cells"  # "chord_cells" | "cloud_graph"


def _migrate_raw(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert legacy field names before merging with defaults."""
    if "cloud_track_base" in raw and "cloud_tracks" not in raw:
        base = int(raw["cloud_track_base"])
        raw["cloud_tracks"] = [base + i for i in range(6)]
    raw.pop("cloud_track_base", None)
    if raw.get("song_display_mode") not in (None, "chord_cells", "cloud_graph"):
        raw["song_display_mode"] = "chord_cells"
    return raw


def load_settings() -> AppSettings:
    if SETTINGS_PATH.exists():
        try:
            raw: dict[str, Any] = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            raw = _migrate_raw(raw)
            defaults = asdict(AppSettings())
            merged = {**defaults, **{k: v for k, v in raw.items() if k in defaults}}
            return AppSettings(**merged)
        except Exception:
            pass
    return AppSettings()


def save_settings(settings: AppSettings) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(
        json.dumps(asdict(settings), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
