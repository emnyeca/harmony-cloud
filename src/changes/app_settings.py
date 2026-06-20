"""Application settings model and persistence for EUB Changes."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

LIBRARY_PATH = Path.home() / "EUBChanges" / "library"
# Settings are stored in a fixed user-level location independent of library_path,
# so changing the library folder does not orphan the settings file.
SETTINGS_PATH = Path.home() / ".eub_changes_settings.json"


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value is not None else default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


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

    # Safety
    confirm_before_hardware_write: bool = True
    pattern_change_policy: str = "auto_song_mode"  # auto_song_mode | off

    # Display
    note_accidental: str = "flat"  # "flat" | "sharp"
    song_display_mode: str = "chord_cells"  # "chord_cells" | "cloud_graph"

    # Local AI generation beta. Defaults can be overridden by environment and
    # then persisted through the normal settings file.
    ai_generation_enabled: bool = field(
        default_factory=lambda: _env_bool("EUB_CHANGES_AI_GENERATION_ENABLED", True)
    )
    ollama_endpoint: str = field(
        default_factory=lambda: _env_str("EUB_CHANGES_OLLAMA_ENDPOINT", "http://localhost:11434")
    )
    ollama_model_name: str = field(
        default_factory=lambda: _env_str("EUB_CHANGES_OLLAMA_MODEL", "llama3.1")
    )
    ai_generation_max_validation_retries: int = field(
        default_factory=lambda: _env_int("EUB_CHANGES_AI_MAX_VALIDATION_RETRIES", 1)
    )
    ai_eval_ui_enabled: bool = field(
        default_factory=lambda: _env_bool("EUB_CHANGES_AI_EVAL_UI", False)
    )
    ai_eval_log_path: str = field(
        default_factory=lambda: _env_str(
            "EUB_CHANGES_AI_EVAL_LOG_PATH",
            str(Path.home() / "EUBChanges" / "ai-evaluation.jsonl"),
        )
    )
    # Failures that never reach the editor (bad JSON, out-of-vocabulary chords,
    # pipeline rejections, Ollama errors) are logged here for later improvement.
    ai_failure_log_path: str = field(
        default_factory=lambda: _env_str(
            "EUB_CHANGES_AI_FAILURE_LOG_PATH",
            str(Path.home() / "EUBChanges" / "ai-generation-failures.jsonl"),
        )
    )
    # Evaluation loop: Prompt Library -> Batch Generate -> Candidate Store ->
    # Critic (external) -> Human review. Each is a JSONL store joined by id.
    ai_prompt_library_path: str = field(
        default_factory=lambda: _env_str(
            "EUB_CHANGES_AI_PROMPT_LIBRARY_PATH",
            str(Path.home() / "EUBChanges" / "ai-prompts.jsonl"),
        )
    )
    ai_candidate_store_path: str = field(
        default_factory=lambda: _env_str(
            "EUB_CHANGES_AI_CANDIDATE_STORE_PATH",
            str(Path.home() / "EUBChanges" / "ai-candidates.jsonl"),
        )
    )
    ai_critic_log_path: str = field(
        default_factory=lambda: _env_str(
            "EUB_CHANGES_AI_CRITIC_LOG_PATH",
            str(Path.home() / "EUBChanges" / "ai-critic.jsonl"),
        )
    )
    ai_human_eval_log_path: str = field(
        default_factory=lambda: _env_str(
            "EUB_CHANGES_AI_HUMAN_EVAL_LOG_PATH",
            str(Path.home() / "EUBChanges" / "human-evaluation.jsonl"),
        )
    )
    ai_batch_candidates_per_prompt: int = field(
        default_factory=lambda: _env_int("EUB_CHANGES_AI_BATCH_CANDIDATES_PER_PROMPT", 5)
    )
    ai_human_review_min_critic_score: int = field(
        default_factory=lambda: _env_int("EUB_CHANGES_AI_HUMAN_REVIEW_MIN_CRITIC_SCORE", 3)
    )


def _migrate_raw(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert legacy field names before merging with defaults."""
    if "cloud_track_base" in raw and "cloud_tracks" not in raw:
        base = int(raw["cloud_track_base"])
        raw["cloud_tracks"] = [base + i for i in range(6)]
    raw.pop("cloud_track_base", None)
    if raw.get("song_display_mode") not in (None, "chord_cells", "cloud_graph"):
        raw["song_display_mode"] = "chord_cells"
    if "ai_enabled" in raw and "ai_generation_enabled" not in raw:
        raw["ai_generation_enabled"] = raw["ai_enabled"]
    raw.pop("ai_enabled", None)
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
