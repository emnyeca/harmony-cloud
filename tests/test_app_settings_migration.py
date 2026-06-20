"""Tests for AppSettings loading and cloud_track_base migration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from changes.app_settings import AppSettings, _migrate_raw, load_settings, save_settings


# ── _migrate_raw ─────────────────────────────────────────────────────────────

def test_migrate_cloud_track_base_1() -> None:
    raw = {"cloud_track_base": 1}
    result = _migrate_raw(raw)
    assert result["cloud_tracks"] == [1, 2, 3, 4, 5, 6]
    assert "cloud_track_base" not in result


def test_migrate_cloud_track_base_3() -> None:
    raw = {"cloud_track_base": 3}
    result = _migrate_raw(raw)
    assert result["cloud_tracks"] == [3, 4, 5, 6, 7, 8]


def test_migrate_no_op_when_cloud_tracks_present() -> None:
    raw = {"cloud_tracks": [1, 2, None, None, 5, 6], "cloud_track_base": 1}
    result = _migrate_raw(raw)
    # cloud_tracks already present → not overwritten, cloud_track_base removed
    assert result["cloud_tracks"] == [1, 2, None, None, 5, 6]
    assert "cloud_track_base" not in result


def test_migrate_noop_when_neither_field() -> None:
    raw = {"library_path": "/tmp/lib"}
    result = _migrate_raw(raw)
    assert "cloud_tracks" not in result
    assert "cloud_track_base" not in result


# ── load_settings round-trip ──────────────────────────────────────────────────

def test_save_load_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings_path = tmp_path / "settings.json"
    monkeypatch.setattr("changes.app_settings.SETTINGS_PATH", settings_path)

    s = AppSettings(
        cloud_tracks=[1, 1, None, None, None, None],
        bass_track=None,
        chord_track=8,
    )
    save_settings(s)
    loaded = load_settings()
    assert loaded.cloud_tracks == [1, 1, None, None, None, None]
    assert loaded.bass_track is None
    assert loaded.chord_track == 8
    assert loaded.song_display_mode == "chord_cells"


def test_save_load_song_display_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings_path = tmp_path / "settings.json"
    monkeypatch.setattr("changes.app_settings.SETTINGS_PATH", settings_path)

    save_settings(AppSettings(song_display_mode="cloud_graph"))
    loaded = load_settings()

    assert loaded.song_display_mode == "cloud_graph"


def test_save_load_ai_generation_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings_path = tmp_path / "settings.json"
    monkeypatch.setattr("changes.app_settings.SETTINGS_PATH", settings_path)

    save_settings(
        AppSettings(
            ai_generation_enabled=False,
            ollama_endpoint="http://127.0.0.1:11434",
            ollama_model_name="qwen2.5",
            ai_generation_max_validation_retries=2,
            ai_eval_ui_enabled=True,
            ai_eval_log_path=str(tmp_path / "eval.jsonl"),
        )
    )
    loaded = load_settings()

    assert loaded.ai_generation_enabled is False
    assert loaded.ollama_endpoint == "http://127.0.0.1:11434"
    assert loaded.ollama_model_name == "qwen2.5"
    assert loaded.ai_generation_max_validation_retries == 2
    assert loaded.ai_eval_ui_enabled is True
    assert loaded.ai_eval_log_path == str(tmp_path / "eval.jsonl")


def test_load_legacy_cloud_track_base(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings_path = tmp_path / "settings.json"
    monkeypatch.setattr("changes.app_settings.SETTINGS_PATH", settings_path)
    settings_path.write_text(
        json.dumps({"cloud_track_base": 1, "bass_track": 7, "chord_track": 8}),
        encoding="utf-8",
    )
    loaded = load_settings()
    assert loaded.cloud_tracks == [1, 2, 3, 4, 5, 6]
    assert "cloud_track_base" not in vars(loaded)


def test_default_settings_have_correct_tracks() -> None:
    s = AppSettings()
    assert s.cloud_tracks == [1, 2, 3, 4, 5, 6]
    assert s.bass_track == 7
    assert s.chord_track == 8
    assert s.pattern_change_policy == "auto_song_mode"


def test_load_legacy_settings_gets_default_pattern_change_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings_path = tmp_path / "settings.json"
    monkeypatch.setattr("changes.app_settings.SETTINGS_PATH", settings_path)
    settings_path.write_text(json.dumps({"bass_track": 7, "chord_track": 8}), encoding="utf-8")

    loaded = load_settings()

    assert loaded.pattern_change_policy == "auto_song_mode"
    assert loaded.song_display_mode == "chord_cells"


def test_load_invalid_song_display_mode_falls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings_path = tmp_path / "settings.json"
    monkeypatch.setattr("changes.app_settings.SETTINGS_PATH", settings_path)
    settings_path.write_text(json.dumps({"song_display_mode": "bad"}), encoding="utf-8")

    loaded = load_settings()

    assert loaded.song_display_mode == "chord_cells"
