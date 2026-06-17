"""compile pipeline から Emiuet Session timeline を export する導線のテスト。"""

from __future__ import annotations

import json
from pathlib import Path

from changes.pipeline_digitone import (
    compile_emiuet_session_timeline,
    save_emiuet_session_timeline,
)
from changes.note import pitch_class_to_semitone as pc


def _payload():
    return {
        "name": "ContrastDemo",
        "tempo": 120,
        "time_signature": "4/4",
        "sections": [{"name": "A", "progression": [["Dm7", "G7", "Cmaj7", "A7"]]}],
    }


def test_pipeline_exports_schema_v2_timeline():
    tl = compile_emiuet_session_timeline(_payload())
    assert tl["schema"] == "emnyeca.emiuet_session.compiled_timeline"
    assert tl["schema_version"] == 2
    assert tl["timeline_basis"] == "digitone_step"
    assert tl["steps"], "expected non-empty steps"


def test_pipeline_steps_have_contexts_and_monotonic_ticks():
    tl = compile_emiuet_session_timeline(_payload())
    prev_end = None
    for step in tl["steps"]:
        assert step["start_tick"] < step["end_tick"]
        if prev_end is not None:
            assert step["start_tick"] >= prev_end  # monotonic, no overlap
        prev_end = step["end_tick"]
        assert "progression" in step["contexts"]
        for ctx in step["contexts"].values():
            core = {pc(n) for n in ctx["resolver_core"]}
            lpc = {pc(n) for n in ctx["lpc"]}
            assert core.issubset(lpc)


def test_pipeline_uses_device_tempo_for_tick_scale():
    # tick scale は device tempo 基準（原曲 tempo そのままではない）。
    tl = compile_emiuet_session_timeline(_payload())
    assert tl["clock"]["original_tempo"] == 120.0
    # Digitone device tempo は SPEED 等価変換で原曲と一致しない想定。
    assert tl["clock"]["digitone_tempo"] != tl["clock"]["original_tempo"]


def test_save_emiuet_session_timeline_writes_file(tmp_path: Path):
    tl = compile_emiuet_session_timeline(_payload())
    path = save_emiuet_session_timeline(tmp_path, tl)
    assert path.exists()
    reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert reloaded["schema_version"] == 2
