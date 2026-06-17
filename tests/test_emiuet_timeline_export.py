"""Emiuet Session compiled timeline exporter のテスト。"""

from __future__ import annotations

from changes.exporters.emiuet_timeline import (
    CompiledStepInput,
    build_emiuet_compiled_timeline,
)
from changes.note import pitch_class_to_semitone as pc


def _pcs(names):
    return {pc(n) for n in names}


def _sample_steps():
    # Digitone compile 済み step（tick 境界は確定済み）を入力に取る。
    chords = ["Dm7", "G7", "Cmaj7", "A7"]
    return [
        CompiledStepInput(id=f"step_{i:03d}", start_tick=i * 24, end_tick=(i + 1) * 24, chord=c, source_step_index=i)
        for i, c in enumerate(chords)
    ]


def test_export_top_level_schema():
    tl = build_emiuet_compiled_timeline(_sample_steps(), original_tempo=120.0, digitone_tempo=60.0)
    assert tl["schema"] == "emnyeca.emiuet_session.compiled_timeline"
    assert tl["schema_version"] == 2
    assert tl["timeline_basis"] == "digitone_step"
    assert tl["clock"]["ppqn"] == 24
    assert tl["form"]["end_tick"] == 96
    assert len(tl["steps"]) == 4


def test_each_step_has_progression_and_contrast():
    tl = build_emiuet_compiled_timeline(_sample_steps())
    for step in tl["steps"]:
        assert "progression" in step["contexts"]
        assert "contrast" in step["contexts"]  # all four have contrast
        assert step["default_context_role"] == "progression"
        assert step["mod_context_role"] == "contrast"


def test_tick_boundaries_preserved():
    tl = build_emiuet_compiled_timeline(_sample_steps())
    assert [(s["start_tick"], s["end_tick"]) for s in tl["steps"]] == [
        (0, 24), (24, 48), (48, 72), (72, 96)
    ]


def test_hard_context_and_resolver_core_separated_in_export():
    steps = [CompiledStepInput("s0", 0, 24, "Dm11")]
    tl = build_emiuet_compiled_timeline(steps)
    prog = tl["steps"][0]["contexts"]["progression"]
    assert _pcs(prog["hard_context"]) == _pcs(["D", "F", "A", "C", "E", "G"])
    assert prog["resolver_core"] == ["D", "F", "A", "C"]


def test_contrast_scales_in_export():
    tl = build_emiuet_compiled_timeline(_sample_steps())
    by_chord = {s["chord"]: s for s in tl["steps"]}
    assert by_chord["Dm7"]["contexts"]["contrast"]["scale_name"] == "Half-Whole Diminished"
    assert by_chord["G7"]["contexts"]["contrast"]["scale_name"] == "Half-Whole Diminished"
    assert by_chord["Cmaj7"]["contexts"]["contrast"]["scale_name"] == "Lydian"
    assert by_chord["A7"]["contexts"]["contrast"]["scale_name"] == "Half-Whole Diminished"


def test_resolver_core_subset_of_lpc_in_all_contexts():
    tl = build_emiuet_compiled_timeline(_sample_steps())
    for step in tl["steps"]:
        for ctx in step["contexts"].values():
            assert _pcs(ctx["resolver_core"]).issubset(_pcs(ctx["lpc"]))


def test_sus_step_has_no_contrast():
    tl = build_emiuet_compiled_timeline([CompiledStepInput("s0", 0, 24, "G7sus4")])
    assert "contrast" not in tl["steps"][0]["contexts"]
    assert "progression" in tl["steps"][0]["contexts"]
