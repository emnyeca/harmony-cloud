"""Contrast Priority context selector のテスト。"""

from __future__ import annotations

import pytest

from changes.emiuet_contrast_context import contrast_context
from changes.note import pitch_class_to_semitone as pc


def _pcs(names):
    return {pc(n) for n in names}


@pytest.mark.parametrize(
    "symbol,scale_name,scale_root",
    [
        ("Dm7", "Half-Whole Diminished", "D"),
        ("G7", "Half-Whole Diminished", "G"),
        ("A7", "Half-Whole Diminished", "A"),
        ("Cmaj7", "Lydian", "C"),
        ("G7#5", "Whole Tone", "G"),
        ("Cmaj7#5", "Lydian Augmented", "C"),
        ("Dm6", "Melodic Minor", "D"),
        ("CmMaj7", "Harmonic Minor", "C"),
        ("Dm7b5", "Locrian natural 2", "D"),
        ("Cdim7", "Whole-Half Diminished", "C"),
    ],
)
def test_contrast_scale_by_quality(symbol, scale_name, scale_root):
    ctx = contrast_context(symbol)
    assert ctx is not None
    assert ctx["scale_name"] == scale_name
    assert ctx["scale_root"] == scale_root
    assert ctx["selection_policy"] == "contrast_priority"


def test_g7_half_whole_pitch_classes():
    ctx = contrast_context("G7")
    # G Ab Bb B Db D E F (enharmonic sharp 表記でも pc は同じ)
    assert _pcs(ctx["lpc"]) == {7, 8, 10, 11, 1, 2, 4, 5}


def test_cmaj7_lydian_pitch_classes():
    ctx = contrast_context("Cmaj7")
    assert _pcs(ctx["lpc"]) == _pcs(["C", "D", "E", "F#", "G", "A", "B"])


def test_contrast_lpc_always_includes_resolver_core():
    for symbol in (
        "Dm7", "G7", "A7", "Cmaj7", "G7#5", "Cmaj7#5", "Dm6", "CmMaj7", "Dm7b5", "Cdim7",
        "Dm11", "G9", "G13",
    ):
        ctx = contrast_context(symbol)
        if ctx is None:
            continue
        assert _pcs(ctx["resolver_core"]).issubset(_pcs(ctx["lpc"]))


def test_no_contrast_for_sus_and_alt():
    assert contrast_context("G7sus4") is None
    assert contrast_context("Galt") is None


def test_contrast_is_not_simple_reverse():
    # Cmaj7 の contrast は dominant_blues / diminished ではなく Lydian。
    assert contrast_context("Cmaj7")["scale_name"] == "Lydian"
    # maj 系は diminished / whole_tone を使わない。
    assert "Diminished" not in contrast_context("Cmaj7")["scale_name"]
    assert contrast_context("Cmaj7")["scale_name"] != "Whole Tone"
