from __future__ import annotations

import pytest

from changes.chord_engine import CHORD_MANDATORY_INTERVALS, ChordConstructionError, construct_chord_pitch_classes
from changes.chord_parser import parse_chord_core


def _selected_collection(*pitch_classes: int) -> tuple[int, ...]:
    return tuple(pitch_classes)


def test_parse_chord_core_normalizes_plain_sus4_to_dominant_sus4():
    core = parse_chord_core("Csus4")

    assert core.quality == "sus4"
    assert core.normalized_quality == "7sus4"


def test_parse_chord_core_accepts_major_sharp_eleven():
    core = parse_chord_core("Emaj7#11")

    assert core.quality == "maj7#11"
    assert core.altered_degrees == frozenset({"#11"})


def test_parse_chord_core_normalizes_7alt_to_alt():
    core = parse_chord_core("B7alt")

    assert core.quality == "7alt"
    assert core.normalized_quality == "alt"
    assert core.seventh_type == "b7"
    assert core.special_semantic_tag == "alt"


@pytest.mark.parametrize("symbol", ["Eb6/9", "C6/9", "F6/9"])
def test_parse_chord_core_accepts_major_six_nine(symbol: str):
    core = parse_chord_core(symbol)

    assert core.quality == "6/9"
    assert core.normalized_quality == "6/9"
    assert core.base_quality == "major"
    assert core.extensions == frozenset({"6", "9"})
    assert core.added_degrees == frozenset({"6", "9"})
    assert core.slash_bass is None


@pytest.mark.parametrize("symbol", ["Cm6/9", "Am6/9"])
def test_parse_chord_core_accepts_minor_six_nine(symbol: str):
    core = parse_chord_core(symbol)

    assert core.quality == "m6/9"
    assert core.normalized_quality == "m6/9"
    assert core.base_quality == "minor"
    assert core.extensions == frozenset({"6", "9"})
    assert core.added_degrees == frozenset({"6", "9"})
    assert core.slash_bass is None


def test_parse_chord_core_keeps_slash_bass_after_six_nine():
    core = parse_chord_core("C6/9/E")

    assert core.quality == "6/9"
    assert core.slash_bass == "E"
    assert core.slash_bass_pc == 4


def test_parse_chord_core_keeps_plain_slash_chord():
    core = parse_chord_core("C/E")

    assert core.quality == ""
    assert core.slash_bass == "E"
    assert core.slash_bass_pc == 4


def test_parse_chord_core_accepts_parenthesized_dominant_tensions():
    core = parse_chord_core("G7(b9,#11)")

    assert core.root == "G"
    assert core.quality == "7(b9,#11)"
    assert core.seventh_type == "b7"
    assert core.extensions == frozenset({"7", "9", "11"})
    assert core.altered_degrees == frozenset({"b9", "#11"})


def test_parse_chord_core_accepts_parenthesized_major_sharp_eleven():
    core = parse_chord_core("Cmaj7(#11)")

    assert core.root == "C"
    assert core.quality == "maj7(#11)"
    assert core.seventh_type == "maj7"
    assert core.extensions == frozenset({"7", "11"})
    assert core.altered_degrees == frozenset({"#11"})


def test_construct_chord_cmaj7_uses_symbol_tones_plus_collection_tensions():
    core = parse_chord_core("Cmaj7")
    selected = _selected_collection(0, 2, 4, 5, 7, 9, 11)

    result = construct_chord_pitch_classes(core, selected)

    assert result.mandatory_intervals == (0, 4, 7, 11)
    assert result.mandatory_pitch_classes == (0, 4, 7, 11)
    assert result.automatic_tension_intervals == (2, 9)
    assert result.automatic_tension_pitch_classes == (2, 9)
    assert result.final_pitch_classes == (0, 4, 7, 11, 2, 9)


def test_construct_chord_c_six_nine_uses_major_triad_sixth_and_ninth():
    core = parse_chord_core("C6/9")
    selected = _selected_collection(0, 2, 4, 7, 9)

    result = construct_chord_pitch_classes(core, selected)

    assert result.mandatory_intervals == (0, 4, 7, 9, 2)
    assert result.mandatory_pitch_classes == (0, 4, 7, 9, 2)
    assert result.final_pitch_classes == (0, 4, 7, 9, 2)


def test_construct_chord_c_minor_six_nine_uses_minor_triad_sixth_and_ninth():
    core = parse_chord_core("Cm6/9")
    selected = _selected_collection(0, 2, 3, 7, 9)

    result = construct_chord_pitch_classes(core, selected)

    assert result.mandatory_intervals == (0, 3, 7, 9, 2)
    assert result.mandatory_pitch_classes == (0, 3, 7, 9, 2)
    assert result.final_pitch_classes == (0, 3, 7, 9, 2)


def test_construct_chord_c7b9_preserves_explicit_b9():
    core = parse_chord_core("C7b9")
    selected = _selected_collection(0, 1, 4, 5, 7, 8, 10)

    result = construct_chord_pitch_classes(core, selected)

    assert 1 in result.mandatory_intervals
    assert 1 in result.mandatory_pitch_classes
    assert len(result.final_pitch_classes) == 6
    assert len(set(result.final_pitch_classes)) == 6
    assert 8 in result.automatic_tension_pitch_classes


def test_construct_chord_c7b9_matches_requested_construction_order():
    core = parse_chord_core("C7b9")
    selected = _selected_collection(0, 1, 4, 5, 7, 8, 10)

    result = construct_chord_pitch_classes(core, selected)

    assert result.mandatory_pitch_classes == (0, 4, 7, 10, 1)
    assert result.automatic_tension_pitch_classes == (8,)
    assert result.final_pitch_classes == (0, 4, 7, 10, 1, 8)


def test_construct_chord_c7b9_does_not_auto_add_sharp9_when_available():
    core = parse_chord_core("C7b9")
    selected = _selected_collection(0, 1, 3, 4, 7, 8, 10)

    result = construct_chord_pitch_classes(core, selected)

    assert 1 in result.mandatory_pitch_classes
    assert 3 not in result.final_pitch_classes
    assert result.automatic_tension_pitch_classes == (8,)
    assert result.final_pitch_classes == (0, 4, 7, 10, 1, 8)


def test_construct_chord_c7b9_does_not_auto_add_natural9_when_written_b9():
    core = parse_chord_core("C7b9")
    selected = _selected_collection(0, 1, 2, 4, 7, 9, 10)

    result = construct_chord_pitch_classes(core, selected)

    assert 1 in result.final_pitch_classes
    assert 2 not in result.final_pitch_classes
    assert 9 in result.automatic_tension_pitch_classes


def test_construct_chord_csus4_adds_d_and_a_above_shell():
    core = parse_chord_core("Csus4")
    selected = _selected_collection(0, 2, 5, 7, 9, 10)

    result = construct_chord_pitch_classes(core, selected)

    assert core.normalized_quality == "7sus4"
    assert result.mandatory_intervals == (0, 5, 7, 10)
    assert result.mandatory_pitch_classes == (0, 5, 7, 10)
    assert result.automatic_tension_pitch_classes == (2, 9)
    assert result.final_pitch_classes == (0, 5, 7, 10, 2, 9)


def test_construct_chord_csus4_matches_requested_construction_order():
    core = parse_chord_core("Csus4")
    selected = _selected_collection(0, 2, 5, 7, 9, 10)

    result = construct_chord_pitch_classes(core, selected)

    assert core.normalized_quality == "7sus4"
    assert result.mandatory_pitch_classes == (0, 5, 7, 10)
    assert result.automatic_tension_pitch_classes == (2, 9)
    assert result.final_pitch_classes == (0, 5, 7, 10, 2, 9)


def test_construct_chord_e7sharp9_in_minor_context_preserves_explicit_sharp9():
    core = parse_chord_core("E7#9")
    selected = _selected_collection(0, 2, 4, 5, 6, 7, 8, 9, 11)

    result = construct_chord_pitch_classes(core, selected)

    assert result.mandatory_intervals == (0, 4, 7, 10, 3)
    assert result.mandatory_pitch_classes == (4, 8, 11, 2, 7)
    assert result.automatic_tension_pitch_classes == (0,)
    assert result.final_pitch_classes == (4, 8, 11, 2, 7, 0)


def test_construct_chord_e7sharp9_matches_requested_construction_order():
    core = parse_chord_core("E7#9")
    selected = _selected_collection(0, 2, 4, 5, 6, 7, 8, 9, 11)

    result = construct_chord_pitch_classes(core, selected)

    assert result.mandatory_pitch_classes == (4, 8, 11, 2, 7)
    assert result.automatic_tension_pitch_classes == (0,)
    assert result.final_pitch_classes == (4, 8, 11, 2, 7, 0)


def test_construct_chord_c7sharp9_does_not_auto_add_other_ninth_variants():
    core = parse_chord_core("C7#9")
    selected = _selected_collection(0, 1, 2, 3, 4, 7, 8, 10)

    result = construct_chord_pitch_classes(core, selected)

    assert 3 in result.mandatory_pitch_classes
    assert 1 not in result.automatic_tension_pitch_classes
    assert 2 not in result.automatic_tension_pitch_classes
    assert 8 in result.automatic_tension_pitch_classes


def test_construct_chord_gm7b5_uses_natural13_when_locrian_natural2_needs_six_notes():
    core = parse_chord_core("Gm7b5")
    selected = _selected_collection(1, 2, 4, 5, 7, 9, 10)

    result = construct_chord_pitch_classes(core, selected)

    assert set(result.mandatory_pitch_classes) == {7, 10, 1, 5}
    assert result.automatic_tension_intervals == (2, 9)
    assert result.automatic_tension_pitch_classes == (9, 4)
    assert set(result.final_pitch_classes) == {7, 10, 1, 5, 9, 4}
    assert len(result.final_pitch_classes) == 6


def test_construct_chord_alt_allows_compound_alterations():
    core = parse_chord_core("Calt")
    selected = _selected_collection(0, 1, 2, 3, 4, 6, 7, 8, 10)

    result = construct_chord_pitch_classes(core, selected)

    assert result.automatic_excluded_intervals == ()
    assert set(result.automatic_tension_pitch_classes) >= {1, 3, 8}
    assert len(result.final_pitch_classes) == 6
    assert len(set(result.final_pitch_classes)) == 6


@pytest.mark.parametrize(
    ("symbol", "expected_pitch_class"),
    [
        ("C7b9", 1),
        ("C7#9", 3),
        ("C7#11", 6),
        ("C7b13", 8),
    ],
)
def test_construct_chord_preserves_explicit_altered_tones(symbol: str, expected_pitch_class: int):
    core = parse_chord_core(symbol)
    selected = tuple(range(12))

    result = construct_chord_pitch_classes(core, selected)

    assert expected_pitch_class in result.mandatory_pitch_classes
    assert len(result.final_pitch_classes) == 6
    assert len(set(result.final_pitch_classes)) == 6


@pytest.mark.parametrize("normalized_quality", sorted(CHORD_MANDATORY_INTERVALS))
def test_construct_chord_supported_quality_smoke(normalized_quality: str):
    if normalized_quality == "alt":
        symbol = "Calt"
    else:
        symbol = f"C{normalized_quality}"
    core = parse_chord_core(symbol)
    selected = tuple(range(12))

    result = construct_chord_pitch_classes(core, selected)

    assert core.normalized_quality == normalized_quality
    assert len(result.final_pitch_classes) == 6
    assert len(set(result.final_pitch_classes)) == 6
    assert all(pc in result.final_pitch_classes for pc in result.mandatory_pitch_classes)


def test_construct_chord_returns_mandatory_notes_when_collection_insufficient():
    core = parse_chord_core("Cmaj7")
    selected = _selected_collection(0, 4, 7, 11)

    result = construct_chord_pitch_classes(core, selected)

    assert result.final_pitch_classes == (0, 4, 7, 11)
    assert len(result.final_pitch_classes) == 4
    assert set(result.mandatory_pitch_classes).issubset(set(result.final_pitch_classes))


def test_construct_chord_raises_for_unknown_normalized_quality():
    core = parse_chord_core("Cmaj7")
    forged = core.__class__(
        symbol=core.symbol,
        root=core.root,
        root_pc=core.root_pc,
        quality=core.quality,
        normalized_quality="not-a-quality",
        base_quality=core.base_quality,
        seventh_type=core.seventh_type,
        extensions=core.extensions,
        added_degrees=core.added_degrees,
        altered_degrees=core.altered_degrees,
        omitted_degrees=core.omitted_degrees,
        slash_bass=core.slash_bass,
        slash_bass_pc=core.slash_bass_pc,
        special_semantic_tag=core.special_semantic_tag,
    )

    with pytest.raises(ChordConstructionError, match="Unsupported normalized chord quality"):
        construct_chord_pitch_classes(forged, tuple(range(12)))


# ── variable note count (Chord layer) ─────────────────────────────────────────

def test_bm_slash_a_does_not_raise_with_b_phrygian_collection():
    core = parse_chord_core("Bm/A")
    # B C D E F# G A — modes of G major (B Phrygian / A Dorian)
    selected = _selected_collection(11, 0, 2, 4, 6, 7, 9)

    result = construct_chord_pitch_classes(core, selected)

    assert 11 in result.final_pitch_classes  # B
    assert 2 in result.final_pitch_classes   # D
    assert 6 in result.final_pitch_classes   # F#


def test_bm_slash_a_mandatory_pitch_classes_all_in_final():
    core = parse_chord_core("Bm/A")
    selected = _selected_collection(11, 0, 2, 4, 6, 7, 9)

    result = construct_chord_pitch_classes(core, selected)

    assert set(result.mandatory_pitch_classes).issubset(set(result.final_pitch_classes))


def test_bm_slash_a_final_pitch_classes_only_from_mandatory_and_selected_collection():
    core = parse_chord_core("Bm/A")
    selected = _selected_collection(11, 0, 2, 4, 6, 7, 9)
    selected_set = set(selected)

    result = construct_chord_pitch_classes(core, selected)

    for pc in result.final_pitch_classes:
        assert pc in selected_set or pc in result.mandatory_pitch_classes


def test_bm_slash_a_adds_a_and_e_from_collection():
    core = parse_chord_core("Bm/A")
    selected = _selected_collection(11, 0, 2, 4, 6, 7, 9)

    result = construct_chord_pitch_classes(core, selected)

    assert 9 in result.final_pitch_classes   # A
    assert 4 in result.final_pitch_classes   # E
    assert set(result.final_pitch_classes) == {11, 2, 6, 9, 4}


def test_insufficient_collection_tensions_does_not_raise():
    # Gm in a very restricted collection that has only mandatory tones
    core = parse_chord_core("Gm")
    selected = _selected_collection(7, 10, 2)  # G Bb D — mandatory only, no extra tensions

    result = construct_chord_pitch_classes(core, selected)

    assert len(result.final_pitch_classes) == 3
    assert result.automatic_tension_pitch_classes == ()


def test_csus4_does_not_add_major_third_even_when_in_collection():
    core = parse_chord_core("Csus4")
    # C major scale includes E natural (pc 4)
    selected = _selected_collection(0, 2, 4, 5, 7, 9, 11)

    result = construct_chord_pitch_classes(core, selected)

    assert 4 not in result.final_pitch_classes  # E natural must not appear


def test_c11_does_not_add_major_third_even_when_in_collection():
    core = parse_chord_core("C11")
    # collection includes E (4)
    selected = _selected_collection(0, 2, 4, 5, 7, 9, 10)

    result = construct_chord_pitch_classes(core, selected)

    assert 4 not in result.final_pitch_classes  # E natural must not appear


def test_c5_does_not_add_third_or_flat_third():
    core = parse_chord_core("C5")
    selected = _selected_collection(0, 3, 4, 7)  # includes both E (4) and Eb (3)

    result = construct_chord_pitch_classes(core, selected)

    assert 4 not in result.final_pitch_classes  # E natural
    assert 3 not in result.final_pitch_classes  # Eb


def test_c5_with_minimal_collection_returns_two_notes():
    core = parse_chord_core("C5")
    selected = _selected_collection(0, 7)  # root and fifth only

    result = construct_chord_pitch_classes(core, selected)

    assert set(result.final_pitch_classes) == {0, 7}
    assert len(result.final_pitch_classes) == 2


def test_c7b9_does_not_add_natural9_or_sharp9_as_tension():
    core = parse_chord_core("C7b9")
    # collection has Db(1) [b9 = mandatory], D(2) [natural 9], D#(3) [#9]
    selected = _selected_collection(0, 1, 2, 3, 4, 7, 10)

    result = construct_chord_pitch_classes(core, selected)

    assert 1 in result.mandatory_pitch_classes   # Db (b9) kept
    assert 2 not in result.final_pitch_classes   # D (natural 9) excluded
    assert 3 not in result.final_pitch_classes   # D# (#9) excluded


def test_c7sharp9_does_not_add_flat9_or_natural9_as_tension():
    core = parse_chord_core("C7#9")
    # collection has Db(1) [b9], D(2) [natural 9], D#(3) [#9 = mandatory]
    selected = _selected_collection(0, 1, 2, 3, 4, 7, 10)

    result = construct_chord_pitch_classes(core, selected)

    assert 3 in result.mandatory_pitch_classes   # D# (#9) kept
    assert 1 not in result.final_pitch_classes   # Db (b9) excluded
    assert 2 not in result.final_pitch_classes   # D (natural 9) excluded
