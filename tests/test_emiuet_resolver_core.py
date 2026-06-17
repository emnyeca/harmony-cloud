"""Emiuet Session export 用 resolver_core の分離テスト。"""

from __future__ import annotations

from changes.emiuet_resolver_core import resolver_core_names, resolver_core_pitch_classes
from changes.harmonic_context import hard_context_pitch_classes
from changes.note import pitch_class_to_semitone as pc


def _pcs(names):
    return {pc(n) for n in names}


def test_dm11_hard_context_and_resolver_core_differ():
    # hard_context = D F A C E G、resolver_core = D F A C
    assert set(hard_context_pitch_classes("Dm11")) == _pcs(["D", "F", "A", "C", "E", "G"])
    assert list(resolver_core_names("Dm11")) == ["D", "F", "A", "C"]


def test_g9_hard_context_and_resolver_core_differ():
    assert set(hard_context_pitch_classes("G9")) == _pcs(["G", "B", "D", "F", "A"])
    assert list(resolver_core_names("G9")) == ["G", "B", "D", "F"]


def test_quality_shells():
    assert list(resolver_core_names("Cmaj7")) == ["C", "E", "G", "B"]  # R 3 5 7
    assert list(resolver_core_names("Dm7")) == ["D", "F", "A", "C"]  # R b3 5 b7
    assert list(resolver_core_names("G7")) == ["G", "B", "D", "F"]  # R 3 5 b7
    assert list(resolver_core_names("C6")) == ["C", "E", "G", "A"]  # R 3 5 6
    assert list(resolver_core_names("G7#5")) == ["G", "B", "D#", "F"]  # R 3 #5 b7
    assert list(resolver_core_names("Cm7b5")) == ["C", "D#", "F#", "A#"]  # R b3 b5 b7


def test_resolver_core_subset_of_hard_context_for_extensions():
    # 拡張コードでも resolver_core は hard_context の部分集合（骨格に寄せる）。
    for symbol in ("Dm11", "G9", "G13", "Cmaj9"):
        assert set(resolver_core_pitch_classes(symbol)).issubset(set(hard_context_pitch_classes(symbol)))
