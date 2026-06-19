"""Pure Chord Engine core for symbol-faithful pitch-class construction.

Produces 2–6 pitch classes per chord: mandatory chord tones are always
included; preferred tensions from the selected collection are added up to six
voices.  Power chords may produce as few as 2 notes; falling short of six is
not an error — Chord layer does not require a fixed voice count.

This module deliberately stops before register realization, velocity policy,
length policy, Track 8 assignment, or any export integration.
"""

from __future__ import annotations

from dataclasses import dataclass

from .chord_parser import ChordSymbolCore
from .note import semitone_to_pitch_class


class ChordConstructionError(ValueError):
    """Raised when the chord engine cannot construct a valid chord.

    This is reserved for truly unrecoverable failures: unknown quality,
    unparseable symbol, or mandatory pitch-class count exceeding six.
    Falling short of six notes is *not* an error.
    """


@dataclass(frozen=True)
class ChordConstructionResult:
    source_symbol: str
    root_pc: int
    normalized_quality: str
    selected_collection_pitch_classes: tuple[int, ...]
    mandatory_intervals: tuple[int, ...]
    mandatory_pitch_classes: tuple[int, ...]
    automatic_excluded_intervals: tuple[int, ...]
    automatic_tension_intervals: tuple[int, ...]
    automatic_tension_pitch_classes: tuple[int, ...]
    final_pitch_classes: tuple[int, ...]
    diagnostics: tuple[str, ...]


CHORD_MANDATORY_INTERVALS: dict[str, tuple[int, ...]] = {
    "": (0, 4, 7),
    "m": (0, 3, 7),
    "6": (0, 4, 7, 9),
    "m6": (0, 3, 7, 9),
    "6/9": (0, 4, 7, 9, 2),
    "m6/9": (0, 3, 7, 9, 2),
    "maj7": (0, 4, 7, 11),
    "maj9": (0, 4, 7, 11, 2),
    "maj7#11": (0, 4, 7, 11, 6),
    "maj7#5": (0, 4, 8, 11),
    "m7": (0, 3, 7, 10),
    "m9": (0, 3, 7, 10, 2),
    "mMaj7": (0, 3, 7, 11),
    "m7b5": (0, 3, 6, 10),
    "dim": (0, 3, 6),
    "dim7": (0, 3, 6, 9),
    "7": (0, 4, 7, 10),
    "9": (0, 4, 7, 10, 2),
    "13": (0, 4, 7, 10, 9),
    "7b9": (0, 4, 7, 10, 1),
    "7#9": (0, 4, 7, 10, 3),
    "13b9": (0, 4, 7, 10, 1, 9),
    "7b9#11": (0, 4, 7, 10, 1, 6),
    "7b5": (0, 4, 6, 10),
    "7#5": (0, 4, 8, 10),
    "7#5b9": (0, 4, 8, 10, 1),
    "7b5b9": (0, 4, 6, 10, 1),
    "7#11": (0, 4, 7, 10, 6),
    "7b13": (0, 4, 7, 10, 8),
    "7#9b5": (0, 4, 6, 10, 3),
    "7sus4": (0, 5, 7, 10),
    "9sus4": (0, 5, 7, 10, 2),
    "7b9sus4": (0, 5, 7, 10, 1),
    "alt": (0, 4, 10),
    "m11": (0, 3, 7, 10, 2, 5),
    "maj13": (0, 4, 7, 11, 2, 9),
    "aug": (0, 4, 8),
    "5": (0, 7),
    "11": (0, 5, 7, 10, 2),
}

CHORD_TENSION_PREFERENCE: dict[str, tuple[int, ...]] = {
    "major_triad": (11, 2, 9, 6, 5),
    "minor_triad": (10, 2, 5, 9, 11),
    "major_seventh": (2, 9, 6, 5),
    "minor_seventh": (2, 5, 9, 8),
    "minor_major": (2, 9, 5, 8),
    "sixth": (2, 6, 5),
    "minor_sixth": (2, 5, 11),
    "dominant_plain": (2, 9, 6, 5),
    "dominant_altered": (1, 3, 8, 6, 2, 9, 5),
    "dominant_augmented": (2, 6, 9, 8),
    "suspended": (2, 9, 1),
    "half_diminished": (2, 5, 9, 8, 1),
    "diminished": (2, 5, 8, 11),
    "augmented": (2, 6, 10, 9, 5),
    "power": (2, 9, 5, 10, 11),
}

_QUALITY_TENSION_FAMILY: dict[str, str] = {
    "": "major_triad",
    "m": "minor_triad",
    "6": "sixth",
    "m6": "minor_sixth",
    "6/9": "sixth",
    "m6/9": "minor_sixth",
    "maj7": "major_seventh",
    "maj9": "major_seventh",
    "maj7#11": "major_seventh",
    "maj7#5": "dominant_augmented",
    "m7": "minor_seventh",
    "m9": "minor_seventh",
    "mMaj7": "minor_major",
    "m7b5": "half_diminished",
    "dim": "diminished",
    "dim7": "diminished",
    "7": "dominant_plain",
    "9": "dominant_plain",
    "13": "dominant_plain",
    "7b9": "dominant_altered",
    "7#9": "dominant_altered",
    "13b9": "dominant_altered",
    "7b9#11": "dominant_altered",
    "7b5": "dominant_altered",
    "7#5": "dominant_augmented",
    "7#5b9": "dominant_altered",
    "7b5b9": "dominant_altered",
    "7#11": "dominant_plain",
    "7b13": "dominant_altered",
    "7#9b5": "dominant_altered",
    "7sus4": "suspended",
    "9sus4": "suspended",
    "7b9sus4": "suspended",
    "alt": "dominant_altered",
    "m11": "minor_seventh",
    "maj13": "major_seventh",
    "aug": "augmented",
    "5": "power",
    "11": "suspended",
}

_QUALITY_TENSION_PREFERENCE_OVERRIDES: dict[str, tuple[int, ...]] = {
    "7#9": (8, 1, 6, 2, 9, 5),
}

_AUTOMATIC_TENSION_EXCLUSIONS_BY_QUALITY: dict[str, frozenset[int]] = {
    "7b9": frozenset({2, 3}),
    "7#9": frozenset({1, 2}),
    "13b9": frozenset({2, 3}),
    "7b9#11": frozenset({2, 3, 5}),
    "7#5b9": frozenset({2, 3}),
    "7b5b9": frozenset({2, 3}),
    "7#9b5": frozenset({1, 2}),
    "7b9sus4": frozenset({2, 3}),
    "7#11": frozenset({5}),
    "maj7#11": frozenset({5}),
    "7b13": frozenset({9}),
}


def _unique_preserving_order(values: tuple[int, ...]) -> tuple[int, ...]:
    out: list[int] = []
    seen: set[int] = set()
    for value in values:
        normalized = value % 12
        if normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return tuple(out)


def _format_pitch_classes(pitch_classes: tuple[int, ...]) -> str:
    names = " ".join(semitone_to_pitch_class(pc) for pc in pitch_classes)
    numbers = ",".join(str(pc) for pc in pitch_classes)
    return f"[{numbers}] ({names})"


def _normalize_selected_collection_pitch_classes(values: tuple[int, ...]) -> tuple[int, ...]:
    normalized: list[int] = []
    seen: set[int] = set()
    for value in values:
        if not isinstance(value, int):
            raise ChordConstructionError(f"selected collection pitch class must be an int: {value!r}")
        if value < 0 or value > 11:
            raise ChordConstructionError(f"selected collection pitch class must be 0..11: {value}")
        if value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return tuple(normalized)


def _mandatory_intervals_for_quality(normalized_quality: str) -> tuple[int, ...]:
    intervals = CHORD_MANDATORY_INTERVALS.get(normalized_quality)
    if intervals is None:
        raise ChordConstructionError(f"Unsupported normalized chord quality for chord engine: {normalized_quality}")
    return _unique_preserving_order(intervals)


def _preference_family_for_quality(normalized_quality: str) -> str:
    family = _QUALITY_TENSION_FAMILY.get(normalized_quality)
    if family is None:
        raise ChordConstructionError(f"Missing chord tension preference family for quality: {normalized_quality}")
    return family


def _preference_order_for_quality(normalized_quality: str) -> tuple[int, ...]:
    return _QUALITY_TENSION_PREFERENCE_OVERRIDES.get(
        normalized_quality,
        CHORD_TENSION_PREFERENCE[_preference_family_for_quality(normalized_quality)],
    )


def _automatic_exclusions_for_quality(normalized_quality: str) -> tuple[int, ...]:
    if normalized_quality == "alt":
        return ()
    return tuple(sorted(_AUTOMATIC_TENSION_EXCLUSIONS_BY_QUALITY.get(normalized_quality, frozenset())))


def _construction_error(
    *,
    source_symbol: str,
    normalized_quality: str,
    mandatory_pitch_classes: tuple[int, ...],
    selected_collection_pitch_classes: tuple[int, ...],
    attempted_preference_family: str,
    reason: str,
) -> ChordConstructionError:
    return ChordConstructionError(
        "Chord construction failed: "
        f"source_symbol={source_symbol} "
        f"normalized_quality={normalized_quality} "
        f"mandatory_pitch_classes={_format_pitch_classes(mandatory_pitch_classes)} "
        f"selected_collection_pitch_classes={_format_pitch_classes(selected_collection_pitch_classes)} "
        f"attempted_preference_family={attempted_preference_family} "
        f"reason={reason}"
    )


def construct_chord_pitch_classes(
    parsed_chord: ChordSymbolCore,
    selected_collection_pitch_classes: tuple[int, ...],
) -> ChordConstructionResult:
    normalized_selected = _normalize_selected_collection_pitch_classes(selected_collection_pitch_classes)
    mandatory_intervals = _mandatory_intervals_for_quality(parsed_chord.normalized_quality)
    mandatory_pitch_classes = _unique_preserving_order(
        tuple((parsed_chord.root_pc + interval) % 12 for interval in mandatory_intervals)
    )

    preference_family = _preference_family_for_quality(parsed_chord.normalized_quality)
    if len(mandatory_pitch_classes) > 6:
        raise _construction_error(
            source_symbol=parsed_chord.symbol,
            normalized_quality=parsed_chord.normalized_quality,
            mandatory_pitch_classes=mandatory_pitch_classes,
            selected_collection_pitch_classes=normalized_selected,
            attempted_preference_family=preference_family,
            reason="mandatory pitch class count exceeds six",
        )

    selected_set = set(normalized_selected)
    current = set(mandatory_pitch_classes)
    automatic_excluded_intervals = _automatic_exclusions_for_quality(parsed_chord.normalized_quality)
    automatic_tension_intervals: list[int] = []
    automatic_tension_pitch_classes: list[int] = []

    if len(current) == 6:
        diagnostics = (
            f"source_symbol={parsed_chord.symbol}",
            f"normalized_quality={parsed_chord.normalized_quality}",
            f"selected_collection_pitch_classes={_format_pitch_classes(normalized_selected)}",
            f"mandatory_intervals={_format_pitch_classes(mandatory_intervals)}",
            f"mandatory_pitch_classes={_format_pitch_classes(mandatory_pitch_classes)}",
            f"preference_family={preference_family}",
            f"automatic_excluded_intervals={automatic_excluded_intervals}",
            "automatic_tension_intervals=[]",
            "automatic_tension_pitch_classes=[]",
            f"final_pitch_classes={_format_pitch_classes(mandatory_pitch_classes)}",
        )
        return ChordConstructionResult(
            source_symbol=parsed_chord.symbol,
            root_pc=parsed_chord.root_pc,
            normalized_quality=parsed_chord.normalized_quality,
            selected_collection_pitch_classes=normalized_selected,
            mandatory_intervals=mandatory_intervals,
            mandatory_pitch_classes=mandatory_pitch_classes,
            automatic_excluded_intervals=automatic_excluded_intervals,
            automatic_tension_intervals=(),
            automatic_tension_pitch_classes=(),
            final_pitch_classes=mandatory_pitch_classes,
            diagnostics=diagnostics,
        )

    for interval in _preference_order_for_quality(parsed_chord.normalized_quality):
        normalized_interval = interval % 12
        if normalized_interval in automatic_excluded_intervals:
            continue
        pitch_class = (parsed_chord.root_pc + normalized_interval) % 12
        if pitch_class in current:
            continue
        if pitch_class not in selected_set:
            continue
        automatic_tension_intervals.append(normalized_interval)
        automatic_tension_pitch_classes.append(pitch_class)
        current.add(pitch_class)
        if len(current) == 6:
            break

    final_pitch_classes = mandatory_pitch_classes + tuple(automatic_tension_pitch_classes)

    diagnostics = (
        f"source_symbol={parsed_chord.symbol}",
        f"normalized_quality={parsed_chord.normalized_quality}",
        f"selected_collection_pitch_classes={_format_pitch_classes(normalized_selected)}",
        f"mandatory_intervals={_format_pitch_classes(mandatory_intervals)}",
        f"mandatory_pitch_classes={_format_pitch_classes(mandatory_pitch_classes)}",
        f"preference_family={preference_family}",
        f"automatic_excluded_intervals={automatic_excluded_intervals}",
        f"automatic_tension_intervals={_format_pitch_classes(tuple(automatic_tension_intervals))}",
        f"automatic_tension_pitch_classes={_format_pitch_classes(tuple(automatic_tension_pitch_classes))}",
        f"final_pitch_classes={_format_pitch_classes(final_pitch_classes)}",
    )

    return ChordConstructionResult(
        source_symbol=parsed_chord.symbol,
        root_pc=parsed_chord.root_pc,
        normalized_quality=parsed_chord.normalized_quality,
        selected_collection_pitch_classes=normalized_selected,
        mandatory_intervals=mandatory_intervals,
        mandatory_pitch_classes=mandatory_pitch_classes,
        automatic_excluded_intervals=automatic_excluded_intervals,
        automatic_tension_intervals=tuple(automatic_tension_intervals),
        automatic_tension_pitch_classes=tuple(automatic_tension_pitch_classes),
        final_pitch_classes=final_pitch_classes,
        diagnostics=diagnostics,
    )
