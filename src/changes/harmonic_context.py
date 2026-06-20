"""Context-aware harmonic selection for six-slot voicing generation.

This module owns musical-context decisions:
- chord constituent extraction
- local pitch collection construction (with deterministic retry)
- prioritized scale-collection selection
- output slot extraction by collection family
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .chord_parser import ChordSymbolCore, parse_chord_core
from .note import pitch_class_to_semitone, semitone_to_pitch_class

@dataclass(frozen=True)
class HarmonicIdentity:
    root_pc: int
    normalized_quality: str
    normalized_modifiers: tuple[str, ...]
    slash_bass_pc: int | None


@dataclass(frozen=True)
class ScaleCollection:
    name: str
    family: str
    priority: int
    pitch_classes: frozenset[int]
    anchor_root_pc: int | None
    extraction_rule: str
    ordered_pitch_classes: tuple[int, ...]
    requires_signature_root_match: bool = False


@dataclass(frozen=True)
class RetryResolution:
    local_pitch_collection: frozenset[int]
    selected_collection: ScaleCollection
    retry_level: str
    hard_context_pitch_classes_used: frozenset[int]
    color_hint_pitch_classes: frozenset[int]
    color_hints_applied_to_constraint_set: bool
    final_local_pitch_collection_used_for_selection: frozenset[int]


class UnsupportedHarmonicContextError(ValueError):
    """Raised when no implemented collection can resolve a chord context."""


SUPPORTED_QUALITIES = {
    "",
    "m",
    "6",
    "m6",
    "6/9",
    "m6/9",
    "dim",
    "maj9",
    "maj7#11",
    "maj7#5",
    "maj7",
    "m7",
    "mMaj7",
    "m9",
    "m7b5",
    "dim7",
    "7",
    "9",
    "7b9",
    "7b9#11",
    "7#9",
    "7b5",
    "7#5",
    "7#5b9",
    "7b5b9",
    "13",
    "13b9",
    "aug7",
    "7#11",
    "7b13",
    "7#9b5",
    "7sus4",
    "9sus4",
    "7b9sus4",
    "alt",
    "m11",
    "maj13",
    "aug",
    "5",
    "11",
}

EXTRACTION_HEPTATONIC = "heptatonic_1351379"
EXTRACTION_WHOLE_TONE = "whole_tone_13sharp11sharp5b79"
EXTRACTION_DIM_HALF_WHOLE = "dim_half_whole_1b3b56b7b9"
EXTRACTION_DIM_WHOLE_HALF = "dim_whole_half_1b3b5679"
EXTRACTION_DOMINANT_BLUES = "dominant_blues_1356b7sharp9"

_HEPTATONIC_DEGREE_INDEXES = (0, 2, 4, 5, 6, 1)

_SYMMETRIC_COLLECTION_FAMILIES = {"whole_tone", "diminished"}

_SYMMETRIC_ELIGIBLE_QUALITIES = {
    "dim",
    "dim7",
    "m7b5",
    "7b9",
    "7b9#11",
    "7#9",
    "7b5",
    "7#5",
    "7#11",
    "7b13",
    "7#9b5",
    "7#5b9",
    "7b5b9",
    "13b9",
    "alt",
    "7b9sus4",
    "aug",
}

_HARD_CONTEXT_INTERVALS_BY_QUALITY: dict[str, tuple[int, ...]] = {
    "": (0, 4, 7),
    "dim": (0, 3, 6),
    "maj7": (0, 4, 7, 11),
    "maj9": (0, 4, 7, 11, 2),
    "maj7#11": (0, 4, 7, 11),
    "maj7#5": (0, 4, 8, 11),
    "m7": (0, 3, 7, 10),
    "7": (0, 4, 7, 10),
    "m": (0, 3, 7),
    "6": (0, 4, 7, 9),
    "m6": (0, 3, 7, 9),
    "6/9": (0, 4, 7, 9, 2),
    "m6/9": (0, 3, 7, 9, 2),
    "mMaj7": (0, 3, 7, 11),
    "m9": (0, 3, 7, 10, 2),
    "m7b5": (0, 3, 6, 10),
    "dim7": (0, 3, 6, 9),
    "9": (0, 4, 7, 10, 2),
    "13": (0, 4, 7, 10, 9),
    "13b9": (0, 4, 7, 10, 9),
    "7b9": (0, 4, 7, 10),
    "7b9#11": (0, 4, 7, 10),
    "7#9": (0, 4, 7, 10),
    "7b5": (0, 4, 6, 10),
    "7#5b9": (0, 4, 8, 10),
    "7b5b9": (0, 4, 6, 10),
    "7#11": (0, 4, 7, 10),
    "7b13": (0, 4, 7, 10),
    "7#9b5": (0, 4, 6, 10),
    "7sus4": (0, 5, 7, 10),
    "9sus4": (0, 5, 7, 10, 2),
    "7b9sus4": (0, 5, 7, 10),
    "7#5": (0, 4, 8, 10),
    "alt": (0, 1, 4, 8),
    "m11": (0, 3, 7, 10, 2, 5),
    "maj13": (0, 4, 7, 11, 2, 9),
    "aug": (0, 4, 8),
    "5": (0, 7),
    "11": (0, 5, 7, 10, 2),
}

_COLOR_HINT_INTERVALS_BY_QUALITY: dict[str, tuple[int, ...]] = {
    "7b9": (1,),
    "7b9#11": (1, 6),
    "7#9": (3,),
    "maj7#11": (6,),
    "7#11": (6,),
    "7b13": (8,),
    "7#9b5": (3,),
    "7#5b9": (1,),
    "7b5b9": (1,),
    "13b9": (1,),
    "7b9sus4": (1,),
    "m11": (2, 5),
    "maj13": (2, 9),
    "11": (2, 5),
}


def allows_symmetric_collection_for_current_chord(core: ChordSymbolCore, family: str) -> bool:
    """Return whether current chord semantics allow symmetric collection selection.

    Plain major/minor tonal qualities remain eligible to contribute context tones,
    but cannot *select* whole-tone or diminished as the current output collection.
    """
    if family not in _SYMMETRIC_COLLECTION_FAMILIES:
        return True
    return core.normalized_quality in _SYMMETRIC_ELIGIBLE_QUALITIES


def _family_rank(family: str) -> int:
    if family == "diatonic_dorian":
        return 1
    if family == "harmonic_minor":
        return 2
    if family == "melodic_minor_lydian_dominant":
        return 3
    if family == "whole_tone":
        return 4
    if family == "diminished":
        return 5
    if family == "dominant_blues":
        return 6
    return 99


def _normalized_modifiers(core: ChordSymbolCore) -> tuple[str, ...]:
    parts: set[str] = set()
    parts.update(f"ext:{x}" for x in core.extensions)
    parts.update(f"add:{x}" for x in core.added_degrees)
    parts.update(f"alt:{x}" for x in core.altered_degrees)
    parts.update(f"omit:{x}" for x in core.omitted_degrees)
    if core.special_semantic_tag is not None:
        parts.add(f"tag:{core.special_semantic_tag}")
    return tuple(sorted(parts))

def _ordered_pcs(anchor_root_pc: int, intervals: tuple[int, ...]) -> tuple[int, ...]:
    return tuple((anchor_root_pc + i) % 12 for i in intervals)


def _collection_from_intervals(
    *,
    name: str,
    family: str,
    priority: int,
    anchor_root_pc: int,
    intervals_from_anchor: tuple[int, ...],
    extraction_rule: str,
    requires_signature_root_match: bool = False,
) -> ScaleCollection:
    ordered = _ordered_pcs(anchor_root_pc, intervals_from_anchor)
    return ScaleCollection(
        name=name,
        family=family,
        priority=priority,
        pitch_classes=frozenset(ordered),
        anchor_root_pc=anchor_root_pc,
        extraction_rule=extraction_rule,
        ordered_pitch_classes=ordered,
        requires_signature_root_match=requires_signature_root_match,
    )


def _all_scale_collections() -> tuple[ScaleCollection, ...]:
    out: list[ScaleCollection] = []

    # Priority 1: Dorian-identified diatonic collections (7-note).
    for dorian_root in range(12):
        major_tonic = (dorian_root - 2) % 12
        out.append(
            _collection_from_intervals(
                name=f"{semitone_to_pitch_class(dorian_root)}_dorian_diatonic",
                family="diatonic_dorian",
                priority=1,
                anchor_root_pc=dorian_root,
                intervals_from_anchor=(0, 2, 3, 5, 7, 9, 10),
                extraction_rule=EXTRACTION_HEPTATONIC,
            )
        )
        # Keep the underlying pitch set equal to the related major collection.
        if major_tonic != dorian_root:
            pcs = _ordered_pcs(major_tonic, (0, 2, 4, 5, 7, 9, 11))
            out[-1] = ScaleCollection(
                name=out[-1].name,
                family=out[-1].family,
                priority=out[-1].priority,
                pitch_classes=frozenset(pcs),
                anchor_root_pc=out[-1].anchor_root_pc,
                extraction_rule=out[-1].extraction_rule,
                ordered_pitch_classes=out[-1].ordered_pitch_classes,
                requires_signature_root_match=out[-1].requires_signature_root_match,
            )

    # Priority 2: Harmonic minor.
    for root in range(12):
        out.append(
            _collection_from_intervals(
                name=f"{semitone_to_pitch_class(root)}_harmonic_minor",
                family="harmonic_minor",
                priority=2,
                anchor_root_pc=root,
                intervals_from_anchor=(0, 2, 3, 5, 7, 8, 11),
                extraction_rule=EXTRACTION_HEPTATONIC,
            )
        )

    # Priority 3: Melodic minor + Lydian dominant families.
    for root in range(12):
        out.append(
            _collection_from_intervals(
                name=f"{semitone_to_pitch_class(root)}_melodic_minor",
                family="melodic_minor_lydian_dominant",
                priority=3,
                anchor_root_pc=root,
                intervals_from_anchor=(0, 2, 3, 5, 7, 9, 11),
                extraction_rule=EXTRACTION_HEPTATONIC,
            )
        )
        out.append(
            _collection_from_intervals(
                name=f"{semitone_to_pitch_class(root)}_lydian_dominant",
                family="melodic_minor_lydian_dominant",
                priority=3,
                anchor_root_pc=root,
                intervals_from_anchor=(0, 2, 4, 6, 7, 9, 10),
                extraction_rule=EXTRACTION_HEPTATONIC,
            )
        )

    # Priority 4: Whole tone.
    for root in range(12):
        out.append(
            _collection_from_intervals(
                name=f"{semitone_to_pitch_class(root)}_whole_tone",
                family="whole_tone",
                priority=4,
                anchor_root_pc=root,
                intervals_from_anchor=(0, 2, 4, 6, 8, 10),
                extraction_rule=EXTRACTION_WHOLE_TONE,
            )
        )

    # Priority 5: Diminished collections.
    for root in range(12):
        out.append(
            _collection_from_intervals(
                name=f"{semitone_to_pitch_class(root)}_half_whole_diminished",
                family="diminished",
                priority=5,
                anchor_root_pc=root,
                intervals_from_anchor=(0, 1, 3, 4, 6, 7, 9, 10),
                extraction_rule=EXTRACTION_DIM_HALF_WHOLE,
                requires_signature_root_match=True,
            )
        )
        out.append(
            _collection_from_intervals(
                name=f"{semitone_to_pitch_class(root)}_whole_half_diminished",
                family="diminished",
                priority=5,
                anchor_root_pc=root,
                intervals_from_anchor=(0, 2, 3, 5, 6, 8, 9, 11),
                extraction_rule=EXTRACTION_DIM_WHOLE_HALF,
                requires_signature_root_match=True,
            )
        )

    # Priority 6: Dominant blues fallback collections.
    for root in range(12):
        out.append(
            _collection_from_intervals(
                name=f"{semitone_to_pitch_class(root)}_dominant_blues",
                family="dominant_blues",
                priority=6,
                anchor_root_pc=root,
                intervals_from_anchor=(0, 2, 3, 4, 5, 6, 7, 9, 10),
                extraction_rule=EXTRACTION_DOMINANT_BLUES,
            )
        )

    return tuple(out)


ALL_SCALE_COLLECTIONS = _all_scale_collections()


def normalized_harmonic_identity(symbol: str) -> HarmonicIdentity:
    core = parse_chord_core(symbol)
    return HarmonicIdentity(
        root_pc=core.root_pc,
        normalized_quality=core.normalized_quality,
        normalized_modifiers=_normalized_modifiers(core),
        slash_bass_pc=core.slash_bass_pc,
    )


def chord_tone_pitch_classes(symbol: str, include_bass: bool = False) -> frozenset[int]:
    core = parse_chord_core(symbol)
    intervals_by_quality: dict[str, tuple[int, ...]] = {
        "": (0, 4, 7),
        "dim": (0, 3, 6),
        "maj7": (0, 4, 7, 11),
        "maj9": (0, 4, 7, 11, 2),
        "maj7#11": (0, 4, 7, 11, 6),
        "maj7#5": (0, 4, 8, 11),
        "m7": (0, 3, 7, 10),
        "7": (0, 4, 7, 10),
        "m": (0, 3, 7),
        "6": (0, 4, 7, 9),
        "m6": (0, 3, 7, 9),
        "6/9": (0, 4, 7, 9, 2),
        "m6/9": (0, 3, 7, 9, 2),
        "mMaj7": (0, 3, 7, 11),
        "m9": (0, 3, 7, 10, 2),
        "m7b5": (0, 3, 6, 10),
        "dim7": (0, 3, 6, 9),
        "9": (0, 4, 7, 10, 2),
        "13": (0, 4, 7, 10, 9),
        "13b9": (0, 4, 7, 10, 1, 9),
        "7b9": (0, 4, 7, 10, 1),
        "7b9#11": (0, 4, 7, 10, 1, 6),
        "7#9": (0, 4, 7, 10, 3),
        "7b5": (0, 4, 6, 10),
        "7#5b9": (0, 4, 8, 10, 1),
        "7b5b9": (0, 4, 6, 10, 1),
        "7#11": (0, 4, 7, 10, 6),
        "7b13": (0, 4, 7, 10, 8),
        "7#9b5": (0, 4, 6, 10, 3),
        "7sus4": (0, 5, 7, 10),
        "9sus4": (0, 5, 7, 10, 2),
        "7b9sus4": (0, 5, 7, 10, 1),
        "7#5": (0, 4, 8, 10),
        "alt": (0, 1, 4, 8),
        "m11": (0, 3, 7, 10, 2, 5),
        "maj13": (0, 4, 7, 11, 2, 9),
        "aug": (0, 4, 8),
        "5": (0, 7),
        "11": (0, 5, 7, 10, 2),
    }

    intervals = intervals_by_quality.get(core.normalized_quality)
    if intervals is None:
        raise ValueError(f"Unsupported chord quality: {core.quality}")

    pcs = {(core.root_pc + i) % 12 for i in intervals}
    if include_bass and core.slash_bass_pc is not None:
        pcs.add(core.slash_bass_pc)
    return frozenset(pcs)


def hard_context_pitch_classes(
    symbol: str,
    *,
    include_slash_bass: bool = True,
) -> frozenset[int]:
    core = parse_chord_core(symbol)
    intervals = _HARD_CONTEXT_INTERVALS_BY_QUALITY.get(core.normalized_quality)
    if intervals is None:
        raise ValueError(f"Unsupported chord quality: {core.quality}")

    pcs = {(core.root_pc + i) % 12 for i in intervals}
    if include_slash_bass and core.slash_bass_pc is not None:
        pcs.add(core.slash_bass_pc)
    return frozenset(pcs)


def color_hint_pitch_classes(symbol: str) -> frozenset[int]:
    core = parse_chord_core(symbol)
    intervals = _COLOR_HINT_INTERVALS_BY_QUALITY.get(core.normalized_quality, ())
    return frozenset((core.root_pc + i) % 12 for i in intervals)


def contextual_constraint_pitch_classes(
    symbol: str,
    *,
    include_color_hints: bool,
    include_slash_bass: bool = True,
) -> frozenset[int]:
    pcs = set(hard_context_pitch_classes(symbol, include_slash_bass=include_slash_bass))
    if include_color_hints:
        pcs.update(color_hint_pitch_classes(symbol))
    return frozenset(pcs)


def _normalize_progression(progression: Sequence[str | Sequence[str]]) -> list[str]:
    out: list[str] = []
    for item in progression:
        if isinstance(item, (list, tuple)):
            for s in item:
                text = str(s).strip()
                if text:
                    out.append(text)
        else:
            text = str(item).strip()
            if text:
                out.append(text)
    return out


def _find_distinct_index(
    identities: Sequence[HarmonicIdentity],
    current_index: int,
    direction: int,
    *,
    circular: bool,
) -> int | None:
    n = len(identities)
    if n == 0:
        return None

    current_identity = identities[current_index]
    cursor = current_index
    visited = 0

    while visited < n - 1:
        cursor += direction
        if circular:
            cursor %= n
        elif cursor < 0 or cursor >= n:
            return None

        visited += 1
        if identities[cursor] != current_identity:
            return cursor

    return None


def build_local_pitch_collection(
    progression: Sequence[str | Sequence[str]],
    index: int,
    *,
    circular: bool = True,
    include_slash_bass: bool = True,
) -> frozenset[int]:
    symbols = _normalize_progression(progression)
    if not symbols:
        raise ValueError("progression must not be empty")
    if index < 0 or index >= len(symbols):
        raise IndexError(f"index out of range: {index}")

    identities = [normalized_harmonic_identity(s) for s in symbols]
    local = set(chord_tone_pitch_classes(symbols[index], include_bass=include_slash_bass))

    prev_idx = _find_distinct_index(identities, index, -1, circular=circular)
    if prev_idx is not None:
        local.update(chord_tone_pitch_classes(symbols[prev_idx], include_bass=include_slash_bass))

    next_idx = _find_distinct_index(identities, index, +1, circular=circular)
    if next_idx is not None:
        local.update(chord_tone_pitch_classes(symbols[next_idx], include_bass=include_slash_bass))

    return frozenset(local)


def _collection_scale_intervals_from_root(collection: Iterable[int], root_pc: int) -> tuple[int, ...]:
    rel = sorted(((pc - root_pc) % 12) for pc in collection)
    return tuple(rel)


def _extract_slots_from_heptatonic_intervals(intervals: tuple[int, ...]) -> tuple[int, int, int, int, int, int]:
    if len(intervals) != 7:
        raise ValueError("heptatonic interval extraction requires 7 notes")
    return tuple(intervals[i] for i in _HEPTATONIC_DEGREE_INDEXES)


def _circular_semitone_distance(a: int, b: int) -> int:
    up = (b - a) % 12
    down = (a - b) % 12
    return min(up, down)


def _relative_order_for_signature_root(collection: ScaleCollection, signature_root_pc: int) -> tuple[int, ...] | None:
    try:
        root_index = collection.ordered_pitch_classes.index(signature_root_pc)
    except ValueError:
        return None

    rotated = collection.ordered_pitch_classes[root_index:] + collection.ordered_pitch_classes[:root_index]
    return tuple((pc - signature_root_pc) % 12 for pc in rotated)


def select_scale_collection(symbol: str, local_pitch_collection: set[int] | frozenset[int]) -> ScaleCollection:
    core = parse_chord_core(symbol)
    local = set(local_pitch_collection)

    candidates: list[ScaleCollection] = []
    for candidate in ALL_SCALE_COLLECTIONS:
        pcs = set(candidate.pitch_classes)
        if not local.issubset(pcs):
            continue
        if core.root_pc not in pcs:
            continue
        if candidate.family in _SYMMETRIC_COLLECTION_FAMILIES:
            if not allows_symmetric_collection_for_current_chord(core, candidate.family):
                continue
        if candidate.requires_signature_root_match and candidate.anchor_root_pc != core.root_pc:
            continue
        candidates.append(candidate)

    if not candidates:
        pcs_text = ",".join(semitone_to_pitch_class(pc) for pc in sorted(local))
        raise UnsupportedHarmonicContextError(
            f"Unsupported harmonic context for {symbol}: local pitch collection {{{pcs_text}}}"
        )

    best_priority = min(c.priority for c in candidates)
    pool = [c for c in candidates if c.priority == best_priority]

    def _sort_key(c: ScaleCollection) -> tuple[int, int, int, int, str]:
        anchor = c.anchor_root_pc if c.anchor_root_pc is not None else core.root_pc
        distance = _circular_semitone_distance(core.root_pc, anchor)
        diminished_bias = 0
        if c.family == "diminished" and c.extraction_rule == EXTRACTION_DIM_WHOLE_HALF:
            diminished_bias = 1
        return (distance, diminished_bias, _family_rank(c.family), anchor, c.name)

    pool.sort(key=_sort_key)
    return pool[0]


def resolve_scale_collection_with_retry(
    progression: Sequence[str | Sequence[str]],
    index: int,
    *,
    circular: bool = True,
    include_slash_bass: bool = True,
) -> tuple[frozenset[int], ScaleCollection]:
    resolved = resolve_scale_collection_with_retry_details(
        progression,
        index,
        circular=circular,
        include_slash_bass=include_slash_bass,
    )
    return resolved.final_local_pitch_collection_used_for_selection, resolved.selected_collection


def resolve_scale_collection_with_retry_details(
    progression: Sequence[str | Sequence[str]],
    index: int,
    *,
    circular: bool = True,
    include_slash_bass: bool = True,
) -> RetryResolution:
    symbols = _normalize_progression(progression)
    if not symbols:
        raise ValueError("progression must not be empty")
    if index < 0 or index >= len(symbols):
        raise IndexError(f"index out of range: {index}")

    identities = [normalized_harmonic_identity(s) for s in symbols]
    prev_idx = _find_distinct_index(identities, index, -1, circular=circular)
    next_idx = _find_distinct_index(identities, index, +1, circular=circular)

    attempts: list[tuple[str, tuple[int, ...], bool]] = []

    if prev_idx is not None or next_idx is not None:
        attempt1_indexes = [index]
        if prev_idx is not None:
            attempt1_indexes.append(prev_idx)
        if next_idx is not None:
            attempt1_indexes.append(next_idx)
        attempts.append(("current+previous+next", tuple(attempt1_indexes), False))

    if prev_idx is not None:
        attempt2_indexes = [index, prev_idx]
        attempts.append(("current+previous", tuple(attempt2_indexes), False))

    attempts.append(("current_only", (index,), True))

    last_error: UnsupportedHarmonicContextError | None = None
    current_color_hints = color_hint_pitch_classes(symbols[index])

    def _attempt_resolution(
        label: str,
        indices: tuple[int, ...],
        *,
        include_color_hints: bool,
    ) -> RetryResolution:
        local: set[int] = set()
        hard_used: set[int] = set()
        for i in indices:
            hard = hard_context_pitch_classes(symbols[i], include_slash_bass=include_slash_bass)
            hard_used.update(hard)
            local.update(hard)
        if include_color_hints:
            local.update(current_color_hints)
        local_frozen = frozenset(local)
        selected = select_scale_collection(symbols[index], local_frozen)
        return RetryResolution(
            local_pitch_collection=local_frozen,
            selected_collection=selected,
            retry_level=label,
            hard_context_pitch_classes_used=frozenset(hard_used),
            color_hint_pitch_classes=current_color_hints,
            color_hints_applied_to_constraint_set=include_color_hints and bool(current_color_hints),
            final_local_pitch_collection_used_for_selection=local_frozen,
        )

    def _dominant_blues_refinement(
        dominant_resolution: RetryResolution,
    ) -> RetryResolution | None:
        if dominant_resolution.selected_collection.family != "dominant_blues":
            return None
        if prev_idx is None or next_idx is None:
            return None

        left_resolution: RetryResolution | None = None
        right_resolution: RetryResolution | None = None

        try:
            left_resolution = _attempt_resolution(
                "current+previous",
                (index, prev_idx),
                include_color_hints=False,
            )
        except UnsupportedHarmonicContextError:
            left_resolution = None

        try:
            right_resolution = _attempt_resolution(
                "current+next",
                (index, next_idx),
                include_color_hints=False,
            )
        except UnsupportedHarmonicContextError:
            right_resolution = None

        if left_resolution is not None and left_resolution.selected_collection.family != "dominant_blues":
            return left_resolution
        if right_resolution is not None and right_resolution.selected_collection.family != "dominant_blues":
            return right_resolution
        if right_resolution is not None:
            return right_resolution
        if left_resolution is not None:
            return left_resolution
        return None

    for _label, indices, include_color_hints in attempts:
        try:
            resolved = _attempt_resolution(
                _label,
                indices,
                include_color_hints=include_color_hints,
            )
            if _label in {"current+previous+next", "current+previous"}:
                refined = _dominant_blues_refinement(resolved)
                if refined is not None:
                    return refined
            return resolved
        except UnsupportedHarmonicContextError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error

    raise UnsupportedHarmonicContextError(f"Unsupported harmonic context for {symbols[index]}")


def extract_output_chord_tone_set(symbol: str, selected_collection: ScaleCollection) -> tuple[int, int, int, int, int, int]:
    core = parse_chord_core(symbol)

    if selected_collection.extraction_rule == EXTRACTION_HEPTATONIC:
        rel = _relative_order_for_signature_root(selected_collection, core.root_pc)
        if rel is None or len(rel) != 7:
            raise UnsupportedHarmonicContextError(
                f"Heptatonic extraction unavailable for {symbol} in {selected_collection.name}"
            )
        if core.special_semantic_tag == "sus":
            # Sus-specific heptatonic extraction: 1,4,5,13,b7,9
            if len(rel) != 7:
                raise UnsupportedHarmonicContextError(
                    f"Sus extraction unavailable for {symbol} in {selected_collection.name}"
                )
            intervals = (rel[0], rel[3], rel[4], rel[5], rel[6], rel[1])
        else:
            intervals = _extract_slots_from_heptatonic_intervals(rel)
    elif selected_collection.extraction_rule == EXTRACTION_WHOLE_TONE:
        intervals = (0, 4, 6, 8, 10, 2)
    elif selected_collection.extraction_rule == EXTRACTION_DIM_HALF_WHOLE:
        intervals = (0, 3, 6, 9, 10, 1)
    elif selected_collection.extraction_rule == EXTRACTION_DIM_WHOLE_HALF:
        intervals = (0, 3, 6, 9, 11, 2)
    elif selected_collection.extraction_rule == EXTRACTION_DOMINANT_BLUES:
        intervals = (0, 4, 7, 9, 10, 3)
    else:
        raise UnsupportedHarmonicContextError(
            f"Unknown extraction rule {selected_collection.extraction_rule} for {selected_collection.name}"
        )

    return tuple((core.root_pc + i) % 12 for i in intervals)


def output_chord_tone_names(symbol: str, progression: Sequence[str | Sequence[str]], index: int) -> tuple[str, ...]:
    local, selected = resolve_scale_collection_with_retry(progression, index)
    tones = extract_output_chord_tone_set(symbol, selected)
    return tuple(semitone_to_pitch_class(pc) for pc in tones)
