"""Emiuet Session export 用の resolver_core 算出。

``hard_context`` は EUB Changes 内部の和声制約（拡張音込み）として維持し、Emiuet
Session の Solo Mode が向かう安定音 ``resolver_core`` を別に出す。拡張コードでも基本
骨格（概ね R 3 5 7 の 4 和音）に寄せることで、Resolve が 9th/11th へ寄りすぎないように
する。docs/schema は emiuet-session 側 docs/compiled_timeline_schema.md。
"""

from __future__ import annotations

from .harmonic_context import chord_tone_pitch_classes, normalized_harmonic_identity
from .note import semitone_to_pitch_class

# normalized_quality -> resolver_core の root 相対 interval（昇順）。
_RESOLVER_CORE_INTERVALS: dict[str, tuple[int, ...]] = {
    "": (0, 4, 7),  # major triad
    "m": (0, 3, 7),
    "6": (0, 4, 7, 9),  # R 3 5 6
    "m6": (0, 3, 7, 9),  # R b3 5 6
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
    "maj7": (0, 4, 7, 11),  # R 3 5 7
    "maj9": (0, 4, 7, 11),
    "maj13": (0, 4, 7, 11),
    "maj7#5": (0, 4, 8, 11),  # R 3 #5 7
    "m7": (0, 3, 7, 10),  # R b3 5 b7
    "m9": (0, 3, 7, 10),
    "m11": (0, 3, 7, 10),
    "mMaj7": (0, 3, 7, 11),  # R b3 5 7
    "m7b5": (0, 3, 6, 10),  # R b3 b5 b7
    "dim7": (0, 3, 6, 9),  # R b3 b5 bb7
    "7": (0, 4, 7, 10),  # R 3 5 b7
    "9": (0, 4, 7, 10),
    "13": (0, 4, 7, 10),
    "11": (0, 5, 7, 10),  # sus-like dominant: R 4 5 b7
    "7b9": (0, 4, 7, 10),
    "7#9": (0, 4, 7, 10),
    "13b9": (0, 4, 7, 10),
    "7#11": (0, 4, 7, 10),
    "7b13": (0, 4, 7, 10),
    "7#5": (0, 4, 8, 10),  # R 3 #5 b7
    "7#5b9": (0, 4, 8, 10),
    "7b5": (0, 4, 6, 10),  # R 3 b5 b7
    "7b5b9": (0, 4, 6, 10),
    "7#9b5": (0, 4, 6, 10),
    "7sus4": (0, 5, 7, 10),  # R 4 5 b7
    "9sus4": (0, 5, 7, 10),
    "7b9sus4": (0, 5, 7, 10),
    "5": (0, 7),  # power chord: R 5
    "alt": (0, 4, 10),  # outside; minimal R 3 b7 shell
}


def resolver_core_pitch_classes(symbol: str) -> tuple[int, ...]:
    """resolver_core の pitch class を root 相対の昇順で返す。"""
    identity = normalized_harmonic_identity(symbol)
    intervals = _RESOLVER_CORE_INTERVALS.get(identity.normalized_quality)
    if intervals is None:
        # 未知 quality は chord tone 全体に fallback（安全側）。
        return tuple(sorted(chord_tone_pitch_classes(symbol)))
    return tuple((identity.root_pc + i) % 12 for i in intervals)


def resolver_core_names(symbol: str) -> tuple[str, ...]:
    return tuple(semitone_to_pitch_class(pc) for pc in resolver_core_pitch_classes(symbol))
