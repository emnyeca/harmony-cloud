"""Emiuet Session export 用の Contrast Priority context selector。

通常の progression context とは明確に違う色の既存 scale を、resolver_core を維持した
まま選ぶ。単純な reverse ではなく、品質別の優先順位とガードで選択する。

必須条件: contrast の lpc は resolver_core をすべて含む（含まない候補は採用しない）。
contrast を作れない品質（sus / alt / power など）は None を返し、Emiuet Session 側は
progression へ fallback する。
"""

from __future__ import annotations

from .emiuet_resolver_core import resolver_core_names, resolver_core_pitch_classes
from .harmonic_context import hard_context_pitch_classes, normalized_harmonic_identity
from .note import semitone_to_pitch_class

# scale 名 -> root 相対 interval。
_SCALE_TEMPLATES: dict[str, tuple[int, ...]] = {
    "Half-Whole Diminished": (0, 1, 3, 4, 6, 7, 9, 10),
    "Whole-Half Diminished": (0, 2, 3, 5, 6, 8, 9, 11),
    "Whole Tone": (0, 2, 4, 6, 8, 10),
    "Lydian": (0, 2, 4, 6, 7, 9, 11),
    "Lydian Augmented": (0, 2, 4, 6, 8, 9, 11),
    "Melodic Minor": (0, 2, 3, 5, 7, 9, 11),
    "Harmonic Minor": (0, 2, 3, 5, 7, 8, 11),
    "Locrian natural 2": (0, 2, 3, 5, 6, 8, 10),
}

# normalized_quality -> contrast scale 名。掲載が無い品質は contrast なし（None）。
_CONTRAST_BY_QUALITY: dict[str, str] = {
    # maj 系は Lydian（diminished / whole_tone は使わない）。
    "maj7": "Lydian",
    "maj9": "Lydian",
    "maj13": "Lydian",
    "6": "Lydian",
    "maj7#5": "Lydian Augmented",
    # minor 系は Half-Whole Diminished（resolver_core は 4 和音骨格）。
    "m7": "Half-Whole Diminished",
    "m9": "Half-Whole Diminished",
    "m11": "Half-Whole Diminished",
    # dominant 系は Half-Whole Diminished。
    "7": "Half-Whole Diminished",
    "9": "Half-Whole Diminished",
    "13": "Half-Whole Diminished",
    "7b9": "Half-Whole Diminished",
    "7#9": "Half-Whole Diminished",
    "13b9": "Half-Whole Diminished",
    "7#11": "Half-Whole Diminished",
    "7b13": "Half-Whole Diminished",
    # 増 5 度系は Whole Tone。
    "7#5": "Whole Tone",
    "7#5b9": "Whole Tone",
    "aug": "Whole Tone",
    "7b5": "Whole Tone",
    "7b5b9": "Whole Tone",
    "7#9b5": "Whole Tone",
    # その他の品質。
    "m6": "Melodic Minor",
    "mMaj7": "Harmonic Minor",
    "m7b5": "Locrian natural 2",
    "dim7": "Whole-Half Diminished",
    # sus / alt / power は v1 では contrast なし。
}

# 表示タグ。root ベースの scale は "<root> <tag>"、chord ベースは "<chord> <tag>"。
_DISPLAY_TAG = {
    "Half-Whole Diminished": ("chord", "HW"),
    "Whole-Half Diminished": ("chord", "WH"),
    "Whole Tone": ("chord", "WT"),
    "Locrian natural 2": ("chord", "Loc2"),
    "Lydian": ("root", "Lyd"),
    "Lydian Augmented": ("root", "LydAug"),
    "Melodic Minor": ("root", "MelMin"),
    "Harmonic Minor": ("root", "HarMin"),
}


def _names(pcs) -> list[str]:
    return [semitone_to_pitch_class(pc) for pc in pcs]


def contrast_context(symbol: str) -> dict | None:
    """contrast context dict を返す。作れない場合は None。"""
    identity = normalized_harmonic_identity(symbol)
    scale_name = _CONTRAST_BY_QUALITY.get(identity.normalized_quality)
    if scale_name is None:
        return None

    root = identity.root_pc
    lpc_pcs = sorted((root + i) % 12 for i in _SCALE_TEMPLATES[scale_name])
    core_pcs = resolver_core_pitch_classes(symbol)

    # 必須ガード: resolver_core を含まない scale は採用しない。
    if not set(core_pcs).issubset(set(lpc_pcs)):
        return None

    kind, tag = _DISPLAY_TAG[scale_name]
    display = f"{symbol} {tag}" if kind == "chord" else f"{semitone_to_pitch_class(root)} {tag}"

    return {
        "role": "contrast",
        "display": display,
        "scale_name": scale_name,
        "scale_root": semitone_to_pitch_class(root),
        "selection_policy": "contrast_priority",
        "hard_context": _names(sorted(hard_context_pitch_classes(symbol))),
        "resolver_core": list(resolver_core_names(symbol)),
        "lpc": _names(lpc_pcs),
    }
