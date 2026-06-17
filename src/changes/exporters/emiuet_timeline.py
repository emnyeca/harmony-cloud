"""Emiuet Session 用 compiled timeline exporter。

EUB Changes の Digitone II 用 compile 結果（tick 境界つき step 列）を、Emiuet Session
の compiled timeline JSON（schema v2）へ export する。

重要:
- Syx 生成と timeline export で別々の step 解釈をしない。本 exporter は**既に compile
  済みの step（id / source_step_index / start_tick / end_tick / chord）を入力に取る**。
  tick 境界は呼び出し側（Digitone compile pipeline）が確定したものを使い、ここで SPEED /
  LENGTH / tempo を再解釈しない。
- 各 step に progression / contrast context を付与する。
- pitch class は sharp 表記に正規化（enharmonic）。Emiuet importer は flat/sharp 両対応。

schema 仕様は emiuet-session 側 docs/compiled_timeline_schema.md。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..emiuet_contrast_context import contrast_context
from ..emiuet_resolver_core import resolver_core_names
from ..harmonic_context import (
    hard_context_pitch_classes,
    normalized_harmonic_identity,
    resolve_scale_collection_with_retry,
)
from ..note import semitone_to_pitch_class

SCHEMA_NAME = "emnyeca.emiuet_session.compiled_timeline"
SCHEMA_VERSION = 2


@dataclass(frozen=True)
class CompiledStepInput:
    """Digitone compile 済み step（tick 境界は確定済み）。"""

    id: str
    start_tick: int
    end_tick: int
    chord: str
    source_step_index: int | None = None


def _names(pcs) -> list[str]:
    return [semitone_to_pitch_class(pc) for pc in sorted(pcs)]


def _progression_context(symbols: list[str], index: int) -> dict:
    chord = symbols[index]
    lpc, collection = resolve_scale_collection_with_retry(symbols, index)
    identity = normalized_harmonic_identity(chord)
    scale_root = (
        collection.anchor_root_pc if collection.anchor_root_pc is not None else identity.root_pc
    )
    return {
        "role": "progression",
        "display": chord,
        "scale_name": collection.name,
        "scale_root": semitone_to_pitch_class(scale_root),
        "hard_context": _names(hard_context_pitch_classes(chord)),
        "resolver_core": list(resolver_core_names(chord)),
        "lpc": _names(lpc),
    }


def build_emiuet_compiled_timeline(
    steps: list[CompiledStepInput],
    *,
    ppqn: int = 24,
    original_tempo: float | None = None,
    digitone_tempo: float | None = None,
    meter: str = "4/4",
    loop: bool = True,
) -> dict:
    """compile 済み step 列から Emiuet Session compiled timeline dict を生成する。"""
    symbols = [s.chord for s in steps]
    out_steps: list[dict] = []
    for index, step in enumerate(steps):
        contexts: dict[str, dict] = {"progression": _progression_context(symbols, index)}
        contrast = contrast_context(step.chord)
        if contrast is not None:
            contexts["contrast"] = contrast
        out_steps.append(
            {
                "id": step.id,
                "source_step_index": (
                    step.source_step_index if step.source_step_index is not None else index
                ),
                "start_tick": step.start_tick,
                "end_tick": step.end_tick,
                "chord": step.chord,
                "default_context_role": "progression",
                "mod_context_role": "contrast",
                "contexts": contexts,
            }
        )

    end_tick = out_steps[-1]["end_tick"] if out_steps else 0
    return {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "timeline_basis": "digitone_step",
        "clock": {
            "ppqn": ppqn,
            "original_tempo": original_tempo,
            "digitone_tempo": digitone_tempo,
            "meter": meter,
        },
        "form": {"loop": loop, "start_tick": 0, "end_tick": end_tick},
        "steps": out_steps,
    }
