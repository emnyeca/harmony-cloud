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
from fractions import Fraction

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
TIMELINE_BASIS_VALUES = frozenset({"original_song", "segment_map", "digitone_step"})


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


def _root_ordered_names(pcs, root_pc: int) -> list[str]:
    """scale root からの順序を保った note 名（数値昇順にしない）。"""
    ordered = sorted(pcs, key=lambda pc: (pc - root_pc) % 12)
    return [semitone_to_pitch_class(pc) for pc in ordered]


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
        "hard_context": _names(hard_context_pitch_classes(chord)),  # 昇順で実害なし
        "resolver_core": list(resolver_core_names(chord)),  # root-relative order
        "lpc": _root_ordered_names(lpc, scale_root),  # scale root からの順序
    }


def build_emiuet_compiled_timeline(
    steps: list[CompiledStepInput],
    *,
    timeline_basis: str = "digitone_step",
    ppqn: int = 24,
    original_tempo: float | None = None,
    digitone_tempo: float | None = None,
    meter: str = "4/4",
    loop: bool = True,
) -> dict:
    """compile 済み step 列から Emiuet Session compiled timeline dict を生成する。"""
    if timeline_basis not in TIMELINE_BASIS_VALUES:
        raise ValueError(f"unsupported timeline_basis: {timeline_basis!r}")
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
        "timeline_basis": timeline_basis,
        "clock": {
            "ppqn": ppqn,
            "original_tempo": original_tempo,
            "digitone_tempo": digitone_tempo,
            "meter": meter,
        },
        "form": {"loop": loop, "start_tick": 0, "end_tick": end_tick},
        "steps": out_steps,
    }


def emiuet_timeline_from_chord_events(
    chord_events,
    *,
    performance_tempo,
    device_tempo,
    ppqn: int = 24,
    meter: str = "4/4",
    loop: bool = True,
) -> dict:
    """Track 8 chord events（Syx の chord track と同一ソース）から timeline を作る。

    tick は **Digitone が実際に送る MIDI Clock の grid**（device tempo 基準）へ落とす:

        tick = round(onset_quarters * ppqn * device_tempo / performance_tempo)

    これにより、Emiuet Session 側で SPEED / LENGTH / tempo を再解釈する必要がなく、
    原曲小節ではなく Digitone Step 進行に同期する。``chord_events`` は
    ``id`` / ``symbol`` / ``onset_quarters`` / ``duration_quarters`` を持つ object。
    """
    scale = Fraction(ppqn) * Fraction(device_tempo) / Fraction(performance_tempo)

    def tick(quarters) -> int:
        return int(round(float(Fraction(quarters) * scale)))

    events = list(chord_events)
    starts = [tick(ev.onset_quarters) for ev in events]
    steps: list[CompiledStepInput] = []
    for i, ev in enumerate(events):
        start = starts[i]
        # 連続した tick range にする（次 step の onset を end とし、最後は onset+duration）。
        if i + 1 < len(events):
            end = starts[i + 1]
        else:
            end = tick(Fraction(ev.onset_quarters) + Fraction(ev.duration_quarters))
        if end <= start:
            end = start + 1  # 退避: 0 長は importer が拒否するため最低 1 tick
        steps.append(
            CompiledStepInput(
                id=getattr(ev, "id", f"step_{i:03d}"),
                start_tick=start,
                end_tick=end,
                chord=ev.symbol,
                source_step_index=i,
            )
        )

    return build_emiuet_compiled_timeline(
        steps,
        ppqn=ppqn,
        original_tempo=float(performance_tempo),
        digitone_tempo=float(device_tempo),
        meter=meter,
        loop=loop,
    )


def clock_song_timeline_from_chords(
    chords: list[str],
    *,
    beats_per_chord: float = 4.0,
    ppqn: int = 24,
    tempo: float = 120.0,
    meter: str = "4/4",
    loop: bool = True,
) -> dict:
    """Build an original-song timeline with one chord per bar.

    This is the minimal Changes-side prototype for clock_song mode. It uses the
    original bar grid, not the Digitone step grid.
    """
    if beats_per_chord <= 0:
        raise ValueError("beats_per_chord must be positive")
    ticks_per_chord = int(round(ppqn * beats_per_chord))
    steps = [
        CompiledStepInput(
            id=f"bar_{index + 1:03d}",
            start_tick=index * ticks_per_chord,
            end_tick=(index + 1) * ticks_per_chord,
            chord=chord,
            source_step_index=index,
        )
        for index, chord in enumerate(chords)
    ]
    return build_emiuet_compiled_timeline(
        steps,
        timeline_basis="original_song",
        ppqn=ppqn,
        original_tempo=tempo,
        digitone_tempo=None,
        meter=meter,
        loop=loop,
    )


def manual_timeline_from_chords(
    chords: list[str],
    *,
    ppqn: int = 24,
    tempo: float = 120.0,
    meter: str = "4/4",
    loop: bool = True,
) -> dict:
    """Build a segment-map timeline for manual advance selection."""
    steps = [
        CompiledStepInput(
            id=f"segment_{index + 1:03d}",
            start_tick=index,
            end_tick=index + 1,
            chord=chord,
            source_step_index=index,
        )
        for index, chord in enumerate(chords)
    ]
    return build_emiuet_compiled_timeline(
        steps,
        timeline_basis="segment_map",
        ppqn=ppqn,
        original_tempo=tempo,
        digitone_tempo=None,
        meter=meter,
        loop=loop,
    )
