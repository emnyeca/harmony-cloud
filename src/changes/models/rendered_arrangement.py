from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Literal, Optional, Tuple, List, Dict, Any

LayerRole = Literal["cloud", "chord", "bass", "lpc_lead"]
ChordLengthMode = Literal["explicit_event_length", "inherit"]


@dataclass(frozen=True)
class RenderedLayerNote:
    note_midi: int
    velocity: Optional[int | str] = None
    lane_id: Optional[str] = None
    degree_label: Optional[str] = None
    diagnostics: Tuple[str, ...] = ()


@dataclass(frozen=True)
class RenderedCloudLayer:
    role: Literal["cloud"]
    notes: Tuple[RenderedLayerNote, ...]
    diagnostics: Tuple[str, ...] = ()


@dataclass(frozen=True)
class RenderedChordLayer:
    role: Literal["chord"]
    source_pitch_classes: Tuple[int, ...]
    canonical_stacked_midi_notes: Tuple[int, ...]
    realized_midi_notes: Tuple[int, ...]
    velocities: Tuple[int, ...]
    length_mode: ChordLengthMode
    notes: Tuple[RenderedLayerNote, ...]
    diagnostics: Tuple[str, ...] = ()


@dataclass(frozen=True)
class RenderedBassLayer:
    role: Literal["bass"]
    note: RenderedLayerNote
    source_pitch_class: Optional[int] = None
    diagnostics: Tuple[str, ...] = ()


@dataclass(frozen=True)
class RenderedLpcLeadLayer:
    """LPC Lead Layer (v0.3.0 beta) — 8 notes for Tracks 9–16."""
    role: Literal["lpc_lead"]
    notes: Tuple[RenderedLayerNote, ...]
    source_pitch_classes: Tuple[int, ...]
    range_start_midi: int
    diagnostics: Tuple[str, ...] = ()


@dataclass(frozen=True)
class RenderedHarmonyOccurrence:
    id: str
    source_harmony_id: str
    symbol: str
    onset_quarters: Fraction
    duration_quarters: Fraction
    cloud: Optional[RenderedCloudLayer] = None
    chord: Optional[RenderedChordLayer] = None
    bass: Optional[RenderedBassLayer] = None
    lpc_lead: Optional[RenderedLpcLeadLayer] = None
    diagnostics: Tuple[str, ...] = ()


@dataclass(frozen=True)
class RenderedArrangement:
    title: str
    performance_tempo: Fraction
    occurrences: Tuple[RenderedHarmonyOccurrence, ...]
    diagnostics: Tuple[str, ...] = ()



def rendered_arrangement_to_dict(arrangement: RenderedArrangement) -> Dict[str, Any]:
    """Serialize a RenderedArrangement to a deterministic dictionary representation."""
    def to_fraction_str(frac: Fraction) -> str:
        return str(frac)

    data: Dict[str, Any] = {
        "version": 1,
        "type": "rendered_arrangement",
        "title": arrangement.title,
        "performance_tempo": to_fraction_str(arrangement.performance_tempo),
        "occurrences": [],
        "diagnostics": list(arrangement.diagnostics),
    }

    for occ in arrangement.occurrences:
        occ_dict: Dict[str, Any] = {
            "id": occ.id,
            "source_harmony_id": occ.source_harmony_id,
            "symbol": occ.symbol,
            "onset_quarters": to_fraction_str(occ.onset_quarters),
            "duration_quarters": to_fraction_str(occ.duration_quarters),
            "cloud": None,
            "chord": None,
            "bass": None,
            "diagnostics": list(occ.diagnostics),
        }
        if occ.cloud:
            occ_dict["cloud"] = {
                "role": "cloud",
                "notes": [
                    {
                        "note_midi": note.note_midi,
                        "velocity": note.velocity,
                        "lane_id": note.lane_id,
                        "degree_label": note.degree_label,
                        "diagnostics": list(note.diagnostics),
                    }
                    for note in occ.cloud.notes
                ],
                "diagnostics": list(occ.cloud.diagnostics),
            }
        if occ.chord:
            occ_dict["chord"] = {
                "role": "chord",
                "source_pitch_classes": list(occ.chord.source_pitch_classes),
                "canonical_stacked_midi_notes": list(occ.chord.canonical_stacked_midi_notes),
                "realized_midi_notes": list(occ.chord.realized_midi_notes),
                "velocities": list(occ.chord.velocities),
                "length_mode": occ.chord.length_mode,
                "notes": [
                    {
                        "note_midi": note.note_midi,
                        "velocity": note.velocity,
                        "lane_id": note.lane_id,
                        "degree_label": note.degree_label,
                        "diagnostics": list(note.diagnostics),
                    }
                    for note in occ.chord.notes
                ],
                "diagnostics": list(occ.chord.diagnostics),
            }
        if occ.bass:
            note = occ.bass.note
            occ_dict["bass"] = {
                "role": "bass",
                "note": {
                    "note_midi": note.note_midi,
                    "velocity": note.velocity,
                    "lane_id": note.lane_id,
                    "degree_label": note.degree_label,
                    "diagnostics": list(note.diagnostics),
                },
                "source_pitch_class": occ.bass.source_pitch_class,
                "diagnostics": list(occ.bass.diagnostics),
            }
        data["occurrences"].append(occ_dict)
    return data


def rendered_arrangement_from_dict(data: Dict[str, Any]) -> RenderedArrangement:
    """Deserialize a dictionary into a RenderedArrangement."""
    if data.get("type") != "rendered_arrangement" or data.get("version") != 1:
        raise ValueError("Invalid rendered arrangement data")

    def to_fraction(val: str) -> Fraction:
        return Fraction(val)

    def tuple_notes(note_dicts: List[Dict[str, Any]]) -> Tuple[RenderedLayerNote, ...]:
        return tuple(
            RenderedLayerNote(
                note_midi=nd["note_midi"],
                velocity=nd.get("velocity"),
                lane_id=nd.get("lane_id"),
                degree_label=nd.get("degree_label"),
                diagnostics=tuple(nd.get("diagnostics", [])),
            )
            for nd in note_dicts
        )

    occurrences: List[RenderedHarmonyOccurrence] = []
    for occ_dict in data["occurrences"]:
        cloud_layer = None
        if occ_dict.get("cloud"):
            cl = occ_dict["cloud"]
            cloud_layer = RenderedCloudLayer(
                role="cloud",
                notes=tuple_notes(cl["notes"]),
                diagnostics=tuple(cl.get("diagnostics", [])),
            )
        chord_layer = None
        if occ_dict.get("chord"):
            cl = occ_dict["chord"]
            chord_layer = RenderedChordLayer(
                role="chord",
                source_pitch_classes=tuple(cl["source_pitch_classes"]),
                canonical_stacked_midi_notes=tuple(cl["canonical_stacked_midi_notes"]),
                realized_midi_notes=tuple(cl["realized_midi_notes"]),
                velocities=tuple(cl["velocities"]),
                length_mode=cl["length_mode"],
                notes=tuple_notes(cl["notes"]),
                diagnostics=tuple(cl.get("diagnostics", [])),
            )
        bass_layer = None
        if occ_dict.get("bass"):
            bl = occ_dict["bass"]
            note_dict = bl["note"]
            bass_layer = RenderedBassLayer(
                role="bass",
                note=RenderedLayerNote(
                    note_midi=note_dict["note_midi"],
                    velocity=note_dict.get("velocity"),
                    lane_id=note_dict.get("lane_id"),
                    degree_label=note_dict.get("degree_label"),
                    diagnostics=tuple(note_dict.get("diagnostics", [])),
                ),
                source_pitch_class=bl.get("source_pitch_class"),
                diagnostics=tuple(bl.get("diagnostics", [])),
            )
        occurrences.append(
            RenderedHarmonyOccurrence(
                id=occ_dict["id"],
                source_harmony_id=occ_dict["source_harmony_id"],
                symbol=occ_dict["symbol"],
                onset_quarters=to_fraction(occ_dict["onset_quarters"]),
                duration_quarters=to_fraction(occ_dict["duration_quarters"]),
                cloud=cloud_layer,
                chord=chord_layer,
                bass=bass_layer,
                diagnostics=tuple(occ_dict.get("diagnostics", [])),
            )
        )
    arrangement = RenderedArrangement(
        title=data["title"],
        performance_tempo=Fraction(data["performance_tempo"]),
        occurrences=tuple(occurrences),
        diagnostics=tuple(data.get("diagnostics", [])),
    )
    return arrangement
