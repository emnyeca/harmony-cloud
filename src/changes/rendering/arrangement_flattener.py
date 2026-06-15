"""Flatten RenderedArrangement into export-compatible RenderedTimeline.

This adapter preserves timing, source identity, lane assignment, and per-note
velocity where the layer supplies it.

Trigger policy is applied here, at flatten time:
  - hold_until_change: contiguous same-pitch events on the same voice are merged
    into a single longer event.
  - retrigger: each harmony occurrence produces a separate event regardless of pitch.
"""

from __future__ import annotations

from changes.models.render_profile import RenderProfile, default_render_profile
from changes.models.rendered_arrangement import RenderedArrangement
from changes.models.rendered_timeline import RenderedNoteEvent, RenderedTimeline

DEFAULT_LAYERS = ("cloud", "bass", "chord", "lpc_lead")
_ALLOWED_LAYERS = frozenset(DEFAULT_LAYERS)

_ROLE_ORDER = {
    "cloud": 0,
    "chord": 1,
    "bass": 2,
    "lpc_lead": 3,
}


def _sort_key(event: RenderedNoteEvent) -> tuple[object, int, str, str]:
    return (
        event.onset_quarters,
        _ROLE_ORDER.get(event.role, 99),
        event.voice_id,
        event.id,
    )


def normalize_layers(layers: str | list[str] | tuple[str, ...] | set[str] | None) -> tuple[str, ...]:
    if layers is None:
        return DEFAULT_LAYERS

    raw_layers = layers.split(",") if isinstance(layers, str) else list(layers)
    normalized = tuple(layer.strip().lower() for layer in raw_layers if layer.strip())
    if not normalized:
        raise ValueError("layers must contain at least one of: cloud, bass, chord")

    unknown = sorted(set(normalized) - _ALLOWED_LAYERS)
    if unknown:
        raise ValueError(f"Unsupported layer selection: {', '.join(unknown)}")

    return tuple(layer for layer in DEFAULT_LAYERS if layer in normalized)


def _merge_hold_events(events: list[RenderedNoteEvent]) -> list[RenderedNoteEvent]:
    """Merge contiguous same-pitch events per voice for hold_until_change policy."""
    by_voice: dict[str, list[RenderedNoteEvent]] = {}
    for event in events:
        by_voice.setdefault(event.voice_id, []).append(event)

    merged: list[RenderedNoteEvent] = []
    for voice_events in by_voice.values():
        sorted_events = sorted(voice_events, key=lambda e: e.onset_quarters)
        current = sorted_events[0]
        for next_ev in sorted_events[1:]:
            is_contiguous = current.onset_quarters + current.duration_quarters == next_ev.onset_quarters
            if is_contiguous and current.note_midi == next_ev.note_midi:
                current = RenderedNoteEvent(
                    id=current.id,
                    voice_id=current.voice_id,
                    role=current.role,
                    note_midi=current.note_midi,
                    onset_quarters=current.onset_quarters,
                    duration_quarters=current.duration_quarters + next_ev.duration_quarters,
                    source_harmony_id=current.source_harmony_id,
                    retrigger=False,
                    velocity=current.velocity,
                )
            else:
                merged.append(current)
                current = next_ev
        merged.append(current)
    return merged


def flatten_arrangement_to_timeline(
    arrangement: RenderedArrangement,
    *,
    layers: str | list[str] | tuple[str, ...] | set[str] | None = None,
    render_profile: RenderProfile | None = None,
) -> RenderedTimeline:
    profile = render_profile if render_profile is not None else default_render_profile()
    trigger_policies = {
        "cloud": profile.cloud_trigger_policy,
        "bass": profile.bass_trigger_policy,
        "chord": profile.chord_trigger_policy,
        "lpc_lead": profile.cloud_trigger_policy,  # LPC Lead follows Cloud timing policy
    }

    selected_layers = set(normalize_layers(layers))
    layer_events: dict[str, list[RenderedNoteEvent]] = {"cloud": [], "bass": [], "chord": [], "lpc_lead": []}

    for occurrence_index, occurrence in enumerate(arrangement.occurrences, start=1):
        occurrence_key = occurrence.id if occurrence.id else f"occ{occurrence_index}"

        if "cloud" in selected_layers and occurrence.cloud is not None:
            for note_index, note in enumerate(occurrence.cloud.notes, start=1):
                layer_events["cloud"].append(
                    RenderedNoteEvent(
                        id=f"{occurrence_key}_cloud_{note_index}",
                        voice_id=note.lane_id or f"cloud_voice_{note_index}",
                        role="cloud",
                        note_midi=note.note_midi,
                        onset_quarters=occurrence.onset_quarters,
                        duration_quarters=occurrence.duration_quarters,
                        source_harmony_id=occurrence.source_harmony_id,
                        retrigger=True,
                        velocity=note.velocity,
                    )
                )

        if "chord" in selected_layers and occurrence.chord is not None:
            for note_index, note in enumerate(occurrence.chord.notes, start=1):
                layer_events["chord"].append(
                    RenderedNoteEvent(
                        id=f"{occurrence_key}_chord_{note_index}",
                        voice_id=note.lane_id or f"chord_note_{note_index}",
                        role="chord",
                        note_midi=note.note_midi,
                        onset_quarters=occurrence.onset_quarters,
                        duration_quarters=occurrence.duration_quarters,
                        source_harmony_id=occurrence.source_harmony_id,
                        retrigger=True,
                        velocity=note.velocity,
                    )
                )

        if "bass" in selected_layers and occurrence.bass is not None:
            layer_events["bass"].append(
                RenderedNoteEvent(
                    id=f"{occurrence_key}_bass",
                    voice_id=occurrence.bass.note.lane_id or "bass",
                    role="bass",
                    note_midi=occurrence.bass.note.note_midi,
                    onset_quarters=occurrence.onset_quarters,
                    duration_quarters=occurrence.duration_quarters,
                    source_harmony_id=occurrence.source_harmony_id,
                    retrigger=True,
                    velocity=occurrence.bass.note.velocity,
                )
            )

        if "lpc_lead" in selected_layers and occurrence.lpc_lead is not None:
            for note_index, note in enumerate(occurrence.lpc_lead.notes, start=1):
                layer_events["lpc_lead"].append(
                    RenderedNoteEvent(
                        id=f"{occurrence_key}_lpc_lead_{note_index}",
                        voice_id=note.lane_id or f"lpc_lead_slot_{note_index}",
                        role="lpc_lead",
                        note_midi=note.note_midi,
                        onset_quarters=occurrence.onset_quarters,
                        duration_quarters=occurrence.duration_quarters,
                        source_harmony_id=occurrence.source_harmony_id,
                        retrigger=True,
                        velocity=note.velocity,
                    )
                )

    events: list[RenderedNoteEvent] = []
    for layer_name in DEFAULT_LAYERS:
        if layer_name not in selected_layers:
            continue
        raw = layer_events[layer_name]
        if trigger_policies[layer_name] == "hold_until_change":
            events.extend(_merge_hold_events(raw))
        else:
            events.extend(raw)

    return RenderedTimeline(
        title=arrangement.title,
        performance_tempo=arrangement.performance_tempo,
        events=tuple(sorted(events, key=_sort_key)),
    )
