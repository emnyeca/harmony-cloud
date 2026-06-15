"""Shared UI pipeline helpers for Preview / Export / Dry-run / Send.

All operations that transform SongModel + AppSettings into output data
(SysEx bytes, preview events, pattern count) live here so the Streamlit UI
does not own the conversion details.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from changes.app_settings import AppSettings
from changes.lpc_lead_layer import LPC_LEAD_FIRST_TRACK, LPC_LEAD_SLOT_COUNT, lpc_lead_range_start_midi
from changes.models.digitone_target_profile import (
    DEFAULT_SPEED_CANDIDATES,
    DigitoneTargetProfile,
    LayerRouting,
    VoiceRouting,
)
from changes.models.render_profile import RenderProfile
from changes.models.rendered_arrangement import RenderedArrangement
from changes.models.rendered_timeline import RenderedNoteEvent, RenderedTimeline
from changes.models.song_model import SongModel

CLOUD_RANGE_SEMITONES = 18
CHANGES_LAYER_MIN_TRACK = 1
CHANGES_LAYER_MAX_TRACK = 8
CLOUD_VOICE_COLUMNS = tuple(f"Voice {i}" for i in range(1, 7))


@dataclass
class UiCompiledSong:
    song: SongModel
    render_profile: RenderProfile
    target_profile: DigitoneTargetProfile
    arrangement: RenderedArrangement
    timeline: RenderedTimeline


def settings_to_render_profile(settings: AppSettings) -> RenderProfile:
    cc = settings.cloud_center_midi
    bc = settings.bass_center_midi
    ch = settings.chord_center_midi
    lpc_range_start = lpc_lead_range_start_midi(
        settings.lpc_lead_layer_range_root,
        settings.lpc_lead_layer_octave,
    )
    return RenderProfile(
        name="ui_custom",
        voices=6,
        voice_leading_strategy="minimum_motion",
        bass_enabled=settings.bass_track is not None,
        bass_strategy="slash_or_root_in_window",
        cloud_trigger_policy=settings.cloud_trigger_policy,
        bass_trigger_policy=settings.bass_trigger_policy,
        chord_trigger_policy=settings.chord_trigger_policy,
        cloud_center_midi=cc,
        cloud_spread_min=settings.cloud_spread_min,
        cloud_spread_max=settings.cloud_spread_max,
        cloud_average_tolerance=settings.cloud_average_tolerance,
        cloud_min_midi=cc - CLOUD_RANGE_SEMITONES // 2,
        cloud_max_midi=cc + CLOUD_RANGE_SEMITONES // 2,
        chord_min_midi=ch - 12,
        chord_max_midi=ch + 12,
        bass_min_midi=bc,
        bass_max_midi=bc + 11,
        lpc_lead_enabled=settings.lpc_lead_layer_enabled,
        lpc_lead_range_start_midi=lpc_range_start,
    )


def settings_to_target_profile(settings: AppSettings) -> DigitoneTargetProfile:
    """Build a DigitoneTargetProfile from AppSettings routing fields.

    Voices assigned to None are excluded from the routing entirely.
    Cloud voices may share tracks with each other. Cloud, Bass, and Chord
    layer categories must not share tracks.
    """
    layer_tracks: dict[str, set[int]] = {"cloud": set(), "bass": set(), "chord": set()}

    def _validate_track(track: int | None, label: str) -> None:
        if track is None:
            return
        if track < CHANGES_LAYER_MIN_TRACK or track > CHANGES_LAYER_MAX_TRACK:
            raise ValueError(
                f"{label} track must be in {CHANGES_LAYER_MIN_TRACK}..{CHANGES_LAYER_MAX_TRACK} "
                "because Tracks 9..16 are reserved for user arrangement tracks"
            )

    for i, track in enumerate(settings.cloud_tracks[:6], start=1):
        _validate_track(track, f"Cloud voice {i}")
        if track is not None:
            layer_tracks["cloud"].add(track)
    _validate_track(settings.bass_track, "Bass")
    if settings.bass_track is not None:
        layer_tracks["bass"].add(settings.bass_track)
    _validate_track(settings.chord_track, "Chord")
    if settings.chord_track is not None:
        layer_tracks["chord"].add(settings.chord_track)

    for left, right in (("cloud", "bass"), ("cloud", "chord"), ("bass", "chord")):
        shared = sorted(layer_tracks[left] & layer_tracks[right])
        if shared:
            raise ValueError(
                f"Digitone layer track conflict: {left} and {right} cannot share track(s) {shared}"
            )

    cloud_voices: dict[str, VoiceRouting] = {}
    for i, track in enumerate(settings.cloud_tracks[:6]):
        if track is not None:
            cloud_voices[f"cloud_voice_{i + 1}"] = VoiceRouting(track=track)

    bass_voices: dict[str, VoiceRouting] = {}
    if settings.bass_track is not None:
        bass_voices["bass"] = VoiceRouting(track=settings.bass_track)

    chord_voices: dict[str, VoiceRouting] = {}
    if settings.chord_track is not None:
        for i in range(1, 7):
            chord_voices[f"chord_note_{i}"] = VoiceRouting(track=settings.chord_track)

    # polyphonic: chord track + any track with 2+ assigned voices
    track_count: dict[int, int] = {}
    for vr in [*cloud_voices.values(), *bass_voices.values(), *chord_voices.values()]:
        track_count[vr.track] = track_count.get(vr.track, 0) + 1
    poly: set[int] = set()
    if settings.chord_track is not None:
        poly.add(settings.chord_track)
    for t, cnt in track_count.items():
        if cnt >= 2:
            poly.add(t)

    routing: dict[str, LayerRouting] = {}
    if cloud_voices:
        routing["cloud"] = LayerRouting(voices=cloud_voices)
    if bass_voices:
        routing["bass"] = LayerRouting(voices=bass_voices)
    if chord_voices:
        routing["chord"] = LayerRouting(voices=chord_voices)

    if settings.lpc_lead_layer_enabled:
        lpc_voices: dict[str, VoiceRouting] = {
            f"lpc_lead_slot_{i}": VoiceRouting(track=LPC_LEAD_FIRST_TRACK + i - 1)
            for i in range(1, LPC_LEAD_SLOT_COUNT + 1)
        }
        routing["lpc_lead"] = LayerRouting(voices=lpc_voices)

    return DigitoneTargetProfile(
        name="ui_custom",
        device="digitone2",
        routing=routing,
        default_velocity="inherit",
        length_strategy="hold_until_next_event",
        allow_inf=False,
        approximation="error",
        preferred_speed=DEFAULT_SPEED_CANDIDATES[0],
        fallback_speeds=DEFAULT_SPEED_CANDIDATES[1:],
        track_default_velocity=None,
        polyphonic_tracks=tuple(sorted(poly)),
    )


def compile_song_for_ui(song: SongModel, settings: AppSettings) -> UiCompiledSong:
    from changes.rendering.arrangement_renderer import render_arrangement
    from changes.rendering.arrangement_flattener import flatten_arrangement_to_timeline

    rp = settings_to_render_profile(settings)
    tp = settings_to_target_profile(settings)
    arrangement = render_arrangement(song, rp)
    timeline = flatten_arrangement_to_timeline(arrangement, render_profile=rp)
    return UiCompiledSong(
        song=song,
        render_profile=rp,
        target_profile=tp,
        arrangement=arrangement,
        timeline=timeline,
    )


def build_cloud_voice_leading_dataframe(song: SongModel, settings: AppSettings):
    """Return Cloud voice pitches by generated harmonic step for UI graphing.

    The data comes from the same render pipeline used by Preview/Send, but reads
    the pre-flattened arrangement so hold/merge policies do not hide sustained
    voice-leading steps.
    """
    import pandas as pd

    compiled = compile_song_for_ui(song, settings)
    rows: list[dict[str, int | None]] = []
    indexes: list[int] = []
    has_cloud_value = False

    for step, occurrence in enumerate(compiled.arrangement.occurrences):
        row: dict[str, int | None] = {column: None for column in CLOUD_VOICE_COLUMNS}
        if occurrence.cloud is not None:
            for note in occurrence.cloud.notes:
                lane_id = str(note.lane_id or "")
                if not lane_id.startswith("cloud_voice_"):
                    continue
                try:
                    voice_index = int(lane_id.rsplit("_", 1)[1])
                except ValueError:
                    continue
                if 1 <= voice_index <= len(CLOUD_VOICE_COLUMNS):
                    row[f"Voice {voice_index}"] = int(note.note_midi)
                    has_cloud_value = True
        rows.append(row)
        indexes.append(step)

    if not has_cloud_value:
        return pd.DataFrame(columns=list(CLOUD_VOICE_COLUMNS))
    return pd.DataFrame(rows, index=indexes, columns=list(CLOUD_VOICE_COLUMNS))


def count_auto_split_patterns(song: SongModel, settings: AppSettings) -> int:
    """Return how many Digitone patterns the song compiles to under current settings."""
    try:
        from changes.digitone.bundle_planner import compile_timeline_to_digitone_bundle_plan

        compiled = compile_song_for_ui(song, settings)
        bundle_plan = compile_timeline_to_digitone_bundle_plan(
            compiled.song, compiled.timeline, compiled.target_profile
        )
        return len(bundle_plan.patterns)
    except Exception:
        return 1


def count_linear_patterns(song: SongModel, settings: AppSettings) -> int:
    """Return how many section-boundary Linear Auto Split patterns the song needs."""
    try:
        from changes.digitone.linear_split_planner import compile_timeline_to_digitone_linear_split_plan

        compiled = compile_song_for_ui(song, settings)
        plan = compile_timeline_to_digitone_linear_split_plan(
            compiled.song,
            compiled.timeline,
            compiled.target_profile,
        )
        return len(plan.patterns)
    except Exception:
        return 1


def song_to_syx_bytes(song: SongModel, settings: AppSettings) -> bytes:
    """Compile song to a Digitone SysEx blob using the current settings routing."""
    import yaml
    from changes.digitone.planner import compile_timeline_to_digitone_plan
    from changes.exporters.digitone_events import digitone_compile_plan_to_events_yaml_payload
    from changes.digitone_backend import build_digitone_syx_from_events_yaml

    compiled = compile_song_for_ui(song, settings)
    plan = compile_timeline_to_digitone_plan(compiled.timeline, compiled.target_profile)
    events_payload = digitone_compile_plan_to_events_yaml_payload(
        plan,
        track_default_velocity=compiled.target_profile.track_default_velocity,
        pattern_change_policy=settings.pattern_change_policy,
        lpc_lead_enabled=settings.lpc_lead_layer_enabled,
    )
    yaml_fd, yaml_path = tempfile.mkstemp(suffix=".yaml")
    syx_fd, syx_path = tempfile.mkstemp(suffix=".syx")
    try:
        os.close(syx_fd)
        with os.fdopen(yaml_fd, "w") as f:
            yaml.safe_dump(events_payload, f, allow_unicode=False, sort_keys=False)
        build_digitone_syx_from_events_yaml(yaml_path, syx_path)
        return Path(syx_path).read_bytes()
    finally:
        for p in (yaml_path, syx_path):
            try:
                os.unlink(p)
            except OSError:
                pass


def song_to_syx_bytes_linear_split(
    song: SongModel,
    settings: AppSettings,
) -> list[tuple[str, bytes]]:
    """Compile Linear mode through section-boundary Auto Split."""
    import yaml

    from changes.digitone.linear_split_planner import compile_timeline_to_digitone_linear_split_plan
    from changes.digitone_backend import build_digitone_syx_from_events_yaml
    from changes.exporters.digitone_events import digitone_pattern_segment_to_events_yaml_payload

    compiled = compile_song_for_ui(song, settings)
    split_plan = compile_timeline_to_digitone_linear_split_plan(
        compiled.song,
        compiled.timeline,
        compiled.target_profile,
    )

    result: list[tuple[str, bytes]] = []
    for segment in split_plan.patterns:
        payload = digitone_pattern_segment_to_events_yaml_payload(
            segment,
            split_plan.timing,
            track_default_velocity=compiled.target_profile.track_default_velocity,
            pattern_change_policy=settings.pattern_change_policy,
            lpc_lead_enabled=settings.lpc_lead_layer_enabled,
        )
        yaml_fd, yaml_path = tempfile.mkstemp(suffix=".yaml")
        syx_fd, syx_path = tempfile.mkstemp(suffix=".syx")
        try:
            os.close(syx_fd)
            with os.fdopen(yaml_fd, "w") as f:
                yaml.safe_dump(payload, f, allow_unicode=False, sort_keys=False)
            build_digitone_syx_from_events_yaml(yaml_path, syx_path)
            result.append((segment.pattern_name, Path(syx_path).read_bytes()))
        finally:
            for p in (yaml_path, syx_path):
                try:
                    os.unlink(p)
                except OSError:
                    pass

    return result


def song_to_syx_bytes_bundle(
    song: SongModel,
    settings: AppSettings,
) -> list[tuple[str, bytes]]:
    """Compile song using the bundle planner and return one (pattern_name, syx_bytes) per segment.

    Uses the shared timing plan so voice-leading is continuous across sections.
    Raises on planning errors; individual segment SysEx failures propagate as well.
    """
    import os
    import tempfile

    import yaml

    from changes.digitone.bundle_planner import compile_timeline_to_digitone_bundle_plan
    from changes.digitone_backend import build_digitone_syx_from_events_yaml
    from changes.exporters.digitone_events import digitone_pattern_segment_to_events_yaml_payload

    compiled = compile_song_for_ui(song, settings)
    bundle_plan = compile_timeline_to_digitone_bundle_plan(
        compiled.song, compiled.timeline, compiled.target_profile
    )

    result: list[tuple[str, bytes]] = []
    for segment in bundle_plan.patterns:
        payload = digitone_pattern_segment_to_events_yaml_payload(
            segment,
            bundle_plan.timing,
            track_default_velocity=compiled.target_profile.track_default_velocity,
            pattern_change_policy=settings.pattern_change_policy,
            lpc_lead_enabled=settings.lpc_lead_layer_enabled,
        )
        yaml_fd, yaml_path = tempfile.mkstemp(suffix=".yaml")
        syx_fd, syx_path = tempfile.mkstemp(suffix=".syx")
        try:
            os.close(syx_fd)
            with os.fdopen(yaml_fd, "w") as f:
                yaml.safe_dump(payload, f, allow_unicode=False, sort_keys=False)
            build_digitone_syx_from_events_yaml(yaml_path, syx_path)
            result.append((segment.pattern_name, Path(syx_path).read_bytes()))
        finally:
            for p in (yaml_path, syx_path):
                try:
                    os.unlink(p)
                except OSError:
                    pass

    return result


def song_to_preview_events(song: SongModel, settings: AppSettings) -> list[RenderedNoteEvent]:
    """Return the rendered timeline events for preview/debugging."""
    compiled = compile_song_for_ui(song, settings)
    return compiled.timeline.events
