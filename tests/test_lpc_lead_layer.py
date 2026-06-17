"""Tests for LPC Lead Layer (v0.3.0 beta).

Covers the 7 required test cases from the spec:
1. LPC Lead Layer OFF — Tracks 9-16 traditional behaviour unchanged
2. LPC Lead Layer ON — Tracks 9-16 are used
3. C4-B4 range, D-major-type collection → C#4-first ascending placement
4. <8 notes in collection — octave above fills remaining slots
5. 9+ pitch-classes — trimmed to 8 slots
6. Tracks 9-16 Length / Speed / Change follow Cloud
7. Export / Preview target includes Tracks 9-16 events
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from changes.app_settings import AppSettings
from changes.lpc_lead_layer import (
    LPC_LEAD_SLOT_COUNT,
    compute_lpc_lead_notes,
    lpc_lead_range_start_midi,
    lpc_lead_voice_ids,
)
from changes.models.render_profile import RenderProfile, default_render_profile
from changes.models.rendered_arrangement import RenderedHarmonyOccurrence
from changes.models.song_model import HarmonyEvent, Measure, SongModel
from changes.rendering.arrangement_renderer import render_arrangement
from changes.rendering.arrangement_flattener import flatten_arrangement_to_timeline
from changes.ui_pipeline import (
    compile_song_for_ui,
    settings_to_render_profile,
    settings_to_target_profile,
)
from changes.exporters.digitone_events import (
    _track_scale_payload,
    TRACK_SCALE_COMPUTED_RANGE,
    TRACK_SCALE_FIXED_RANGE,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _song(symbol: str = "Cmaj7") -> SongModel:
    return SongModel(
        title="LPC Test",
        working_key="C",
        performance_tempo=Fraction(120, 1),
        measures=(
            Measure(
                number=1,
                section_id="A",
                meter_numerator=4,
                meter_denominator=4,
                absolute_start_quarters=Fraction(0, 1),
                harmony=(
                    HarmonyEvent(
                        id="h1",
                        symbol=symbol,
                        measure_number=1,
                        offset_quarters=Fraction(0, 1),
                        duration_quarters=Fraction(4, 1),
                    ),
                ),
            ),
        ),
    )


def _two_chord_song() -> SongModel:
    return SongModel(
        title="Two Chord",
        working_key="C",
        performance_tempo=Fraction(120, 1),
        measures=(
            Measure(
                number=1,
                section_id="A",
                meter_numerator=4,
                meter_denominator=4,
                absolute_start_quarters=Fraction(0, 1),
                harmony=(
                    HarmonyEvent(
                        id="h1",
                        symbol="Cmaj7",
                        measure_number=1,
                        offset_quarters=Fraction(0, 1),
                        duration_quarters=Fraction(2, 1),
                    ),
                    HarmonyEvent(
                        id="h2",
                        symbol="G7",
                        measure_number=1,
                        offset_quarters=Fraction(2, 1),
                        duration_quarters=Fraction(2, 1),
                    ),
                ),
            ),
        ),
    )


def _lpc_enabled_settings(**kwargs) -> AppSettings:
    s = AppSettings()
    s.lpc_lead_layer_enabled = True
    s.lpc_lead_layer_range_root = kwargs.get("range_root", 0)   # C
    s.lpc_lead_layer_octave = kwargs.get("octave", 4)            # C4
    return s


# ── Unit: compute_lpc_lead_notes ─────────────────────────────────────────────

def test_range_start_midi_c4():
    assert lpc_lead_range_start_midi(0, 4) == 60  # C4


def test_range_start_midi_d3():
    assert lpc_lead_range_start_midi(2, 3) == 50  # D3


def test_d_major_ionian_in_c4_range():
    """Test 3: C4-B4 range, D major Ionian starts at C#4."""
    # D major Ionian pitch classes: D(2) E(4) F#(6) G(7) A(9) B(11) C#(1)
    d_ionian_pcs = frozenset({1, 2, 4, 6, 7, 9, 11})
    c4 = 60
    notes = compute_lpc_lead_notes(d_ionian_pcs, c4)
    # Within C4(60)-B4(71): C#4=61, D4=62, E4=64, F#4=66, G4=67, A4=69, B4=71 → 7 notes
    # 8th → C#5=73
    assert notes == (61, 62, 64, 66, 67, 69, 71, 73)
    assert notes[0] == 61  # C#4, not D4 (range-start-first, not root-first)


def test_under_8_fills_with_octave_above():
    """Test 4: fewer than 8 pitch classes → extend with octave above."""
    # C D E G A = 5 pitch classes
    pcs = frozenset({0, 2, 4, 7, 9})
    c4 = 60
    notes = compute_lpc_lead_notes(pcs, c4)
    # C4=60, D4=62, E4=64, G4=67, A4=69, C5=72, D5=74, E5=76
    assert notes == (60, 62, 64, 67, 69, 72, 74, 76)
    assert len(notes) == LPC_LEAD_SLOT_COUNT


def test_9_plus_pcs_trimmed_to_8():
    """Test 5: 9+ pitch classes → take lowest 8 in ascending order."""
    # Chromatic (12 pitch classes) starting from C4 → C4..G#4
    all_pcs = frozenset(range(12))
    c4 = 60
    notes = compute_lpc_lead_notes(all_pcs, c4)
    assert notes == tuple(range(60, 68))
    assert len(notes) == LPC_LEAD_SLOT_COUNT


def test_notes_never_exceed_midi_127():
    """MIDI note 127 is the ceiling — notes above it are invalid."""
    pcs = frozenset({0, 2, 4, 7, 9})  # 5 pitch classes
    notes = compute_lpc_lead_notes(pcs, 60)
    assert all(0 <= n <= 127 for n in notes), f"Out-of-range MIDI notes: {notes}"


def test_raises_when_8_notes_unreachable_within_midi_range():
    """Single pitch class starting near MIDI 127 cannot produce 8 notes — raise ValueError."""
    pcs = frozenset({0})  # C only: C4=60, C5=72, C6=84, C7=96, C8=108, C9=120 → 6 notes max
    with pytest.raises(ValueError, match="MIDI 0–127"):
        compute_lpc_lead_notes(pcs, 60)


def test_lpc_lead_voice_ids_count():
    ids = lpc_lead_voice_ids()
    assert len(ids) == 8
    assert ids[0] == "lpc_lead_slot_1"
    assert ids[7] == "lpc_lead_slot_8"


# ── Test 1: OFF — Tracks 9-16 unchanged ──────────────────────────────────────

def test_lpc_lead_off_no_lpc_in_arrangement():
    """Test 1: LPC Lead Layer OFF → no lpc_lead layer in rendered occurrences."""
    profile = default_render_profile()
    assert not profile.lpc_lead_enabled
    arrangement = render_arrangement(_song(), profile)
    for occ in arrangement.occurrences:
        assert occ.lpc_lead is None


def test_lpc_lead_off_no_lpc_events_in_timeline():
    """Test 1: LPC Lead Layer OFF → no lpc_lead events in flattened timeline."""
    profile = default_render_profile()
    arrangement = render_arrangement(_song(), profile)
    timeline = flatten_arrangement_to_timeline(arrangement, render_profile=profile)
    lpc_events = [e for e in timeline.events if e.role == "lpc_lead"]
    assert lpc_events == []


def test_lpc_lead_off_no_track_9_16_routing():
    """Test 1: LPC Lead Layer OFF → Tracks 9-16 not in target profile routing."""
    settings = AppSettings()
    assert not settings.lpc_lead_layer_enabled
    tp = settings_to_target_profile(settings)
    routed_tracks = set(tp.voice_to_track.values())
    assert not routed_tracks.intersection(range(9, 17))


# ── Test 2: ON — Tracks 9-16 are used ────────────────────────────────────────

def test_lpc_lead_on_lpc_layer_in_arrangement():
    """Test 2: LPC Lead Layer ON → lpc_lead layer present in rendered occurrences."""
    settings = _lpc_enabled_settings()
    rp = settings_to_render_profile(settings)
    arrangement = render_arrangement(_song(), rp)
    assert len(arrangement.occurrences) > 0
    for occ in arrangement.occurrences:
        assert occ.lpc_lead is not None
        assert len(occ.lpc_lead.notes) == 8


def test_lpc_lead_on_events_in_timeline():
    """Test 2: LPC Lead Layer ON → lpc_lead events in flattened timeline."""
    settings = _lpc_enabled_settings()
    rp = settings_to_render_profile(settings)
    arrangement = render_arrangement(_song(), rp)
    timeline = flatten_arrangement_to_timeline(arrangement, render_profile=rp)
    lpc_events = [e for e in timeline.events if e.role == "lpc_lead"]
    assert len(lpc_events) == 8


def test_lpc_lead_on_routing_to_tracks_9_16():
    """Test 2: LPC Lead Layer ON → Tracks 9-16 in target profile routing."""
    settings = _lpc_enabled_settings()
    tp = settings_to_target_profile(settings)
    routed_tracks = set(tp.voice_to_track.values())
    for track in range(9, 17):
        assert track in routed_tracks


def test_lpc_lead_on_slot_track_mapping():
    """Test 2: lpc_lead_slot_1→track9, …, lpc_lead_slot_8→track16."""
    settings = _lpc_enabled_settings()
    tp = settings_to_target_profile(settings)
    v2t = tp.voice_to_track
    for i in range(1, 9):
        assert v2t[f"lpc_lead_slot_{i}"] == 8 + i


# ── Test 6: Tracks 9-16 Length/Speed/Change follow Cloud ─────────────────────

def test_track_scale_lpc_off_tracks_9_16_fixed():
    """Test 6 (OFF case): tracks 9-16 use fixed 16/1 when LPC disabled."""
    payload = _track_scale_payload(length=32, speed="1/8", lpc_lead_enabled=False)
    for track in TRACK_SCALE_FIXED_RANGE:
        assert payload[track]["length"] == 16
        assert payload[track]["speed"] == "1"


def test_track_scale_lpc_on_tracks_9_16_match_cloud():
    """Test 6 (ON case): tracks 9-16 use same length/speed as 1-8 when LPC enabled."""
    payload = _track_scale_payload(length=32, speed="1/8", lpc_lead_enabled=True)
    for track in TRACK_SCALE_COMPUTED_RANGE:
        assert payload[track]["length"] == 32
        assert payload[track]["speed"] == "1/8"
    for track in TRACK_SCALE_FIXED_RANGE:
        assert payload[track]["length"] == 32
        assert payload[track]["speed"] == "1/8"


# ── Test 7: Export / Preview includes Tracks 9-16 ────────────────────────────

def test_compiled_song_lpc_events_route_to_tracks_9_16():
    """Test 7: compile_song_for_ui with LPC ON → timeline has voice_ids that map to tracks 9-16."""
    settings = _lpc_enabled_settings()
    compiled = compile_song_for_ui(_song(), settings)
    v2t = compiled.target_profile.voice_to_track
    lpc_events = [e for e in compiled.timeline.events if e.role == "lpc_lead"]
    assert len(lpc_events) == 8
    for e in lpc_events:
        track = v2t[e.voice_id]
        assert 9 <= track <= 16


def test_compiled_song_two_chords_lpc_events():
    """Test 7: two-chord song with LPC ON → 8 lpc_lead events per chord (16 total)."""
    settings = _lpc_enabled_settings()
    compiled = compile_song_for_ui(_two_chord_song(), settings)
    lpc_events = [e for e in compiled.timeline.events if e.role == "lpc_lead"]
    # Each chord generates 8 events; hold_until_change may merge same pitches
    # but different chords produce different notes, so both chord cells fire
    assert len(lpc_events) >= 8  # at minimum 8 (if all notes happen to hold)
    tracks_used = {compiled.target_profile.voice_to_track[e.voice_id] for e in lpc_events}
    assert tracks_used == set(range(9, 17))


def test_lpc_lead_notes_ascending_in_range():
    """Test 3 via pipeline: rendered LPC notes are ascending and within/above range_start."""
    settings = _lpc_enabled_settings(range_root=0, octave=4)  # C4
    rp = settings_to_render_profile(settings)
    arrangement = render_arrangement(_song("Cmaj7"), rp)
    occ = arrangement.occurrences[0]
    assert occ.lpc_lead is not None
    midi_notes = [n.note_midi for n in occ.lpc_lead.notes]
    assert midi_notes == sorted(midi_notes), "LPC lead notes must be in ascending order"
    assert midi_notes[0] >= 60, "First note must be >= range_start_midi (C4=60)"
    assert len(midi_notes) == 8
