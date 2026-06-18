"""Emiuet Session library and advance architecture export tests."""

from changes.exporters.emiuet_session_library import (
    EmiuetSessionLibraryEntry,
    EmiuetSessionSong,
    EmiuetSessionTimeline,
    build_library_index,
    build_song_payload,
    default_runtime_transpose_policy,
    library_entry_from_song,
)
from changes.exporters.emiuet_timeline import (
    CompiledStepInput,
    build_emiuet_compiled_timeline,
    clock_song_timeline_from_chords,
)


def test_song_payload_can_contain_multiple_timelines():
    clock = clock_song_timeline_from_chords(["Dm7", "G7", "Cmaj7", "A7"])
    digitone = build_emiuet_compiled_timeline(
        [CompiledStepInput("step_000", 0, 24, "Dm7")]
    )
    song = EmiuetSessionSong(
        song_id="contrast_demo",
        title="Contrast Demo",
        default_key="C",
        default_tempo=120.0,
        meter="4/4",
        timelines=(
            EmiuetSessionTimeline("clock_song", "clock_song", "original_song", clock),
            EmiuetSessionTimeline(
                "digitone_ii_a01",
                "device_step",
                "digitone_step",
                digitone,
                device="digitone_ii",
            ),
        ),
    )
    payload = build_song_payload(song)
    assert payload["schema"] == "emnyeca.emiuet_session.song_payload"
    assert [timeline["id"] for timeline in payload["timelines"]] == [
        "clock_song",
        "digitone_ii_a01",
    ]
    assert payload["timelines"][0]["runtime_transpose_policy"] == "allowed"
    assert payload["timelines"][1]["runtime_transpose_policy"] == "locked"
    assert payload["timelines"][1]["compiled_timeline"]["timeline_basis"] == "digitone_step"


def test_clock_song_original_song_timeline_uses_bar_ticks():
    timeline = clock_song_timeline_from_chords(["Dm7", "G7", "Cmaj7", "A7"], ppqn=24)
    assert timeline["timeline_basis"] == "original_song"
    assert [(step["start_tick"], step["end_tick"]) for step in timeline["steps"]] == [
        (0, 96),
        (96, 192),
        (192, 288),
        (288, 384),
    ]
    assert timeline["clock"]["meter"] == "4/4"


def test_existing_digitone_step_export_stays_default():
    timeline = build_emiuet_compiled_timeline(
        [CompiledStepInput("step_000", 0, 24, "Dm7")]
    )
    assert timeline["timeline_basis"] == "digitone_step"
    assert timeline["steps"][0]["contexts"]["progression"]["display"] == "Dm7"
    assert "contrast" in timeline["steps"][0]["contexts"]


def test_runtime_transpose_policy_defaults():
    assert default_runtime_transpose_policy("clock_song", "original_song") == "allowed"
    assert default_runtime_transpose_policy("manual", "segment_map") == "allowed"
    assert default_runtime_transpose_policy("device_step", "digitone_step") == "locked"


def test_library_index_represents_many_songs_without_payloads():
    entries = [
        EmiuetSessionLibraryEntry(
            song_id=f"song_{i:04d}",
            title=f"Song {i}",
            default_key="C",
            default_tempo=120.0,
            meter="4/4",
            available_timelines=("clock_song",),
            payload_ref=f"songs/song_{i:04d}.json",
        )
        for i in range(2000)
    ]
    index = build_library_index(entries)
    assert index["schema"] == "emnyeca.emiuet_session.library_index"
    assert len(index["songs"]) == 2000
    assert index["songs"][1999]["payload_ref"] == "songs/song_1999.json"
    assert "compiled_timeline" not in index["songs"][1999]


def test_library_entry_from_song_uses_timeline_ids():
    song = EmiuetSessionSong(
        song_id="contrast_demo",
        title="Contrast Demo",
        default_key="C",
        default_tempo=120.0,
        meter="4/4",
        timelines=(
            EmiuetSessionTimeline(
                "clock_song",
                "clock_song",
                "original_song",
                clock_song_timeline_from_chords(["Dm7"]),
            ),
        ),
    )
    entry = library_entry_from_song(song, payload_ref="songs/contrast_demo.json")
    assert entry.available_timelines == ("clock_song",)
    assert entry.payload_ref == "songs/contrast_demo.json"

