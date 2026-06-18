"""Emiuet Session song payload and library index export helpers."""

from __future__ import annotations

from dataclasses import dataclass

from .emiuet_timeline import (
    CompiledStepInput,
    build_emiuet_compiled_timeline,
    clock_song_timeline_from_chords,
    manual_timeline_from_chords,
)

SONG_PAYLOAD_SCHEMA_NAME = "emnyeca.emiuet_session.song_payload"
LIBRARY_INDEX_SCHEMA_NAME = "emnyeca.emiuet_session.library_index"
SCHEMA_VERSION = 1
ADVANCE_MODE_VALUES = frozenset({"clock_song", "manual", "device_step"})
TIMELINE_BASIS_VALUES = frozenset({"original_song", "segment_map", "digitone_step"})
RUNTIME_TRANSPOSE_POLICY_VALUES = frozenset({"allowed", "warn", "locked"})


def _validate_value(name: str, value: str, allowed: frozenset[str]) -> None:
    if value not in allowed:
        raise ValueError(f"unsupported {name}: {value!r}")


@dataclass(frozen=True)
class EmiuetSessionTimeline:
    id: str
    advance_mode: str
    timeline_basis: str
    compiled_timeline: dict
    runtime_transpose_policy: str | None = None
    device: str | None = None

    def __post_init__(self) -> None:
        _validate_value("advance_mode", self.advance_mode, ADVANCE_MODE_VALUES)
        _validate_value("timeline_basis", self.timeline_basis, TIMELINE_BASIS_VALUES)
        if self.runtime_transpose_policy is not None:
            _validate_value(
                "runtime_transpose_policy",
                self.runtime_transpose_policy,
                RUNTIME_TRANSPOSE_POLICY_VALUES,
            )
        compiled_basis = self.compiled_timeline.get("timeline_basis")
        if compiled_basis != self.timeline_basis:
            raise ValueError(
                "timeline descriptor basis must match compiled_timeline timeline_basis: "
                f"{self.timeline_basis!r} != {compiled_basis!r}"
            )

    def to_dict(self) -> dict:
        policy = self.runtime_transpose_policy or default_runtime_transpose_policy(
            self.advance_mode, self.timeline_basis
        )
        out = {
            "id": self.id,
            "advance_mode": self.advance_mode,
            "timeline_basis": self.timeline_basis,
            "runtime_transpose_policy": policy,
            "compiled_timeline": self.compiled_timeline,
        }
        if self.device is not None:
            out["device"] = self.device
        return out


@dataclass(frozen=True)
class EmiuetSessionSong:
    song_id: str
    title: str
    default_key: str
    default_tempo: float
    meter: str
    timelines: tuple[EmiuetSessionTimeline, ...]
    default_timeline_id: str | None = None

    def __post_init__(self) -> None:
        if not self.timelines:
            raise ValueError("EmiuetSessionSong must have at least one timeline")
        ids = [timeline.id for timeline in self.timelines]
        if len(ids) != len(set(ids)):
            raise ValueError("EmiuetSessionSong timeline ids must be unique")
        if self.default_timeline_id is not None and self.default_timeline_id not in set(ids):
            raise ValueError("default_timeline_id must reference an existing timeline")

    def to_payload_dict(self) -> dict:
        out = {
            "schema": SONG_PAYLOAD_SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "song_id": self.song_id,
            "title": self.title,
            "default_key": self.default_key,
            "default_tempo": self.default_tempo,
            "meter": self.meter,
            "timelines": [timeline.to_dict() for timeline in self.timelines],
        }
        if self.default_timeline_id is not None:
            out["default_timeline_id"] = self.default_timeline_id
        return out


@dataclass(frozen=True)
class EmiuetSessionLibraryEntry:
    song_id: str
    title: str
    default_key: str
    default_tempo: float
    meter: str
    available_timelines: tuple[str, ...]
    favorite: bool = False
    recent_order: int | None = None
    payload_ref: str | None = None

    def to_dict(self) -> dict:
        out = {
            "song_id": self.song_id,
            "title": self.title,
            "default_key": self.default_key,
            "default_tempo": self.default_tempo,
            "meter": self.meter,
            "available_timelines": list(self.available_timelines),
            "favorite": self.favorite,
        }
        if self.recent_order is not None:
            out["recent_order"] = self.recent_order
        if self.payload_ref is not None:
            out["payload_ref"] = self.payload_ref
        return out


def default_runtime_transpose_policy(advance_mode: str, timeline_basis: str) -> str:
    _validate_value("advance_mode", advance_mode, ADVANCE_MODE_VALUES)
    _validate_value("timeline_basis", timeline_basis, TIMELINE_BASIS_VALUES)
    if advance_mode == "device_step" or timeline_basis == "digitone_step":
        return "locked"
    return "allowed"


def build_song_payload(song: EmiuetSessionSong) -> dict:
    return song.to_payload_dict()


def build_emiuet_session_song_payload(
    *,
    song_id: str,
    title: str,
    default_key: str,
    default_tempo: float,
    meter: str,
    timelines: list[EmiuetSessionTimeline] | tuple[EmiuetSessionTimeline, ...],
    default_timeline_id: str | None = None,
) -> dict:
    return build_song_payload(
        EmiuetSessionSong(
            song_id=song_id,
            title=title,
            default_key=default_key,
            default_tempo=default_tempo,
            meter=meter,
            timelines=tuple(timelines),
            default_timeline_id=default_timeline_id,
        )
    )


def build_library_index(entries: list[EmiuetSessionLibraryEntry]) -> dict:
    return {
        "schema": LIBRARY_INDEX_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "songs": [entry.to_dict() for entry in entries],
    }


def library_entry_from_song(
    song: EmiuetSessionSong,
    *,
    payload_ref: str | None = None,
    favorite: bool = False,
    recent_order: int | None = None,
) -> EmiuetSessionLibraryEntry:
    return EmiuetSessionLibraryEntry(
        song_id=song.song_id,
        title=song.title,
        default_key=song.default_key,
        default_tempo=song.default_tempo,
        meter=song.meter,
        available_timelines=tuple(timeline.id for timeline in song.timelines),
        favorite=favorite,
        recent_order=recent_order,
        payload_ref=payload_ref,
    )


def build_contrast_demo_session_artifacts(
    *,
    payload_ref: str = "songs/contrast_demo.song.json",
) -> dict:
    """Build developer fixtures for the Dm7/G7/Cmaj7/A7 Session payload."""
    chords = ["Dm7", "G7", "Cmaj7", "A7"]
    clock = clock_song_timeline_from_chords(chords, ppqn=24, tempo=120.0, meter="4/4")
    manual = manual_timeline_from_chords(chords, ppqn=24, tempo=120.0, meter="4/4")
    digitone = build_emiuet_compiled_timeline(
        [
            CompiledStepInput(
                id=f"step_{index:03d}",
                start_tick=index * 24,
                end_tick=(index + 1) * 24,
                chord=chord,
                source_step_index=index,
            )
            for index, chord in enumerate(chords)
        ],
        ppqn=24,
        original_tempo=120.0,
        digitone_tempo=60.0,
        meter="4/4",
    )
    song = EmiuetSessionSong(
        song_id="contrast_demo",
        title="Contrast Demo",
        default_key="C",
        default_tempo=120.0,
        meter="4/4",
        timelines=(
            EmiuetSessionTimeline("clock_song", "clock_song", "original_song", clock),
            EmiuetSessionTimeline("manual", "manual", "segment_map", manual),
            EmiuetSessionTimeline(
                "digitone_ii_a01",
                "device_step",
                "digitone_step",
                digitone,
                device="digitone_ii",
            ),
        ),
        default_timeline_id="clock_song",
    )
    payload = build_song_payload(song)
    library_index = build_library_index([library_entry_from_song(song, payload_ref=payload_ref)])
    return {
        "compiled_timeline": digitone,
        "song_payload": payload,
        "library_index": library_index,
    }
