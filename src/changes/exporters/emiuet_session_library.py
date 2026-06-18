"""Emiuet Session song payload and library index export helpers."""

from __future__ import annotations

from dataclasses import dataclass

SONG_PAYLOAD_SCHEMA_NAME = "emnyeca.emiuet_session.song_payload"
LIBRARY_INDEX_SCHEMA_NAME = "emnyeca.emiuet_session.library_index"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EmiuetSessionTimeline:
    id: str
    advance_mode: str
    timeline_basis: str
    compiled_timeline: dict
    runtime_transpose_policy: str | None = None
    device: str | None = None

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

    def to_payload_dict(self) -> dict:
        return {
            "schema": SONG_PAYLOAD_SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "song_id": self.song_id,
            "title": self.title,
            "default_key": self.default_key,
            "default_tempo": self.default_tempo,
            "meter": self.meter,
            "timelines": [timeline.to_dict() for timeline in self.timelines],
        }


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
    if advance_mode == "device_step" or timeline_basis == "digitone_step":
        return "locked"
    return "allowed"


def build_song_payload(song: EmiuetSessionSong) -> dict:
    return song.to_payload_dict()


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

