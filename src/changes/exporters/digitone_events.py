"""Export DigitoneCompilePlan as toolkit-compatible events YAML payload."""

from __future__ import annotations

from collections.abc import Iterable
from fractions import Fraction

from changes.models.digitone_bundle_plan import DigitonePatternSegment, DigitoneSongTimingPlan
from changes.models.digitone_compile_plan import DigitoneCompilePlan


TRACK_SCALE_COMPUTED_RANGE = range(1, 9)
TRACK_SCALE_FIXED_RANGE = range(9, 17)
TRACK_SCALE_FIXED_LENGTH = 16
TRACK_SCALE_FIXED_SPEED = "1"
PATTERN_CHANGE_POLICY_AUTO_SONG_MODE = "auto_song_mode"
PATTERN_CHANGE_POLICY_OFF = "off"


def _compiled_events_to_yaml_rows(events: Iterable) -> list[dict]:
    rows: list[dict] = []
    for e in sorted(events, key=lambda x: (x.track, x.step, x.source_event_id)):
        rows.append(
            {
                "step": int(e.step),
                "track": int(e.track),
                "note": str(e.note),
                "velocity": e.velocity,
                "length_code": f"0x{int(e.length_code):02X}",
            }
        )
    return rows


def _track_defaults_payload(track_default_velocity: dict[int, int] | None) -> dict | None:
    if not track_default_velocity:
        return None
    ordered = {int(track): int(velocity) for track, velocity in sorted(track_default_velocity.items())}
    return {"velocity": ordered}


def _track_scale_payload(
    *, length: int, speed: str, lpc_lead_enabled: bool = False
) -> dict[int, dict[str, int | str]]:
    payload = {
        track: {"length": int(length), "speed": str(speed)}
        for track in TRACK_SCALE_COMPUTED_RANGE
    }
    if lpc_lead_enabled:
        # Tracks 9–16 follow the same length/speed as Tracks 1–8 (Cloud timing)
        payload.update(
            {
                track: {"length": int(length), "speed": str(speed)}
                for track in TRACK_SCALE_FIXED_RANGE
            }
        )
    else:
        payload.update(
            {
                track: {"length": TRACK_SCALE_FIXED_LENGTH, "speed": TRACK_SCALE_FIXED_SPEED}
                for track in TRACK_SCALE_FIXED_RANGE
            }
        )
    return payload


def pattern_change_value(*, length: int, speed_ratio: Fraction, policy: str) -> int | str:
    normalized_policy = str(policy or PATTERN_CHANGE_POLICY_AUTO_SONG_MODE)
    if normalized_policy == PATTERN_CHANGE_POLICY_OFF:
        return "OFF"
    if normalized_policy != PATTERN_CHANGE_POLICY_AUTO_SONG_MODE:
        raise ValueError(f"Unknown pattern_change_policy: {policy!r}")

    change = Fraction(int(length), 1) / Fraction(speed_ratio)
    if change.denominator != 1:
        raise ValueError("Cannot derive integer pattern CHANGE from length/speed")
    if change < 2 or change > 1024:
        raise ValueError("Cannot derive supported pattern CHANGE from length/speed (expected 2..1024)")
    return int(change)


def pattern_change_basis_payload(*, length: int, speed: str) -> dict[str, int | str]:
    return {
        "generated_tracks": "1..8",
        "length": int(length),
        "speed": str(speed),
    }


def digitone_compile_plan_to_events_yaml_payload(
    plan: DigitoneCompilePlan,
    *,
    track_default_velocity: dict[int, int] | None = None,
    pattern_change_policy: str = PATTERN_CHANGE_POLICY_AUTO_SONG_MODE,
    lpc_lead_enabled: bool = False,
) -> dict:
    pattern_change = pattern_change_value(
        length=plan.total_steps,
        speed_ratio=plan.speed_ratio,
        policy=pattern_change_policy,
    )
    payload: dict = {
        "version": 1,
        "device": "digitone2",
        "name": plan.pattern_name,
        "pattern": {
            "mode": "per-track",
            "tempo": float(plan.device_tempo),
            "change": pattern_change,
            "reset": "INF",
        },
        "track_scale": _track_scale_payload(
            length=plan.total_steps, speed=plan.speed, lpc_lead_enabled=lpc_lead_enabled
        ),
    }
    track_defaults = _track_defaults_payload(track_default_velocity)
    if track_defaults is not None:
        payload["track_defaults"] = track_defaults
    payload["events"] = _compiled_events_to_yaml_rows(plan.events)
    return payload


def digitone_pattern_segment_to_events_yaml_payload(
    segment: DigitonePatternSegment,
    timing: DigitoneSongTimingPlan,
    *,
    track_default_velocity: dict[int, int] | None = None,
    pattern_change_policy: str = PATTERN_CHANGE_POLICY_AUTO_SONG_MODE,
    lpc_lead_enabled: bool = False,
) -> dict:
    pattern_change = pattern_change_value(
        length=segment.total_steps,
        speed_ratio=timing.speed_ratio,
        policy=pattern_change_policy,
    )
    payload: dict = {
        "version": 1,
        "device": "digitone2",
        "name": segment.pattern_name,
        "pattern": {
            "mode": "per-track",
            "tempo": float(timing.device_tempo),
            "change": pattern_change,
            "reset": "INF",
        },
        "track_scale": _track_scale_payload(
            length=segment.total_steps, speed=timing.speed, lpc_lead_enabled=lpc_lead_enabled
        ),
    }
    track_defaults = _track_defaults_payload(track_default_velocity)
    if track_defaults is not None:
        payload["track_defaults"] = track_defaults
    payload["events"] = _compiled_events_to_yaml_rows(segment.events)
    return payload
