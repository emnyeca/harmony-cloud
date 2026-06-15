"""Render profile model and defaults."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RenderProfile:
    name: str
    voices: int
    voice_leading_strategy: str
    bass_enabled: bool
    bass_strategy: str
    cloud_trigger_policy: str
    bass_trigger_policy: str
    chord_trigger_policy: str
    # Cloud center/spread control (replaces cloud_min_midi/cloud_max_midi for voice leading)
    cloud_center_midi: int
    cloud_spread_min: int
    cloud_spread_max: int
    cloud_average_tolerance: int
    # Cloud range fields kept for backwards compatibility but not used in voice leading
    cloud_min_midi: int
    cloud_max_midi: int
    # Chord and Bass ranges (still used)
    chord_min_midi: int
    chord_max_midi: int
    bass_min_midi: int
    bass_max_midi: int
    # LPC Lead Layer (v0.3.0 beta)
    lpc_lead_enabled: bool = False
    lpc_lead_range_start_midi: int = 60  # C4


def default_render_profile() -> RenderProfile:
    return RenderProfile(
        name="default_jazz_cloud",
        voices=6,
        voice_leading_strategy="minimum_motion",
        bass_enabled=True,
        bass_strategy="slash_or_root_in_window",
        cloud_trigger_policy="hold_until_change",
        bass_trigger_policy="hold_until_change",
        chord_trigger_policy="retrigger",
        cloud_center_midi=60,
        cloud_spread_min=14,
        cloud_spread_max=16,
        cloud_average_tolerance=2,
        cloud_min_midi=51,
        cloud_max_midi=69,
        chord_min_midi=48,
        chord_max_midi=72,
        bass_min_midi=36,
        bass_max_midi=47,
        lpc_lead_enabled=False,
        lpc_lead_range_start_midi=60,
    )


def render_profile_to_dict(profile: RenderProfile) -> dict:
    return {
        "version": 1,
        "type": "render_profile",
        "name": profile.name,
        "voicing": {"voices": profile.voices},
        "voice_leading": {
            "strategy": profile.voice_leading_strategy,
            "cloud_center_midi": profile.cloud_center_midi,
            "cloud_spread_min": profile.cloud_spread_min,
            "cloud_spread_max": profile.cloud_spread_max,
            "cloud_average_tolerance": profile.cloud_average_tolerance,
            "cloud_min_midi": profile.cloud_min_midi,
            "cloud_max_midi": profile.cloud_max_midi,
            "chord_min_midi": profile.chord_min_midi,
            "chord_max_midi": profile.chord_max_midi,
        },
        "bass": {
            "enabled": profile.bass_enabled,
            "strategy": profile.bass_strategy,
            "bass_min_midi": profile.bass_min_midi,
            "bass_max_midi": profile.bass_max_midi,
        },
        "trigger": {
            "cloud": profile.cloud_trigger_policy,
            "bass": profile.bass_trigger_policy,
            "chord": profile.chord_trigger_policy,
        },
    }
