"""High-level SongModel -> RenderedArrangement -> RenderedTimeline -> DigitoneCompilePlan pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from fractions import Fraction

import yaml

from changes.digitone_backend import build_digitone_syx_from_events_yaml
from changes.digitone.bundle_planner import compile_timeline_to_digitone_bundle_plan
from changes.digitone.planner import compile_timeline_to_digitone_plan
from changes.digitone.track8_chord_events import extract_track8_chord_events
from changes.exporters.emiuet_timeline import emiuet_timeline_from_chord_events
from changes.exporters.digitone_events import (
    digitone_compile_plan_to_events_yaml_payload,
    digitone_pattern_segment_to_events_yaml_payload,
)
from changes.models.digitone_bundle_plan import (
    DigitonePatternBundlePlan,
    digitone_pattern_bundle_plan_to_dict,
)
from changes.importers.compact_progression import compact_progression_to_song_model
from changes.importers.musicxml import ImportedSong, load_musicxml_song, imported_song_to_song_model
from changes.harmonic_context import (
    UnsupportedHarmonicContextError,
    chord_tone_pitch_classes,
    extract_output_chord_tone_set,
    resolve_scale_collection_with_retry_details,
)
from changes.no_chord import is_no_chord_symbol
from changes.note import semitone_to_pitch_class
from changes.models.digitone_compile_plan import DigitoneCompilePlan, digitone_compile_plan_to_dict
from changes.models.digitone_target_profile import DigitoneTargetProfile, default_digitone_target_profile
from changes.models.render_profile import RenderProfile, default_render_profile
from changes.models.rendered_timeline import RenderedTimeline, rendered_timeline_to_dict
from changes.models.song_model import SongModel, song_model_to_dict
from changes.rendering.arrangement_flattener import flatten_arrangement_to_timeline
from changes.rendering.arrangement_renderer import render_arrangement


class DigitonePipelineArtifacts(dict):
    pass


def _pc_names(pcs: tuple[int, ...] | frozenset[int]) -> list[str]:
    return [semitone_to_pitch_class(pc) for pc in pcs]


def build_musicxml_harmony_resolution_diagnostic(imported_song: ImportedSong) -> dict:
    progression = [event.symbol for bar in imported_song.bars for event in bar.events if not is_no_chord_symbol(event.symbol)]

    occurrences: list[dict] = []
    cursor = 0

    for bar in imported_song.bars:
        for event in bar.events:
            symbol = event.symbol
            if is_no_chord_symbol(symbol):
                continue
            try:
                resolved = resolve_scale_collection_with_retry_details(
                    progression,
                    cursor,
                    circular=True,
                    include_slash_bass=True,
                )
                output = extract_output_chord_tone_set(symbol, resolved.selected_collection)
            except UnsupportedHarmonicContextError as exc:
                local_current_only = chord_tone_pitch_classes(symbol, include_bass=True)
                pcs_names = ",".join(_pc_names(tuple(sorted(local_current_only))))
                raise UnsupportedHarmonicContextError(
                    "Unresolved MusicXML harmony context: "
                    f"measure={bar.source_measure_number} event={event.source_order_in_measure} "
                    f"symbol={symbol} local_pitch_collection=[{pcs_names}]"
                ) from exc

            occurrences.append(
                {
                    "global_event_index": cursor + 1,
                    "measure_number": bar.source_measure_number,
                    "event_order_in_measure": event.source_order_in_measure,
                    "canonical_chord_symbol": symbol,
                    "source": {
                        "raw_kind_value": event.raw_kind_value,
                        "raw_kind_text": event.raw_kind_text,
                        "raw_degrees": [
                            {
                                "value": d.value,
                                "alter": d.alter,
                                "degree_type": d.degree_type,
                            }
                            for d in event.raw_degrees
                        ],
                        "raw_root": event.raw_root,
                        "raw_bass": event.raw_bass,
                    },
                    "source_position_quarters": (
                        None
                        if event.source_position_quarters is None
                        else str(event.source_position_quarters)
                    ),
                    "hard_context_pitch_classes_used": _pc_names(
                        tuple(sorted(resolved.hard_context_pitch_classes_used))
                    ),
                    "color_hint_pitch_classes": _pc_names(
                        tuple(sorted(resolved.color_hint_pitch_classes))
                    ),
                    "color_hints_applied_to_constraint_set": resolved.color_hints_applied_to_constraint_set,
                    "final_local_pitch_collection_used_for_selection": _pc_names(
                        tuple(sorted(resolved.final_local_pitch_collection_used_for_selection))
                    ),
                    "local_pitch_collection": _pc_names(tuple(sorted(resolved.local_pitch_collection))),
                    "selected_collection_name": resolved.selected_collection.name,
                    "selected_collection_family": resolved.selected_collection.family,
                    "output_chord_tone_set": _pc_names(output),
                    "retry_level": resolved.retry_level,
                }
            )
            cursor += 1

    return {
        "title": imported_song.title,
        "source_software": imported_song.source_software,
        "source_musicxml_version": imported_song.source_musicxml_version,
        "occurrence_count": len(occurrences),
        "occurrences": occurrences,
    }


def _safe_ascii_slug(text: str, fallback: str = "UNTITLED") -> str:
    chars: list[str] = []
    prev_underscore = False
    for ch in text:
        code = ord(ch)
        if 0x61 <= code <= 0x7A:
            ch = chr(code - 0x20)
            code = ord(ch)

        is_ascii_alnum = (0x30 <= code <= 0x39) or (0x41 <= code <= 0x5A)
        if is_ascii_alnum:
            chars.append(ch)
            prev_underscore = False
            continue

        if not prev_underscore:
            chars.append("_")
            prev_underscore = True

    slug = "".join(chars).strip("_")
    return slug if slug else fallback


def _extract_explicit_pattern_name_overrides(payload: dict) -> dict[int, str]:
    direct = payload.get("digitone_pattern_name_overrides")
    nested = payload.get("digitone")
    nested_overrides = nested.get("pattern_name_overrides") if isinstance(nested, dict) else None
    raw = direct if direct is not None else nested_overrides

    if raw is None:
        return {}

    if isinstance(raw, list):
        return {idx: str(name) for idx, name in enumerate(raw, start=1) if name is not None}

    if isinstance(raw, dict):
        out: dict[int, str] = {}
        for key, value in raw.items():
            if value is None:
                continue
            try:
                index = int(key)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid pattern_name_overrides key: {key!r}") from exc
            if index < 1:
                raise ValueError(f"pattern_name_overrides index must be >= 1: {key!r}")
            out[index] = str(value)
        return out

    raise ValueError("pattern_name_overrides must be a list or mapping")


def compile_digitone_pipeline(
    payload: dict,
    render_profile: RenderProfile | None = None,
    target_profile: DigitoneTargetProfile | None = None,
    layers: str | list[str] | tuple[str, ...] | set[str] | None = None,
    pattern_change_policy: str = "auto_song_mode",
) -> tuple[SongModel, RenderedTimeline, DigitoneCompilePlan, dict]:
    rp = render_profile or default_render_profile()
    tp = target_profile or default_digitone_target_profile()

    song = compact_progression_to_song_model(payload)
    arrangement = render_arrangement(song, rp)
    timeline = flatten_arrangement_to_timeline(arrangement, layers=layers, render_profile=rp)
    plan = compile_timeline_to_digitone_plan(timeline, tp)
    events_payload = digitone_compile_plan_to_events_yaml_payload(
        plan,
        track_default_velocity=tp.track_default_velocity,
        pattern_change_policy=pattern_change_policy,
    )

    return song, timeline, plan, events_payload


def compile_emiuet_session_timeline(
    payload: dict,
    render_profile: RenderProfile | None = None,
    target_profile: DigitoneTargetProfile | None = None,
    layers: str | list[str] | tuple[str, ...] | set[str] | None = None,
    *,
    ppqn: int = 24,
    meter: str = "4/4",
) -> dict:
    """Emiuet Session 用 compiled timeline JSON を生成する（Syx と同一の compile path）。

    Syx 生成と同じ render→flatten→plan を辿り、その plan の device/performance tempo と、
    同じ arrangement から取った Track 8 chord events（Syx の chord track と同一ソース）を
    使って Digitone-step 基準の timeline を出す。tick は device tempo の MIDI Clock grid。
    """
    rp = render_profile or default_render_profile()
    tp = target_profile or default_digitone_target_profile()

    song = compact_progression_to_song_model(payload)
    arrangement = render_arrangement(song, rp)
    timeline = flatten_arrangement_to_timeline(arrangement, layers=layers, render_profile=rp)
    plan = compile_timeline_to_digitone_plan(timeline, tp)
    chord_events = extract_track8_chord_events(arrangement)

    return emiuet_timeline_from_chord_events(
        chord_events,
        performance_tempo=plan.performance_tempo,
        device_tempo=plan.device_tempo,
        ppqn=ppqn,
        meter=meter,
    )


def save_emiuet_session_timeline(
    output_dir: str | Path,
    timeline_dict: dict,
    filename: str = "emiuet_session_timeline.json",
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    path.write_text(json.dumps(timeline_dict, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def compile_digitone_bundle_pipeline(
    payload: dict,
    render_profile: RenderProfile | None = None,
    target_profile: DigitoneTargetProfile | None = None,
    layers: str | list[str] | tuple[str, ...] | set[str] | None = None,
) -> tuple[SongModel, RenderedTimeline, DigitonePatternBundlePlan]:
    """Compile song into bundle-oriented Digitone plan (section/capacity split aware)."""
    rp = render_profile or default_render_profile()
    tp = target_profile or default_digitone_target_profile()

    song = compact_progression_to_song_model(payload)
    arrangement = render_arrangement(song, rp)
    timeline = flatten_arrangement_to_timeline(arrangement, layers=layers, render_profile=rp)
    explicit_overrides = _extract_explicit_pattern_name_overrides(payload)
    bundle_plan = compile_timeline_to_digitone_bundle_plan(
        song,
        timeline,
        tp,
        explicit_pattern_name_overrides=explicit_overrides,
    )
    return song, timeline, bundle_plan


def compile_musicxml_digitone_bundle_pipeline(
    musicxml_path: str | Path,
    *,
    tempo: Fraction | int | str = 120,
    render_profile: RenderProfile | None = None,
    target_profile: DigitoneTargetProfile | None = None,
    layers: str | list[str] | tuple[str, ...] | set[str] | None = None,
) -> tuple[SongModel, RenderedTimeline, DigitonePatternBundlePlan, dict]:
    rp = render_profile or default_render_profile()
    tp = target_profile or default_digitone_target_profile()

    imported = load_musicxml_song(musicxml_path)
    diagnostics = build_musicxml_harmony_resolution_diagnostic(imported)
    song = imported_song_to_song_model(imported, tempo=tempo)
    arrangement = render_arrangement(song, rp)
    timeline = flatten_arrangement_to_timeline(arrangement, layers=layers, render_profile=rp)
    bundle_plan = compile_timeline_to_digitone_bundle_plan(song, timeline, tp)
    return song, timeline, bundle_plan, diagnostics


def save_digitone_pipeline_artifacts(
    output_dir: str | Path,
    song: SongModel,
    timeline: RenderedTimeline,
    plan: DigitoneCompilePlan,
    events_payload: dict,
    write_syx: bool = False,
    syx_filename: str = "digitone_pattern.syx",
) -> DigitonePipelineArtifacts:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    song_json = out / "song_model.json"
    timeline_json = out / "rendered_timeline.json"
    plan_json = out / "digitone_compile_plan.json"
    events_yaml = out / "digitone.events.yaml"

    song_json.write_text(json.dumps(song_model_to_dict(song), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    timeline_json.write_text(
        json.dumps(rendered_timeline_to_dict(timeline), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    plan_json.write_text(json.dumps(digitone_compile_plan_to_dict(plan), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    events_yaml.write_text(yaml.safe_dump(events_payload, sort_keys=False, allow_unicode=False), encoding="utf-8")

    artifacts: DigitonePipelineArtifacts = DigitonePipelineArtifacts(
        {
            "song_model_json": song_json,
            "rendered_timeline_json": timeline_json,
            "digitone_compile_plan_json": plan_json,
            "events_yaml": events_yaml,
        }
    )

    if write_syx:
        syx_path = out / syx_filename
        build_digitone_syx_from_events_yaml(events_yaml, syx_path)
        artifacts["syx"] = syx_path

    return artifacts


def save_digitone_bundle_artifacts(
    output_dir: str | Path,
    song: SongModel,
    timeline: RenderedTimeline,
    bundle_plan: DigitonePatternBundlePlan,
    write_syx: bool = False,
    bundle_syx_filename: str | None = None,
    harmony_resolution: dict | None = None,
    target_profile: DigitoneTargetProfile | None = None,
    pattern_change_policy: str = "auto_song_mode",
) -> DigitonePipelineArtifacts:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    patterns_dir = out / "patterns"
    patterns_dir.mkdir(parents=True, exist_ok=True)

    song_json = out / "song_model.json"
    timeline_json = out / "rendered_timeline.json"
    bundle_plan_json = out / "digitone_bundle_plan.json"
    bundle_manifest_json = out / "bundle_manifest.json"
    harmony_resolution_json = out / "musicxml_harmony_resolution.json"

    song_json.write_text(json.dumps(song_model_to_dict(song), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    timeline_json.write_text(
        json.dumps(rendered_timeline_to_dict(timeline), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    bundle_plan_json.write_text(
        json.dumps(digitone_pattern_bundle_plan_to_dict(bundle_plan), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    tp = target_profile or default_digitone_target_profile()

    pattern_entries: list[dict] = []
    syx_paths: list[Path] = []
    for index, pattern in enumerate(bundle_plan.patterns, start=1):
        safe_name = _safe_ascii_slug(pattern.pattern_name, fallback=f"PATTERN_{index:02d}")
        base_name = f"{index:02d}_{safe_name}"
        events_file = patterns_dir / f"{base_name}.digitone.events.yaml"
        events_payload = digitone_pattern_segment_to_events_yaml_payload(
            pattern,
            bundle_plan.timing,
            track_default_velocity=tp.track_default_velocity,
            pattern_change_policy=pattern_change_policy,
        )
        events_file.write_text(yaml.safe_dump(events_payload, sort_keys=False, allow_unicode=False), encoding="utf-8")

        entry = {
            "index": index,
            "segment_index": pattern.segment_index,
            "pattern_name": pattern.pattern_name,
            "pattern_name_source": pattern.pattern_name_source,
            "section_id": pattern.section_id,
            "section_label": pattern.section_label,
            "section_token": pattern.section_token,
            "section_occurrence_index": pattern.section_occurrence_index,
            "section_global_order_index": pattern.section_global_order_index,
            "section_split_index": pattern.section_split_index,
            "section_split_count": pattern.section_split_count,
            "global_step_start": pattern.global_step_start,
            "global_step_end": pattern.global_step_end,
            "total_steps": pattern.total_steps,
            "warnings": list(pattern.warnings),
            "events_yaml": str(events_file.relative_to(out).as_posix()),
            "events_yaml_path": str(events_file.relative_to(out).as_posix()),
        }

        if write_syx:
            syx_file = patterns_dir / f"{base_name}.syx"
            build_digitone_syx_from_events_yaml(events_file, syx_file)
            syx_paths.append(syx_file)
            entry["syx"] = str(syx_file.relative_to(out).as_posix())
            entry["syx_path"] = str(syx_file.relative_to(out).as_posix())

        pattern_entries.append(entry)

    bundle_file: Path | None = None
    if write_syx and syx_paths:
        if bundle_syx_filename is None:
            bundle_syx_filename = f"{_safe_ascii_slug(bundle_plan.source_title)}.bundle.syx"
        bundle_file = out / bundle_syx_filename
        bundle_file.write_bytes(b"".join(path.read_bytes() for path in syx_paths))

    bundle_manifest = {
        "source_title": bundle_plan.source_title,
        "pattern_count": len(pattern_entries),
        "timing": {
            "performance_tempo": str(bundle_plan.timing.performance_tempo),
            "speed": bundle_plan.timing.speed,
            "speed_ratio": str(bundle_plan.timing.speed_ratio),
            "q_step": str(bundle_plan.timing.q_step),
            "device_tempo": str(bundle_plan.timing.device_tempo),
        },
        "warnings": list(bundle_plan.warnings),
        "patterns": pattern_entries,
    }
    if bundle_file is not None:
        bundle_manifest["bundle_syx"] = str(bundle_file.relative_to(out).as_posix())

    bundle_manifest_json.write_text(json.dumps(bundle_manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    if harmony_resolution is not None:
        harmony_resolution_json.write_text(
            json.dumps(harmony_resolution, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )

    artifacts: DigitonePipelineArtifacts = DigitonePipelineArtifacts(
        {
            "song_model_json": song_json,
            "rendered_timeline_json": timeline_json,
            "digitone_bundle_plan_json": bundle_plan_json,
            "bundle_manifest_json": bundle_manifest_json,
            "patterns_dir": patterns_dir,
        }
    )

    if harmony_resolution is not None:
        artifacts["musicxml_harmony_resolution_json"] = harmony_resolution_json

    if bundle_file is not None:
        artifacts["bundle_syx"] = bundle_file

    return artifacts
