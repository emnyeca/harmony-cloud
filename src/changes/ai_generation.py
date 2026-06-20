"""Local Ollama-backed harmony generation for the Changes editor."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from changes.chord_parser import parse_chord_core
from changes.editor import EditorState
from changes.models.song_model import HarmonyEvent, Measure, SongModel

AI_COMPOSER = "Emnyeca Harmony AI"

# Chord qualities the AI may use. Every entry is verified to pass the full UI
# pipeline (parse -> SongModel -> compile_song_for_ui -> render_arrangement ->
# harmonic_context / voicing) by
# tests/test_ai_generation.py::test_ai_safe_chord_qualities_representatives_compile_through_ui_pipeline.
# This is intentionally an MVP set: stability before breadth. The parser and the
# rest of the pipeline support more qualities for manual input; we simply do not
# ask the model to produce them yet. Dense double-altered dominants
# (7b9#11, 7#5b9, 13b9, ...) are deliberately excluded to keep generated harmony
# from drowning in color.
AI_SAFE_CHORD_QUALITIES = frozenset(
    {
        # major / triad
        "",
        "6",
        "6/9",
        "maj7",
        "maj9",
        "maj13",
        "maj7#11",
        # minor
        "m",
        "m6",
        "m6/9",
        "m7",
        "m9",
        "m11",
        "mMaj7",
        # dominant
        "7",
        "9",
        "13",
        "7b9",
        "7#9",
        "7#11",
        "7b13",
        "7#5",
        "7b5",
        "7sus4",
        "9sus4",
        "7b9sus4",
        "alt",
        # diminished / half-diminished
        "m7b5",
        "dim7",
    }
)
_AI_SAFE_CHORD_QUALITY_PROMPT = ", ".join(sorted(AI_SAFE_CHORD_QUALITIES, key=lambda item: (len(item), item)))

# Normalization only repairs spelling. It never rewrites a quality that is already
# a known canonical form, so half-diminished `m7b5` is never corrupted into the
# non-canonical `m7(b5)`. See normalize_ai_chord_symbol for the ordered steps.
_AI_ALTERATION_RE = re.compile(r"[#b](?:5|9|11|13)")
_AI_ALT_EXPLICIT_ALTERATION_RE = re.compile(
    r"^(?P<root>[A-G](?:#|b)?)(?P<body>.*alt.*[#b](?:5|9|11|13).*|.*[#b](?:5|9|11|13).*alt.*)$"
)
_AI_ROOT_QUALITY_RE = re.compile(r"^(?P<root>[A-G](?:#|b)?)(?P<quality>.*)$")
_AI_MAJOR_SPELLING_RE = re.compile(r"^(?:major|maj)", re.IGNORECASE)
_AI_MINOR_SPELLING_RE = re.compile(r"^(?:minor|min)", re.IGNORECASE)


# Failure stages recorded in the failure log. Kept as constants so the log schema
# stays stable and greppable.
FAILURE_STAGE_OLLAMA_RESPONSE = "ollama_response"
FAILURE_STAGE_JSON_PARSE = "json_parse"
FAILURE_STAGE_PAYLOAD_SCHEMA = "payload_schema"
FAILURE_STAGE_NORMALIZATION = "normalization"
FAILURE_STAGE_CHORD_PARSE = "chord_parse"
FAILURE_STAGE_VOCABULARY = "ai_pipeline_vocabulary_validation"
FAILURE_STAGE_PIPELINE = "pipeline_validation"


class AiGenerationError(RuntimeError):
    """Raised when generation cannot safely be applied to the dirty editor.

    Carries optional structured metadata so the generation flow can record a
    machine-readable failure record without re-parsing the message text.
    """

    def __init__(
        self,
        message: str,
        *,
        stage: str = "unknown",
        failed_chord: str | None = None,
        normalized_chord: str | None = None,
        normalized_quality: str | None = None,
        suggested_fix: str | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.failed_chord = failed_chord
        self.normalized_chord = normalized_chord
        self.normalized_quality = normalized_quality
        self.suggested_fix = suggested_fix


@dataclass(frozen=True)
class AiGenerationSettings:
    enabled: bool
    endpoint: str
    model_name: str
    max_validation_retries: int = 1


@dataclass(frozen=True)
class AiGenerationResult:
    user_prompt: str
    generated_json: dict[str, Any]
    song: SongModel
    editor_state: EditorState
    model_name: str


@dataclass(frozen=True)
class OllamaBootstrapStatus:
    ok: bool
    message: str
    detail: str | None = None


def ensure_ollama_ready(
    settings: AiGenerationSettings,
    *,
    startup_timeout_seconds: float = 8.0,
    pull_timeout_seconds: float = 1800.0,
) -> OllamaBootstrapStatus:
    if not settings.enabled:
        return OllamaBootstrapStatus(ok=True, message="AI generation is disabled.")

    endpoint = settings.endpoint.rstrip("/")
    ollama_path = _find_ollama_executable()
    if ollama_path is None:
        return OllamaBootstrapStatus(
            ok=False,
            message="Ollama command was not found. Install Ollama or add it to PATH.",
        )

    tags = _fetch_ollama_tags(endpoint, timeout_seconds=1.5)
    if tags is None:
        _start_ollama_server(ollama_path)
        deadline = time.monotonic() + startup_timeout_seconds
        while time.monotonic() < deadline:
            tags = _fetch_ollama_tags(endpoint, timeout_seconds=1.0)
            if tags is not None:
                break
            time.sleep(0.4)

    if tags is None:
        return OllamaBootstrapStatus(
            ok=False,
            message=f"Ollama did not start at {endpoint}.",
            detail="Check that Ollama is installed correctly and that the endpoint setting matches its port.",
        )

    if _ollama_has_model(tags, settings.model_name):
        return OllamaBootstrapStatus(ok=True, message=f"Ollama is ready: {settings.model_name}")

    pull = subprocess.run(
        [ollama_path, "pull", settings.model_name],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=pull_timeout_seconds,
    )
    if pull.returncode != 0:
        detail = (pull.stderr or pull.stdout or "").strip() or f"ollama pull exited with {pull.returncode}"
        return OllamaBootstrapStatus(
            ok=False,
            message=f"Ollama model pull failed: {settings.model_name}",
            detail=detail,
        )

    tags = _fetch_ollama_tags(endpoint, timeout_seconds=3.0)
    if tags is not None and _ollama_has_model(tags, settings.model_name):
        return OllamaBootstrapStatus(ok=True, message=f"Ollama model is ready: {settings.model_name}")
    return OllamaBootstrapStatus(
        ok=False,
        message=f"Ollama model was pulled but is not listed: {settings.model_name}",
    )


def generate_harmony_from_ollama(
    user_prompt: str,
    settings: AiGenerationSettings,
    *,
    timeout_seconds: float = 90.0,
    failure_log_path: str | Path | None = None,
) -> AiGenerationResult:
    """Generate a progression, validating it through the full UI pipeline.

    Every rejected attempt (bad JSON, unparseable chord, out-of-vocabulary
    quality, pipeline failure, or an unreachable Ollama) is written to
    ``failure_log_path`` before raising or retrying, so failures remain decision
    material even when the user never sees the result. The dirty editor is only
    touched by the caller on success.
    """
    if not settings.enabled:
        raise AiGenerationError("AI generation is disabled in settings.", stage="disabled")

    max_retries = max(0, int(settings.max_validation_retries))
    failure_memory = recent_failure_memory(failure_log_path)
    retry_context: dict[str, str] | None = None

    for attempt in range(1, max_retries + 2):
        try:
            raw = _call_ollama(
                user_prompt,
                settings,
                timeout_seconds=timeout_seconds,
                retry_context=retry_context,
                failure_memory=failure_memory,
            )
        except AiGenerationError as exc:
            _log_generation_failure(
                failure_log_path,
                user_prompt=user_prompt,
                model_name=settings.model_name,
                attempt=attempt,
                raw_response_text=None,
                error=exc,
            )
            raise

        try:
            return result_from_json_text(
                raw, user_prompt=user_prompt, model_name=settings.model_name
            )
        except AiGenerationError as exc:
            _log_generation_failure(
                failure_log_path,
                user_prompt=user_prompt,
                model_name=settings.model_name,
                attempt=attempt,
                raw_response_text=raw,
                error=exc,
            )
            if attempt >= max_retries + 1:
                raise
            retry_context = {"previous_output": raw, "error": str(exc)}

    raise AiGenerationError(
        "AI generation failed after validation retry.", stage=FAILURE_STAGE_VOCABULARY
    )


def result_from_json_text(
    json_text: str,
    *,
    user_prompt: str,
    model_name: str,
) -> AiGenerationResult:
    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise AiGenerationError(
            f"JSON could not be parsed: {exc.msg}", stage=FAILURE_STAGE_JSON_PARSE
        ) from exc

    if not isinstance(payload, dict):
        raise AiGenerationError(
            "Generated JSON must be an object.", stage=FAILURE_STAGE_PAYLOAD_SCHEMA
        )

    song, editor_state = song_and_editor_from_payload(payload, fallback_title=user_prompt)
    return AiGenerationResult(
        user_prompt=user_prompt,
        generated_json=payload,
        song=song,
        editor_state=editor_state,
        model_name=model_name,
    )


def song_and_editor_from_payload(
    payload: dict[str, Any],
    *,
    fallback_title: str,
) -> tuple[SongModel, EditorState]:
    title = str(payload.get("title") or fallback_title or "NO TITLE").strip() or "NO TITLE"
    composer = str(payload.get("composer") or AI_COMPOSER).strip() or AI_COMPOSER
    if composer != AI_COMPOSER:
        composer = AI_COMPOSER

    try:
        tempo = Fraction(str(payload.get("tempo", 72))).limit_denominator(1000)
    except Exception as exc:
        raise AiGenerationError("Tempo is not readable.", stage=FAILURE_STAGE_PAYLOAD_SCHEMA) from exc
    if tempo <= 0:
        raise AiGenerationError("Tempo must be positive.", stage=FAILURE_STAGE_PAYLOAD_SCHEMA)

    meter = str(payload.get("meter") or "4/4").strip()
    meter_num, meter_den = _parse_meter(meter)
    measure_length = Fraction(4 * meter_num, meter_den)

    progression = payload.get("progression")
    if not isinstance(progression, list) or not progression:
        raise AiGenerationError(
            "Progression must be a non-empty list.", stage=FAILURE_STAGE_PAYLOAD_SCHEMA
        )

    measures: list[Measure] = []
    cells: list[str] = []
    measure_events: list[tuple[str, Fraction, Fraction]] = []
    measure_number = 1
    absolute_start = Fraction(0)
    offset = Fraction(0)

    def flush_measure() -> None:
        nonlocal measure_events, measure_number, absolute_start, offset
        if not measure_events:
            return
        harmony = tuple(
            HarmonyEvent(
                id=f"m{measure_number}_h{idx}",
                symbol=symbol,
                measure_number=measure_number,
                offset_quarters=event_offset,
                duration_quarters=duration,
            )
            for idx, (symbol, event_offset, duration) in enumerate(measure_events, start=1)
        )
        measures.append(
            Measure(
                number=measure_number,
                section_id="A__OCC1",
                meter_numerator=meter_num,
                meter_denominator=meter_den,
                absolute_start_quarters=absolute_start,
                harmony=harmony,
            )
        )
        cells.append("|")
        measure_number += 1
        absolute_start += measure_length
        offset = Fraction(0)
        measure_events = []

    for index, item in enumerate(progression, start=1):
        if not isinstance(item, dict):
            raise AiGenerationError(
                f"Progression item {index} must be an object.", stage=FAILURE_STAGE_PAYLOAD_SCHEMA
            )
        symbol = str(item.get("chord") or "").strip()
        if not symbol:
            raise AiGenerationError(
                f"Progression item {index} has no chord.", stage=FAILURE_STAGE_PAYLOAD_SCHEMA
            )
        normalized_symbol = normalize_ai_chord_symbol(symbol)
        try:
            core = parse_chord_core(normalized_symbol)
        except Exception as exc:
            raise AiGenerationError(
                f"Chord name cannot be parsed: {symbol} -> {normalized_symbol}",
                stage=FAILURE_STAGE_CHORD_PARSE,
                failed_chord=symbol,
                normalized_chord=normalized_symbol,
                suggested_fix="Use a chord name in AI_SAFE_CHORD_QUALITIES with a valid root.",
            ) from exc
        if core.normalized_quality not in AI_SAFE_CHORD_QUALITIES:
            raise AiGenerationError(
                "Chord failed AI pipeline vocabulary validation: "
                f"{symbol} -> {normalized_symbol} "
                f"(normalized quality: {core.normalized_quality})",
                stage=FAILURE_STAGE_VOCABULARY,
                failed_chord=symbol,
                normalized_chord=normalized_symbol,
                normalized_quality=core.normalized_quality,
                suggested_fix=_vocabulary_suggested_fix(core.normalized_quality),
            )
        try:
            beats = Fraction(str(item.get("beats", 4))).limit_denominator(1000)
        except Exception as exc:
            raise AiGenerationError(
                f"Beats cannot be parsed for chord: {symbol}",
                stage=FAILURE_STAGE_PAYLOAD_SCHEMA,
                failed_chord=symbol,
            ) from exc
        if beats <= 0:
            raise AiGenerationError(
                f"Beats must be positive for chord: {symbol}",
                stage=FAILURE_STAGE_PAYLOAD_SCHEMA,
                failed_chord=symbol,
            )
        if offset + beats > measure_length:
            raise AiGenerationError(
                f"Chord duration crosses a barline: {symbol} at progression item {index}",
                stage=FAILURE_STAGE_PAYLOAD_SCHEMA,
                failed_chord=symbol,
            )

        measure_events.append((normalized_symbol, offset, beats))
        cells.append(normalized_symbol)
        offset += beats
        if offset == measure_length:
            flush_measure()

    if measure_events:
        flush_measure()

    if not measures:
        raise AiGenerationError(
            "Generated progression produced no measures.", stage=FAILURE_STAGE_PAYLOAD_SCHEMA
        )

    state = EditorState(
        title=title,
        tempo=int(tempo),
        meter=meter,
        composer=AI_COMPOSER,
        cells=cells,
        cursor=len(cells),
    )
    song = SongModel(
        title=title,
        working_key=state.working_key or None,
        working_key_mode=None,
        performance_tempo=tempo,
        measures=tuple(measures),
        composer=AI_COMPOSER,
    )
    _validate_generated_song_pipeline(song)
    return song, state


def normalize_ai_chord_symbol(symbol: str) -> str:
    """Repair chord-name spelling without rewriting known canonical qualities.

    Ordered steps (see ai-governance: protect known qualities first):
      1. Split root + quality from an optional slash bass.
      2. Normalize spelling variants (Maj/MAJ/Major -> maj, Min/minor -> m).
      3. Reject `alt` mixed with explicit alterations (ambiguous, not safe).
      4. Collapse the `(alt)` alias into `alt` (e.g. D7(alt) -> D7alt).

    It deliberately does NOT translate compact alterations into parenthesized
    form. The parser already accepts both spellings, so any extra rewrite only
    risked corrupting canonical qualities such as `m7b5` into `m7(b5)`.
    """
    main, slash = _split_ai_slash(str(symbol))
    main = _normalize_ai_quality_spelling(main)
    _validate_ai_alt_not_mixed_with_explicit_alterations(f"{main}{slash}")
    if "(" in main or ")" in main:
        main = _normalize_parenthesized_ai_chord_symbol(main)
    return f"{main}{slash}"


def _split_ai_slash(symbol: str) -> tuple[str, str]:
    text = symbol.strip()
    if "/" in text:
        main, bass = text.split("/", 1)
        return main.strip(), f"/{bass.strip()}"
    return text, ""


def _normalize_ai_quality_spelling(main: str) -> str:
    match = _AI_ROOT_QUALITY_RE.match(main)
    if not match:
        return main
    root = match.group("root")
    quality = match.group("quality")
    # Major spelling first so "minMaj7" -> "mMaj7" keeps the canonical capital M.
    quality = _AI_MAJOR_SPELLING_RE.sub("maj", quality)
    quality = _AI_MINOR_SPELLING_RE.sub("m", quality)
    return f"{root}{quality}"


def _validate_ai_alt_not_mixed_with_explicit_alterations(symbol: str) -> None:
    main = symbol.split("/", 1)[0].strip()
    mixed = bool(_AI_ALT_EXPLICIT_ALTERATION_RE.match(main))
    if not mixed and "alt" in main and "(" in main and ")" in main:
        inner = main.split("(", 1)[1].split(")", 1)[0]
        mixed = bool(_AI_ALTERATION_RE.search(inner))
    if mixed:
        raise AiGenerationError(
            f"Chord name cannot be parsed: {symbol} -> {symbol}; "
            "alt cannot be combined with explicit alterations",
            stage=FAILURE_STAGE_NORMALIZATION,
            failed_chord=symbol,
            normalized_chord=symbol,
            suggested_fix="Use plain `7alt` for an altered dominant, or list explicit alterations without `alt`.",
        )


def _normalize_parenthesized_ai_chord_symbol(main: str) -> str:
    if "(" not in main or ")" not in main:
        return main
    before, rest = main.split("(", 1)
    inner, after = rest.split(")", 1)
    tensions = [part.strip() for part in inner.split(",") if part.strip()]
    if len(tensions) == 1 and tensions[0] == "alt" and before.strip().endswith("7"):
        return f"{before.strip()}alt{after.strip()}"
    return f"{before.strip()}({','.join(tensions)}){after.strip()}"


def _validate_generated_song_pipeline(song: SongModel) -> None:
    try:
        _compile_generated_song(song)
        return
    except Exception as whole_song_exc:
        for symbol in _song_chord_symbols(song):
            probe = _single_chord_song(song, symbol)
            try:
                _compile_generated_song(probe)
            except Exception as chord_exc:
                quality = _normalized_quality_for_error(symbol)
                raise AiGenerationError(
                    "Chord failed pipeline validation at compile_song_for_ui: "
                    f"{symbol} (normalized quality: {quality}): {chord_exc}",
                    stage=FAILURE_STAGE_PIPELINE,
                    failed_chord=symbol,
                    normalized_quality=quality,
                    suggested_fix="Remove this chord; it parses but breaks the UI pipeline.",
                ) from chord_exc
        raise AiGenerationError(
            "Generated song failed pipeline validation at compile_song_for_ui: "
            f"{whole_song_exc}",
            stage=FAILURE_STAGE_PIPELINE,
        ) from whole_song_exc


def _compile_generated_song(song: SongModel) -> None:
    from changes.app_settings import AppSettings
    from changes.ui_pipeline import compile_song_for_ui

    compile_song_for_ui(song, AppSettings())


def _normalized_quality_for_error(symbol: str) -> str:
    try:
        return parse_chord_core(symbol).normalized_quality
    except Exception:
        return "unreadable"


# Short, specific guidance for the most common out-of-vocabulary qualities the
# model invents. Falls back to a generic message for the rest.
_VOCABULARY_FIX_HINTS = {
    "m9(b5)": "Use `m7b5` for half-diminished, or `m9` for a minor ninth. Do not combine them.",
    "m7(b5)": "Use the canonical `m7b5` spelling.",
    "m13": "Use `m11` or `m9`; `m13` is not in the AI vocabulary.",
    "13b9": "Use `7b9` or plain `13`.",
    "7b9#11": "Use `7b9` or `7#11`, not both.",
}


def _vocabulary_suggested_fix(normalized_quality: str) -> str:
    return _VOCABULARY_FIX_HINTS.get(
        normalized_quality,
        "Use only qualities listed in AI_SAFE_CHORD_QUALITIES.",
    )


def _song_chord_symbols(song: SongModel) -> tuple[str, ...]:
    symbols: list[str] = []
    for measure in song.measures:
        for harmony in measure.harmony:
            symbols.append(harmony.symbol)
    return tuple(symbols)


def _single_chord_song(source: SongModel, symbol: str) -> SongModel:
    meter_num = source.measures[0].meter_numerator if source.measures else 4
    meter_den = source.measures[0].meter_denominator if source.measures else 4
    duration = Fraction(4 * meter_num, meter_den)
    measure = Measure(
        number=1,
        section_id="A__OCC1",
        meter_numerator=meter_num,
        meter_denominator=meter_den,
        absolute_start_quarters=Fraction(0),
        harmony=(
            HarmonyEvent(
                id="m1_h1",
                symbol=symbol,
                measure_number=1,
                offset_quarters=Fraction(0),
                duration_quarters=duration,
            ),
        ),
    )
    return SongModel(
        title=source.title,
        working_key=source.working_key,
        working_key_mode=source.working_key_mode,
        performance_tempo=source.performance_tempo,
        measures=(measure,),
        composer=source.composer,
    )


def append_evaluation_log(
    log_path: str | Path,
    *,
    user_prompt: str,
    generated_json: dict[str, Any] | None,
    decision: str,
    rating: int | None,
    model_name: str,
    error: str | None = None,
) -> None:
    if decision not in {"use", "reject"}:
        raise ValueError("decision must be 'use' or 'reject'")
    if rating is not None and not 1 <= int(rating) <= 5:
        raise ValueError("rating must be 1..5 or None")
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_prompt": user_prompt,
        "generated_json": generated_json,
        "decision": decision,
        "rating": int(rating) if rating is not None else None,
        "model_name": model_name,
        "error": error,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def append_generation_failure_log(
    log_path: str | Path,
    *,
    user_prompt: str,
    model_name: str,
    attempt: int,
    raw_response_text: str | None,
    failure_stage: str,
    failure_reason: str,
    parsed_json: dict[str, Any] | None = None,
    failed_chord: str | None = None,
    normalized_chord: str | None = None,
    normalized_quality: str | None = None,
    suggested_fix: str | None = None,
    event_type: str = "generation_rejected",
) -> None:
    """Append one machine-readable failure record (JSONL).

    This is separate from append_evaluation_log: evaluation logs capture the
    user's Use/Reject/Rating on results that reached the editor, while these
    records capture failures that never got that far. Both feed future
    improvement (prompt, normalization, fine-tuning).
    """
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "event_type": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_prompt": user_prompt,
        "model_name": model_name,
        "attempt": int(attempt),
        "raw_response_text": raw_response_text,
        "parsed_json": parsed_json,
        "failed_chord": failed_chord,
        "normalized_chord": normalized_chord,
        "normalized_quality": normalized_quality,
        "failure_stage": failure_stage,
        "failure_reason": failure_reason,
        "suggested_fix": suggested_fix,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def _log_generation_failure(
    log_path: str | Path | None,
    *,
    user_prompt: str,
    model_name: str,
    attempt: int,
    raw_response_text: str | None,
    error: Exception,
) -> None:
    """Best-effort failure logging. Never raises into the generation path."""
    if not log_path:
        return
    stage = getattr(error, "stage", FAILURE_STAGE_OLLAMA_RESPONSE)
    try:
        append_generation_failure_log(
            log_path,
            user_prompt=user_prompt,
            model_name=model_name,
            attempt=attempt,
            raw_response_text=raw_response_text,
            parsed_json=_loads_dict_or_none(raw_response_text),
            failure_stage=stage,
            failure_reason=str(error),
            failed_chord=getattr(error, "failed_chord", None),
            normalized_chord=getattr(error, "normalized_chord", None),
            normalized_quality=getattr(error, "normalized_quality", None),
            suggested_fix=getattr(error, "suggested_fix", None),
        )
    except Exception:
        # Logging must not mask the real failure or break generation.
        pass


def _loads_dict_or_none(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    try:
        value = json.loads(text)
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def recent_failure_memory(
    log_path: str | Path | None,
    *,
    limit: int = 5,
) -> tuple[str, ...]:
    """Short, deduplicated guidance lines built from the most recent failures.

    Fed back into the next prompt so the model stops repeating the same spelling
    or vocabulary mistakes. Kept tiny (a few lines) so generation stays fast.
    """
    if not log_path:
        return ()
    path = Path(log_path)
    if not path.is_file():
        return ()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return ()

    memory: list[str] = []
    seen: set[str] = set()
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except Exception:
            continue
        hint = _failure_memory_line(record)
        if hint and hint not in seen:
            seen.add(hint)
            memory.append(hint)
        if len(memory) >= limit:
            break
    memory.reverse()
    return tuple(memory)


def _failure_memory_line(record: dict[str, Any]) -> str | None:
    chord = record.get("failed_chord")
    fix = record.get("suggested_fix")
    if chord and fix:
        return f"- `{chord}` was rejected. {fix}"
    if chord:
        return f"- `{chord}` was rejected; use only AI_SAFE_CHORD_QUALITIES."
    return None


def _call_ollama(
    user_prompt: str,
    settings: AiGenerationSettings,
    *,
    timeout_seconds: float,
    retry_context: dict[str, str] | None = None,
    failure_memory: tuple[str, ...] = (),
) -> str:
    endpoint = settings.endpoint.rstrip("/")
    body = {
        "model": settings.model_name,
        "prompt": (
            _build_retry_prompt(
                user_prompt,
                previous_output=retry_context["previous_output"],
                validation_error=retry_context["error"],
                failure_memory=failure_memory,
            )
            if retry_context is not None
            else _build_prompt(user_prompt, failure_memory=failure_memory)
        ),
        "stream": False,
        "format": "json",
        "think": False,
    }
    request = Request(
        f"{endpoint}/api/generate",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
    except URLError as exc:
        raise AiGenerationError(
            f"Ollama is not reachable at {endpoint}. Start Ollama or update the Ollama endpoint in Settings. "
            f"Detail: {exc.reason}",
            stage=FAILURE_STAGE_OLLAMA_RESPONSE,
        ) from exc
    except TimeoutError as exc:
        raise AiGenerationError(
            "Ollama request timed out.", stage=FAILURE_STAGE_OLLAMA_RESPONSE
        ) from exc
    except Exception as exc:
        raise AiGenerationError(
            f"Ollama request failed: {exc}", stage=FAILURE_STAGE_OLLAMA_RESPONSE
        ) from exc

    return _extract_ollama_response_text(data)


def _fetch_ollama_tags(endpoint: str, *, timeout_seconds: float) -> dict[str, Any] | None:
    request = Request(f"{endpoint.rstrip('/')}/api/tags", method="GET")
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _extract_ollama_response_text(data: Any) -> str:
    if not isinstance(data, dict):
        raise AiGenerationError(
            "Ollama returned an unreadable response payload.", stage=FAILURE_STAGE_OLLAMA_RESPONSE
        )
    error = data.get("error")
    if error:
        raise AiGenerationError(
            f"Ollama generation error: {error}", stage=FAILURE_STAGE_OLLAMA_RESPONSE
        )
    text = data.get("response")
    if isinstance(text, str) and text.strip():
        return text.strip()
    thinking = data.get("thinking")
    if isinstance(thinking, str) and thinking.strip():
        raise AiGenerationError(
            "Ollama returned thinking text but no final JSON response. "
            "This usually means the selected model is in thinking mode. "
            "Restart EUB Changes and try again, or use a non-thinking model.",
            stage=FAILURE_STAGE_OLLAMA_RESPONSE,
        )
    keys = ", ".join(sorted(str(k) for k in data.keys()))
    raise AiGenerationError(
        f"Ollama returned no JSON response. Payload keys: {keys}",
        stage=FAILURE_STAGE_OLLAMA_RESPONSE,
    )


def _find_ollama_executable() -> str | None:
    found = shutil.which("ollama")
    if found:
        return found
    candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
        Path(os.environ.get("ProgramFiles", "")) / "Ollama" / "ollama.exe",
        Path(os.environ.get("ProgramFiles(x86)", "")) / "Ollama" / "ollama.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def _start_ollama_server(ollama_path: str) -> None:
    startupinfo = None
    creationflags = 0
    if hasattr(subprocess, "STARTUPINFO"):
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    subprocess.Popen(
        [ollama_path, "serve"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        startupinfo=startupinfo,
        creationflags=creationflags,
    )


def _ollama_has_model(tags: dict[str, Any], model_name: str) -> bool:
    wanted = str(model_name).strip()
    wanted_base = wanted.split(":", 1)[0]
    models = tags.get("models")
    if not isinstance(models, list):
        return False
    for model in models:
        if not isinstance(model, dict):
            continue
        name = str(model.get("name") or "")
        if name == wanted or name.split(":", 1)[0] == wanted_base:
            return True
    return False


def _build_prompt(user_prompt: str, *, failure_memory: tuple[str, ...] = ()) -> str:
    title = (user_prompt or "").strip() or "Untitled harmony sketch"
    return f"""/no_think
You generate chord progressions for EUB Changes.

User title or natural-language prompt:
{title}

Musical balance is the first priority. Match the mood of the prompt. Use non-diatonic color only when it fits the prompt. You may use nearby modulation, borrowed chords, modal colors, remote key relationships, or floating harmony when the mood calls for it, but do not make harmony complex only for theory's sake.

Tension density:
- Do not add extensions or alterations to every chord.
- For simple, pop, light, bright, or upbeat prompts, use mostly maj7, 6, m7, 7, sus4, and occasional 9.
- Use highly colored chords such as maj7#11, 7b9, 7#11, 7b13, or alt only 0-2 times in an 8-bar progression unless the prompt clearly asks for dense jazz harmony.
- Do not add #11, b9, b13, or alt unless the prompt clearly benefits from it.

Tempo guidance (choose a tempo that fits the prompt; do not default to a slow tempo):
- slow / ambient / floating / solemn / divine / 神々 / 浮遊 / アンビエント / 荘厳: 48-72 BPM is allowed.
- rainy / mellow / downtempo / 雨 / ダウナー / 深夜: 68-88 BPM.
- simple pop / medium / gentle / シンプル / ポップス / 明るい: 90-130 BPM.
- upbeat / light / energetic / dance / 元気 / 軽快 / ノリノリ / アップテンポ: 120-150 BPM.
- Bright, pop, light, or energetic prompts should land in the 90-150 BPM range, not slow.

Return JSON only. Do not include explanation, markdown, poetic comments, or any text outside JSON.

Chord notation rules:
- Use compact, parseable chord names (for example C, G7, Cmaj7, Am7, D7alt).
- For altered dominants, either name one alteration (G7b9, G7b13) or use the single token `alt` (G7alt).
- Do not combine `alt` with explicit alterations: use `G7alt`, not `G7b13alt`.
- Write `D7alt`, not `D7(alt)`.
- Keep half-diminished as `m7b5` (for example F#m7b5), and diminished sevenths as `dim7`.
- Use only these pipeline-safe chord qualities: {_AI_SAFE_CHORD_QUALITY_PROMPT}.
- Return JSON only.
{_failure_memory_block(failure_memory)}
Format example only. This is not a musical recommendation; copy the shape, not the chords or tempo:
{{
  "title": "{title}",
  "composer": "{AI_COMPOSER}",
  "tempo": 120,
  "meter": "4/4",
  "progression": [
    {{ "chord": "C", "beats": 4 }},
    {{ "chord": "G7", "beats": 4 }}
  ]
}}

Slash bass chords with note basses are allowed when musically useful. Beats must be positive and should normally complete whole bars in the meter.
/no_think"""


def _failure_memory_block(failure_memory: tuple[str, ...]) -> str:
    if not failure_memory:
        return ""
    lines = "\n".join(failure_memory)
    return (
        "\nRecent rejected chord spellings (do not repeat these mistakes):\n"
        f"{lines}\n"
    )


def _build_retry_prompt(
    user_prompt: str,
    *,
    previous_output: str,
    validation_error: str,
    failure_memory: tuple[str, ...] = (),
) -> str:
    title = (user_prompt or "").strip() or "Untitled harmony sketch"
    return f"""{_build_prompt(user_prompt, failure_memory=failure_memory)}

Previous output was rejected.

Rejected output:
{previous_output}

Validation failure:
{validation_error}

Regeneration rules:
- Previous output was rejected.
- The chord `Dm9(b5)` is not allowed.
- Use only AI_SAFE_CHORD_QUALITIES: {_AI_SAFE_CHORD_QUALITY_PROMPT}.
- Do not invent combined qualities such as `m9(b5)`.
- If b5 minor is intended, use `m7b5`.
- If minor 9 is intended, use `m9`.
- Fix the specific chord named in the validation failure above.
- Keep the title close to: {title}
- Return JSON only.
/no_think"""


def _parse_meter(meter: str) -> tuple[int, int]:
    parts = meter.split("/")
    if len(parts) != 2:
        raise AiGenerationError(f"Meter is not readable: {meter}", stage=FAILURE_STAGE_PAYLOAD_SCHEMA)
    try:
        numerator = int(parts[0])
        denominator = int(parts[1])
    except ValueError as exc:
        raise AiGenerationError(
            f"Meter is not readable: {meter}", stage=FAILURE_STAGE_PAYLOAD_SCHEMA
        ) from exc
    if numerator <= 0 or denominator <= 0:
        raise AiGenerationError(
            f"Meter must be positive: {meter}", stage=FAILURE_STAGE_PAYLOAD_SCHEMA
        )
    return numerator, denominator
