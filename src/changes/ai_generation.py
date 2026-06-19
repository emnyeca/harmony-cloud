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
_AI_COMPACT_ALTERATION_RE = re.compile(
    r"^(?P<root>[A-G](?:#|b)?)(?P<base>maj13|maj9|maj7|mMaj7|m7b5|m11|m9|m7|13|11|9|7)"
    r"(?P<alterations>(?:[#b](?:5|9|11|13))+)$"
)
_AI_ALTERATION_RE = re.compile(r"[#b](?:5|9|11|13)")
_AI_ALT_EXPLICIT_ALTERATION_RE = re.compile(
    r"^(?P<root>[A-G](?:#|b)?)(?P<body>.*alt.*[#b](?:5|9|11|13).*|.*[#b](?:5|9|11|13).*alt.*)$"
)
_AI_MINOR_ALIAS_RE = re.compile(r"^(?P<root>[A-G](?:#|b)?)(?:minor|min)(?P<rest>.*)$")


class AiGenerationError(RuntimeError):
    """Raised when generation cannot safely be applied to the dirty editor."""


@dataclass(frozen=True)
class AiGenerationSettings:
    enabled: bool
    endpoint: str
    model_name: str


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
) -> AiGenerationResult:
    if not settings.enabled:
        raise AiGenerationError("AI generation is disabled in settings.")

    raw = _call_ollama(user_prompt, settings, timeout_seconds=timeout_seconds)
    return result_from_json_text(raw, user_prompt=user_prompt, model_name=settings.model_name)


def result_from_json_text(
    json_text: str,
    *,
    user_prompt: str,
    model_name: str,
) -> AiGenerationResult:
    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise AiGenerationError(f"JSON could not be parsed: {exc.msg}") from exc

    if not isinstance(payload, dict):
        raise AiGenerationError("Generated JSON must be an object.")

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
        raise AiGenerationError("Tempo is not readable.") from exc
    if tempo <= 0:
        raise AiGenerationError("Tempo must be positive.")

    meter = str(payload.get("meter") or "4/4").strip()
    meter_num, meter_den = _parse_meter(meter)
    measure_length = Fraction(4 * meter_num, meter_den)

    progression = payload.get("progression")
    if not isinstance(progression, list) or not progression:
        raise AiGenerationError("Progression must be a non-empty list.")

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
            raise AiGenerationError(f"Progression item {index} must be an object.")
        symbol = str(item.get("chord") or "").strip()
        if not symbol:
            raise AiGenerationError(f"Progression item {index} has no chord.")
        normalized_symbol = normalize_ai_chord_symbol(symbol)
        try:
            parse_chord_core(normalized_symbol)
        except Exception as exc:
            raise AiGenerationError(
                f"Chord name cannot be parsed: {symbol} -> {normalized_symbol}"
            ) from exc
        try:
            beats = Fraction(str(item.get("beats", 4))).limit_denominator(1000)
        except Exception as exc:
            raise AiGenerationError(f"Beats cannot be parsed for chord: {symbol}") from exc
        if beats <= 0:
            raise AiGenerationError(f"Beats must be positive for chord: {symbol}")
        if offset + beats > measure_length:
            raise AiGenerationError(
                f"Chord duration crosses a barline: {symbol} at progression item {index}"
            )

        measure_events.append((normalized_symbol, offset, beats))
        cells.append(normalized_symbol)
        offset += beats
        if offset == measure_length:
            flush_measure()

    if measure_events:
        flush_measure()

    if not measures:
        raise AiGenerationError("Generated progression produced no measures.")

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
    return song, state


def normalize_ai_chord_symbol(symbol: str) -> str:
    text = str(symbol).strip()
    text = _normalize_ai_minor_alias(text)
    _validate_ai_alt_not_mixed_with_explicit_alterations(text)
    if "(" in text or ")" in text:
        return _normalize_parenthesized_ai_chord_symbol(text)

    slash = ""
    main = text
    if "/" in text:
        main, slash_part = text.split("/", 1)
        slash = f"/{slash_part.strip()}"

    match = _AI_COMPACT_ALTERATION_RE.match(main)
    if not match:
        return text

    alterations = _AI_ALTERATION_RE.findall(match.group("alterations"))
    if not alterations or "".join(alterations) != match.group("alterations"):
        return text
    return f"{match.group('root')}{match.group('base')}({','.join(alterations)}){slash}"


def _normalize_ai_minor_alias(symbol: str) -> str:
    slash = ""
    main = symbol
    if "/" in symbol:
        main, slash_part = symbol.split("/", 1)
        slash = f"/{slash_part.strip()}"
    match = _AI_MINOR_ALIAS_RE.match(main.strip())
    if not match:
        return symbol
    return f"{match.group('root')}m{match.group('rest')}{slash}"


def _validate_ai_alt_not_mixed_with_explicit_alterations(symbol: str) -> None:
    main = symbol.split("/", 1)[0].strip()
    if _AI_ALT_EXPLICIT_ALTERATION_RE.match(main):
        raise AiGenerationError(
            f"Chord name cannot be parsed: {symbol} -> {symbol}; "
            "alt cannot be combined with explicit alterations"
        )
    if "alt" in main and "(" in main and ")" in main:
        inner = main.split("(", 1)[1].split(")", 1)[0]
        if _AI_ALTERATION_RE.search(inner):
            raise AiGenerationError(
                f"Chord name cannot be parsed: {symbol} -> {symbol}; "
                "alt cannot be combined with explicit alterations"
            )


def _normalize_parenthesized_ai_chord_symbol(symbol: str) -> str:
    if "(" not in symbol or ")" not in symbol:
        return symbol
    before, rest = symbol.split("(", 1)
    inner, after = rest.split(")", 1)
    tensions = [part.strip() for part in inner.split(",") if part.strip()]
    return f"{before.strip()}({','.join(tensions)}){after.strip()}"


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


def _call_ollama(
    user_prompt: str,
    settings: AiGenerationSettings,
    *,
    timeout_seconds: float,
) -> str:
    endpoint = settings.endpoint.rstrip("/")
    body = {
        "model": settings.model_name,
        "prompt": _build_prompt(user_prompt),
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
            f"Detail: {exc.reason}"
        ) from exc
    except TimeoutError as exc:
        raise AiGenerationError("Ollama request timed out.") from exc
    except Exception as exc:
        raise AiGenerationError(f"Ollama request failed: {exc}") from exc

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
        raise AiGenerationError("Ollama returned an unreadable response payload.")
    error = data.get("error")
    if error:
        raise AiGenerationError(f"Ollama generation error: {error}")
    text = data.get("response")
    if isinstance(text, str) and text.strip():
        return text.strip()
    thinking = data.get("thinking")
    if isinstance(thinking, str) and thinking.strip():
        raise AiGenerationError(
            "Ollama returned thinking text but no final JSON response. "
            "This usually means the selected model is in thinking mode. "
            "Restart EUB Changes and try again, or use a non-thinking model."
        )
    keys = ", ".join(sorted(str(k) for k in data.keys()))
    raise AiGenerationError(f"Ollama returned no JSON response. Payload keys: {keys}")


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


def _build_prompt(user_prompt: str) -> str:
    title = (user_prompt or "").strip() or "Untitled harmony sketch"
    return f"""/no_think
You generate chord progressions for EUB Changes.

User title or natural-language prompt:
{title}

Create a musically balanced progression with a non-diatonic tendency. Avoid ordinary pop loops, simple diatonic cycles, and easy I-vi-IV-V behavior. Use nearby modulation, borrowed chords, modal colors, remote key relationships, or floating harmony when useful. Do not make the result complex only for theory's sake.

Return JSON only. Do not include explanation, markdown, poetic comments, or any text outside JSON.

Chord notation rules:
- Use compact but parseable chord names.
- For multiple alterations or tensions, use parentheses and comma separators.
- Prefer `G7(b9,#11)` instead of `G7b9#11`.
- Prefer `Cmaj7(#11)` instead of `Cmaj7#11` if multiple parser-safe forms are available.
- Do not combine `alt` with explicit alterations.
- Use `G7alt`, not `G7b13alt`.
- Use `G7(b13)` if only b13 is intended.
- Use `G7(b9,#11)` if specific alterations are intended.
- Return JSON only.

Schema:
{{
  "title": "{title}",
  "composer": "{AI_COMPOSER}",
  "tempo": 48,
  "meter": "4/4",
  "progression": [
    {{ "chord": "Emaj7(#11)", "beats": 4 }},
    {{ "chord": "Cmaj7(#11)", "beats": 4 }},
    {{ "chord": "Abmaj7(#11)", "beats": 4 }},
    {{ "chord": "Dbmaj9", "beats": 4 }}
  ]
}}

Use only chord symbols compatible with this style: maj7, maj9, maj7(#11), maj13, m7, m9, m11, mMaj7, m7b5, dim7, 7, 9, 13, 7(b9), 7(#9), 7(#11), 7(b13), 7(#5), 7(b5), 7sus4, 9sus4, 7b9sus4, alt, slash bass chords with note basses. Beats must be positive and should normally complete whole bars in the meter.
/no_think"""


def _parse_meter(meter: str) -> tuple[int, int]:
    parts = meter.split("/")
    if len(parts) != 2:
        raise AiGenerationError(f"Meter is not readable: {meter}")
    try:
        numerator = int(parts[0])
        denominator = int(parts[1])
    except ValueError as exc:
        raise AiGenerationError(f"Meter is not readable: {meter}") from exc
    if numerator <= 0 or denominator <= 0:
        raise AiGenerationError(f"Meter must be positive: {meter}")
    return numerator, denominator
