from __future__ import annotations

import json

import pytest

from changes.ai_generation import (
    AI_COMPOSER,
    AI_SAFE_CHORD_QUALITIES,
    AiGenerationSettings,
    AiGenerationError,
    _build_prompt,
    _build_retry_prompt,
    append_evaluation_log,
    append_generation_failure_log,
    ensure_ollama_ready,
    generate_harmony_from_ollama,
    recent_failure_memory,
    _extract_ollama_response_text,
    _ollama_has_model,
    normalize_ai_chord_symbol,
    result_from_json_text,
)
from changes.app_settings import AppSettings
from changes.chord_parser import parse_chord_core
from changes.ui_pipeline import compile_song_for_ui


def _payload() -> dict:
    return {
        "title": "神々の住まう領域",
        "composer": AI_COMPOSER,
        "tempo": 48,
        "meter": "4/4",
        "progression": [
            {"chord": "Emaj7#11", "beats": 4},
            {"chord": "Cmaj7#11", "beats": 4},
            {"chord": "Abmaj7#11", "beats": 4},
            {"chord": "Dbmaj9", "beats": 4},
        ],
    }


def test_result_from_json_text_builds_dirty_song_and_editor_state() -> None:
    result = result_from_json_text(
        json.dumps(_payload(), ensure_ascii=False),
        user_prompt="神々の住まう領域",
        model_name="test-model",
    )

    assert result.song.title == "神々の住まう領域"
    assert result.song.composer == AI_COMPOSER
    assert result.song.performance_tempo == 48
    assert len(result.song.measures) == 4
    assert result.editor_state.title == "神々の住まう領域"
    assert result.editor_state.composer == AI_COMPOSER
    assert result.editor_state.cells == [
        "Emaj7#11",
        "|",
        "Cmaj7#11",
        "|",
        "Abmaj7#11",
        "|",
        "Dbmaj9",
        "|",
    ]


def test_result_from_json_text_rejects_invalid_json() -> None:
    with pytest.raises(AiGenerationError, match="JSON could not be parsed"):
        result_from_json_text("not json", user_prompt="x", model_name="test-model")


def test_result_from_json_text_rejects_unparseable_chord() -> None:
    payload = _payload()
    payload["progression"][0]["chord"] = "Hmaj7"

    with pytest.raises(AiGenerationError, match="Chord name cannot be parsed: Hmaj7 -> Hmaj7"):
        result_from_json_text(json.dumps(payload), user_prompt="x", model_name="test-model")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Known canonical qualities must survive normalization untouched. The
        # half-diminished `m7b5` regression (it used to become `m7(b5)`) is the
        # headline case here.
        ("F#m7b5", "F#m7b5"),
        ("Cdim7", "Cdim7"),
        ("G7b9", "G7b9"),
        ("Bb7#11", "Bb7#11"),
        ("F7b13", "F7b13"),
        # Spelling repair only; compact and parenthesized tensions both parse.
        ("EbMaj7", "Ebmaj7"),
        ("EbMaj7(#11)", "Ebmaj7(#11)"),
        ("Cmaj7#11", "Cmaj7#11"),
        ("D7(alt)", "D7alt"),
        ("Eb6/9", "Eb6/9"),
        ("Cm6/9", "Cm6/9"),
    ],
)
def test_normalize_ai_chord_symbol_protects_canonical_and_repairs_spelling(
    raw: str, expected: str
) -> None:
    assert normalize_ai_chord_symbol(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Ebmin6", "Ebm6"),
        ("Cmin7", "Cm7"),
        ("Fmin9", "Fm9"),
        ("Bbmin11", "Bbm11"),
        ("AminMaj7", "AmMaj7"),
        ("Ebminor6", "Ebm6"),
        ("EbMAJ7", "Ebmaj7"),
        ("CMajor7", "Cmaj7"),
    ],
)
def test_normalize_ai_chord_symbol_converts_spelling_aliases(raw: str, expected: str) -> None:
    assert normalize_ai_chord_symbol(raw) == expected


@pytest.mark.parametrize("raw", ["G7alt", "G7(b13)", "G7(b9,#11)"])
def test_normalize_ai_chord_symbol_allows_alt_or_explicit_alterations(raw: str) -> None:
    assert normalize_ai_chord_symbol(raw) == raw


def test_normalize_ai_chord_symbol_converts_parenthesized_alt() -> None:
    assert normalize_ai_chord_symbol("D7(alt)") == "D7alt"


@pytest.mark.parametrize("raw", ["G7b13alt", "G7altb9", "C7#9alt", "G7alt(b9)"])
def test_normalize_ai_chord_symbol_rejects_alt_mixed_with_explicit_alterations(raw: str) -> None:
    with pytest.raises(AiGenerationError, match="alt cannot be combined with explicit alterations"):
        normalize_ai_chord_symbol(raw)


def test_result_from_json_text_rejects_alt_mixed_with_explicit_alterations() -> None:
    payload = _payload()
    payload["progression"] = [{"chord": "G7b13alt", "beats": 4}]

    with pytest.raises(
        AiGenerationError,
        match="Chord name cannot be parsed: G7b13alt -> G7b13alt; alt cannot be combined",
    ):
        result_from_json_text(json.dumps(payload), user_prompt="x", model_name="test-model")


def test_result_from_json_text_stores_normalized_ai_chords() -> None:
    payload = _payload()
    payload["progression"] = [
        {"chord": "EbMaj7", "beats": 4},
        {"chord": "F#m7b5", "beats": 4},
    ]

    result = result_from_json_text(json.dumps(payload), user_prompt="x", model_name="test-model")

    # Spelling repaired, half-diminished left canonical (not corrupted to m7(b5)).
    assert result.song.measures[0].harmony[0].symbol == "Ebmaj7"
    assert result.song.measures[1].harmony[0].symbol == "F#m7b5"
    assert result.editor_state.cells == ["Ebmaj7", "|", "F#m7b5", "|"]


def test_result_from_json_text_accepts_parenthesized_major_sharp_eleven_through_ui_pipeline() -> None:
    payload = _payload()
    payload["progression"] = [{"chord": "Cmaj7(#11)", "beats": 4}]

    result = result_from_json_text(json.dumps(payload), user_prompt="x", model_name="test-model")

    assert result.song.measures[0].harmony[0].symbol == "Cmaj7(#11)"
    assert parse_chord_core("Cmaj7(#11)").normalized_quality == "maj7#11"
    compile_song_for_ui(result.song, AppSettings())


def test_result_from_json_text_rejects_quality_not_in_ai_safe_pipeline_vocabulary() -> None:
    payload = _payload()
    payload["progression"] = [{"chord": "Dm13", "beats": 4}]

    with pytest.raises(AiGenerationError, match="Chord name cannot be parsed|pipeline vocabulary"):
        result_from_json_text(json.dumps(payload), user_prompt="x", model_name="test-model")


def test_result_from_json_text_stores_normalized_minor_alias() -> None:
    payload = _payload()
    payload["progression"] = [{"chord": "Ebmin6", "beats": 4}]

    result = result_from_json_text(json.dumps(payload), user_prompt="x", model_name="test-model")

    assert result.song.measures[0].harmony[0].symbol == "Ebm6"
    assert result.editor_state.cells == ["Ebm6", "|"]


def test_result_from_json_text_accepts_six_nine_quality() -> None:
    payload = _payload()
    payload["progression"] = [{"chord": "Eb6/9", "beats": 4}]

    result = result_from_json_text(json.dumps(payload), user_prompt="x", model_name="test-model")

    assert result.song.measures[0].harmony[0].symbol == "Eb6/9"
    assert result.editor_state.cells == ["Eb6/9", "|"]


def test_ai_safe_chord_qualities_representatives_compile_through_ui_pipeline() -> None:
    # One representative AI chord per safe quality. Each is run through the real
    # AI path (normalize -> parse -> vocabulary validation -> SongModel ->
    # compile_song_for_ui -> render_arrangement -> harmonic_context / voicing).
    # If any quality cannot pass the pipeline it must not be in the safe set.
    symbols_by_quality = {
        "": "C",
        "6": "C6",
        "6/9": "C6/9",
        "maj7": "Cmaj7",
        "maj9": "Cmaj9",
        "maj13": "Cmaj13",
        "maj7#11": "Cmaj7#11",
        "m": "Cm",
        "m6": "Cm6",
        "m6/9": "Cm6/9",
        "m7": "Cm7",
        "m9": "Cm9",
        "m11": "Cm11",
        "mMaj7": "CmMaj7",
        "7": "G7",
        "9": "G9",
        "13": "G13",
        "7b9": "G7b9",
        "7#9": "G7#9",
        "7#11": "G7#11",
        "7b13": "G7b13",
        "7#5": "G7#5",
        "7b5": "G7b5",
        "7sus4": "G7sus4",
        "9sus4": "G9sus4",
        "7b9sus4": "G7b9sus4",
        "alt": "G7alt",
        "m7b5": "Cm7b5",
        "dim7": "Cdim7",
    }
    assert set(symbols_by_quality) == set(AI_SAFE_CHORD_QUALITIES)

    for quality, symbol in symbols_by_quality.items():
        payload = _payload()
        payload["progression"] = [{"chord": symbol, "beats": 4}]
        result = result_from_json_text(
            json.dumps(payload), user_prompt="pipeline", model_name="test-model"
        )
        assert parse_chord_core(result.song.measures[0].harmony[0].symbol).normalized_quality == quality
        # result_from_json_text already runs full pipeline validation, but assert
        # the dirty song compiles for the UI as the explicit safety boundary.
        compile_song_for_ui(result.song, AppSettings())


def test_ai_safe_slash_bass_chord_passes_pipeline() -> None:
    payload = _payload()
    payload["progression"] = [{"chord": "C/E", "beats": 4}]
    result = result_from_json_text(json.dumps(payload), user_prompt="x", model_name="test-model")
    assert result.song.measures[0].harmony[0].symbol == "C/E"
    compile_song_for_ui(result.song, AppSettings())


def test_build_prompt_uses_neutral_schema_example_and_light_prompt_guidance() -> None:
    prompt = _build_prompt("元気で軽快")

    assert '"tempo": 48' not in prompt
    assert "Format example only. This is not a musical recommendation" in prompt
    assert "90-150 BPM" in prompt
    assert "48-72 BPM is allowed" in prompt
    assert "Cmaj7(#11)" not in prompt.split("Format example only.", 1)[1]


def test_build_retry_prompt_includes_rejected_chord_guidance() -> None:
    prompt = _build_retry_prompt(
        "x",
        previous_output='{"progression":[{"chord":"Dm9(b5)","beats":4}]}',
        validation_error=(
            "Chord failed AI pipeline vocabulary validation: "
            "Dm9(b5) -> Dm9(b5) (normalized quality: m9(b5))"
        ),
    )

    assert "Previous output was rejected." in prompt
    assert "The chord `Dm9(b5)` is not allowed." in prompt
    assert "Use only AI_SAFE_CHORD_QUALITIES" in prompt
    assert "Do not invent combined qualities such as `m9(b5)`." in prompt
    assert "If b5 minor is intended, use `m7b5`." in prompt
    assert "If minor 9 is intended, use `m9`." in prompt
    assert "normalized quality: m9(b5)" in prompt


def test_generate_harmony_retries_once_after_validation_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    invalid_payload = _payload()
    invalid_payload["progression"] = [{"chord": "Dm9(b5)", "beats": 4}]
    valid_payload = _payload()
    valid_payload["progression"] = [{"chord": "Dm9", "beats": 4}]
    responses = [
        json.dumps(invalid_payload, ensure_ascii=False),
        json.dumps(valid_payload, ensure_ascii=False),
    ]
    retry_contexts: list[dict[str, str] | None] = []

    def fake_call_ollama(
        user_prompt: str,
        settings: AiGenerationSettings,
        *,
        timeout_seconds: float,
        retry_context: dict[str, str] | None = None,
        failure_memory: tuple[str, ...] = (),
    ) -> str:
        retry_contexts.append(retry_context)
        return responses.pop(0)

    monkeypatch.setattr("changes.ai_generation._call_ollama", fake_call_ollama)

    result = generate_harmony_from_ollama(
        "retry me",
        AiGenerationSettings(True, "http://localhost:11434", "test-model", max_validation_retries=1),
    )

    assert result.song.measures[0].harmony[0].symbol == "Dm9"
    assert retry_contexts[0] is None
    assert retry_contexts[1] is not None
    assert "Dm9(b5) -> Dm9(b5)" in retry_contexts[1]["error"]
    assert "normalized quality: m9(b5)" in retry_contexts[1]["error"]


def test_generate_harmony_reports_validation_error_after_retry_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    invalid_payload = _payload()
    invalid_payload["progression"] = [{"chord": "Dm9(b5)", "beats": 4}]
    calls = 0

    def fake_call_ollama(
        user_prompt: str,
        settings: AiGenerationSettings,
        *,
        timeout_seconds: float,
        retry_context: dict[str, str] | None = None,
        failure_memory: tuple[str, ...] = (),
    ) -> str:
        nonlocal calls
        calls += 1
        return json.dumps(invalid_payload, ensure_ascii=False)

    monkeypatch.setattr("changes.ai_generation._call_ollama", fake_call_ollama)

    with pytest.raises(AiGenerationError, match=r"Dm9\(b5\).*normalized quality: m9\(b5\)"):
        generate_harmony_from_ollama(
            "retry me",
            AiGenerationSettings(True, "http://localhost:11434", "test-model", max_validation_retries=1),
        )
    assert calls == 2


@pytest.mark.parametrize("chord", ["Dm9(b5)", "Cm9(b5)"])
def test_result_from_json_text_rejects_non_canonical_half_diminished(chord: str) -> None:
    payload = _payload()
    payload["progression"] = [{"chord": chord, "beats": 4}]

    with pytest.raises(AiGenerationError, match="pipeline vocabulary validation"):
        result_from_json_text(json.dumps(payload), user_prompt="x", model_name="test-model")


@pytest.mark.parametrize("chord", ["G7b13alt", "G7altb9", "C7#9alt"])
def test_result_from_json_text_rejects_alt_mixed_chords(chord: str) -> None:
    payload = _payload()
    payload["progression"] = [{"chord": chord, "beats": 4}]

    with pytest.raises(AiGenerationError, match="alt cannot be combined"):
        result_from_json_text(json.dumps(payload), user_prompt="x", model_name="test-model")


def test_append_generation_failure_log_writes_full_schema(tmp_path) -> None:
    log_path = tmp_path / "ai-failures.jsonl"

    append_generation_failure_log(
        log_path,
        user_prompt="神々の住まう領域",
        model_name="test-model",
        attempt=2,
        raw_response_text='{"x":1}',
        parsed_json={"x": 1},
        failure_stage="chord_parse",
        failure_reason="Chord name cannot be parsed: Hmaj7 -> Hmaj7",
        failed_chord="Hmaj7",
        normalized_chord="Hmaj7",
        normalized_quality="unreadable",
        suggested_fix="Use a chord name in AI_SAFE_CHORD_QUALITIES with a valid root.",
    )

    record = json.loads(log_path.read_text(encoding="utf-8").strip())
    for key in (
        "event_type",
        "timestamp",
        "user_prompt",
        "model_name",
        "attempt",
        "raw_response_text",
        "parsed_json",
        "failed_chord",
        "normalized_chord",
        "normalized_quality",
        "failure_stage",
        "failure_reason",
        "suggested_fix",
    ):
        assert key in record
    assert record["event_type"] == "generation_rejected"
    assert record["attempt"] == 2
    assert record["failed_chord"] == "Hmaj7"
    assert record["failure_stage"] == "chord_parse"


def test_generate_harmony_logs_each_failed_attempt(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    invalid_payload = _payload()
    invalid_payload["progression"] = [{"chord": "Dm9(b5)", "beats": 4}]

    def fake_call_ollama(
        user_prompt: str,
        settings: AiGenerationSettings,
        *,
        timeout_seconds: float,
        retry_context: dict[str, str] | None = None,
        failure_memory: tuple[str, ...] = (),
    ) -> str:
        return json.dumps(invalid_payload, ensure_ascii=False)

    monkeypatch.setattr("changes.ai_generation._call_ollama", fake_call_ollama)
    log_path = tmp_path / "ai-failures.jsonl"

    with pytest.raises(AiGenerationError):
        generate_harmony_from_ollama(
            "retry me",
            AiGenerationSettings(True, "http://localhost:11434", "test-model", max_validation_retries=1),
            failure_log_path=log_path,
        )

    records = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 2  # initial attempt + one retry, both rejected and logged
    first = records[0]
    assert first["failure_stage"] == "ai_pipeline_vocabulary_validation"
    assert first["failed_chord"] == "Dm9(b5)"
    assert first["normalized_quality"] == "m9(b5)"
    assert first["attempt"] == 1
    assert first["suggested_fix"]


def test_generate_harmony_logs_ollama_response_failure(tmp_path) -> None:
    log_path = tmp_path / "ai-failures.jsonl"

    with pytest.raises(AiGenerationError, match="Ollama is not reachable"):
        generate_harmony_from_ollama(
            "x",
            AiGenerationSettings(True, "http://127.0.0.1:9", "missing"),
            timeout_seconds=0.2,
            failure_log_path=log_path,
        )

    record = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert record["failure_stage"] == "ollama_response"
    assert record["attempt"] == 1


def test_recent_failure_memory_summarizes_recent_failures(tmp_path) -> None:
    log_path = tmp_path / "ai-failures.jsonl"
    append_generation_failure_log(
        log_path,
        user_prompt="x",
        model_name="m",
        attempt=1,
        raw_response_text=None,
        failure_stage="ai_pipeline_vocabulary_validation",
        failure_reason="...",
        failed_chord="Dm9(b5)",
        normalized_quality="m9(b5)",
        suggested_fix="Use m7b5 for half-diminished, or m9 for a minor ninth.",
    )

    memory = recent_failure_memory(log_path)

    assert memory
    assert any("Dm9(b5)" in line for line in memory)


def test_build_prompt_includes_failure_memory_when_present() -> None:
    prompt = _build_prompt(
        "x", failure_memory=("- `Dm9(b5)` was rejected. Use m7b5 or m9.",)
    )

    assert "Recent rejected chord spellings" in prompt
    assert "Dm9(b5)" in prompt


def test_append_evaluation_log_writes_jsonl(tmp_path) -> None:
    log_path = tmp_path / "ai-eval.jsonl"

    append_evaluation_log(
        log_path,
        user_prompt="神々の住まう領域",
        generated_json=_payload(),
        decision="use",
        rating=5,
        model_name="test-model",
        error=None,
    )

    line = log_path.read_text(encoding="utf-8").strip()
    record = json.loads(line)
    assert record["user_prompt"] == "神々の住まう領域"
    assert record["decision"] == "use"
    assert record["rating"] == 5
    assert record["model_name"] == "test-model"
    assert record["error"] is None


def test_ollama_connection_error_is_actionable() -> None:
    with pytest.raises(AiGenerationError, match="Ollama is not reachable"):
        generate_harmony_from_ollama(
            "x",
            AiGenerationSettings(True, "http://127.0.0.1:9", "missing"),
            timeout_seconds=0.2,
        )


def test_ollama_model_match_accepts_latest_tag() -> None:
    tags = {"models": [{"name": "llama3.1:latest"}]}

    assert _ollama_has_model(tags, "llama3.1")
    assert _ollama_has_model(tags, "llama3.1:latest")


def test_ensure_ollama_ready_reports_missing_command(monkeypatch) -> None:
    monkeypatch.setattr("changes.ai_generation._find_ollama_executable", lambda: None)

    status = ensure_ollama_ready(AiGenerationSettings(True, "http://localhost:11434", "llama3.1"))

    assert status.ok is False
    assert "Ollama command was not found" in status.message


def test_extract_ollama_response_reports_error_payload() -> None:
    with pytest.raises(AiGenerationError, match="Ollama generation error: model not found"):
        _extract_ollama_response_text({"error": "model not found"})


def test_extract_ollama_response_reports_thinking_without_response() -> None:
    with pytest.raises(AiGenerationError, match="thinking text but no final JSON response"):
        _extract_ollama_response_text({"response": "", "thinking": '{"ok": true}'})


