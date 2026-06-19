from __future__ import annotations

import json

import pytest

from changes.ai_generation import (
    AI_COMPOSER,
    AiGenerationSettings,
    AiGenerationError,
    append_evaluation_log,
    ensure_ollama_ready,
    generate_harmony_from_ollama,
    _extract_ollama_response_text,
    _ollama_has_model,
    normalize_ai_chord_symbol,
    result_from_json_text,
)


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
        "Emaj7(#11)",
        "|",
        "Cmaj7(#11)",
        "|",
        "Abmaj7(#11)",
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
        ("G7b9#11", "G7(b9,#11)"),
        ("G7#9b13", "G7(#9,b13)"),
        ("Cmaj7#11", "Cmaj7(#11)"),
        ("F13b9", "F13(b9)"),
        ("Bb7#11", "Bb7(#11)"),
    ],
)
def test_normalize_ai_chord_symbol_for_parser_readable_tensions(raw: str, expected: str) -> None:
    assert normalize_ai_chord_symbol(raw) == expected


@pytest.mark.parametrize("raw", ["G7alt", "G7(b13)", "G7(b9,#11)"])
def test_normalize_ai_chord_symbol_allows_alt_or_explicit_alterations(raw: str) -> None:
    assert normalize_ai_chord_symbol(raw) == raw


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
    payload["progression"] = [{"chord": "G7b9#11", "beats": 4}]

    result = result_from_json_text(json.dumps(payload), user_prompt="x", model_name="test-model")

    assert result.song.measures[0].harmony[0].symbol == "G7(b9,#11)"
    assert result.editor_state.cells == ["G7(b9,#11)", "|"]


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
