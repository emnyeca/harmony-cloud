"""Tests for the Emnyeca Harmony AI evaluation loop (Prompt Library -> Batch ->
Candidate Store -> Critic Export/Import -> Human Evaluation Queue)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from fractions import Fraction

import pytest

from changes.ai_evaluation import (
    BatchProgress,
    CandidateView,
    add_prompt,
    append_human_evaluation,
    build_candidate_views,
    candidate_record_from_result,
    critic_export_rows,
    enabled_prompts,
    filter_candidate_views,
    format_critic_export,
    human_review_queue,
    import_critic_results,
    latest_critic_by_candidate,
    load_candidates,
    load_prompts,
    next_candidate_id,
    next_prompt_id,
    parse_critic_import,
    reviewed_candidate_ids,
    run_batch_generation,
    seed_prompt_library,
    set_prompt_enabled,
    update_prompt,
)
from changes.ai_generation import AI_COMPOSER, AiGenerationError, AiGenerationResult, result_from_json_text
from changes.editor import EditorState
from changes.models.song_model import SongModel


_DAY = datetime(2026, 6, 19, 12, 0, tzinfo=timezone.utc)


def _result(prompt: str, progression=None, tempo: int = 120) -> AiGenerationResult:
    payload = {
        "title": prompt,
        "composer": AI_COMPOSER,
        "tempo": tempo,
        "meter": "4/4",
        "progression": progression
        or [
            {"chord": "C", "beats": 4},
            {"chord": "G7", "beats": 4},
        ],
    }
    return result_from_json_text(json.dumps(payload, ensure_ascii=False), user_prompt=prompt, model_name="test-model")


# ── Phase 1: Prompt Library ─────────────────────────────────────────────────────


def test_prompt_library_add_edit_disable(tmp_path) -> None:
    path = tmp_path / "prompts.jsonl"

    a = add_prompt(path, "元気で軽快", category="upbeat", now=_DAY)
    b = add_prompt(path, "静かな湖", category="ambient", now=_DAY)

    assert a.prompt_id != b.prompt_id  # unique
    assert [p.prompt for p in load_prompts(path)] == ["元気で軽快", "静かな湖"]

    update_prompt(path, a.prompt_id, prompt="とても元気", category="energetic")
    set_prompt_enabled(path, b.prompt_id, False)

    reloaded = {p.prompt_id: p for p in load_prompts(path)}
    assert reloaded[a.prompt_id].prompt == "とても元気"
    assert reloaded[a.prompt_id].category == "energetic"
    assert reloaded[b.prompt_id].enabled is False
    assert [p.prompt_id for p in enabled_prompts(path)] == [a.prompt_id]


def test_prompt_id_is_unique_and_sequenced() -> None:
    existing = ["prompt-20260619-0001", "prompt-20260619-0002"]
    assert next_prompt_id(existing, _DAY) == "prompt-20260619-0003"
    assert next_prompt_id([], _DAY) == "prompt-20260619-0001"


def test_add_prompt_rejects_empty(tmp_path) -> None:
    with pytest.raises(ValueError):
        add_prompt(tmp_path / "p.jsonl", "   ")


def test_seed_prompt_library_is_idempotent(tmp_path) -> None:
    path = tmp_path / "prompts.jsonl"
    add_prompt(path, "元気で軽快", category="manual", now=_DAY)

    added = seed_prompt_library(path, now=_DAY)
    # The already-present text is skipped; the rest of the seed is added once.
    assert "元気で軽快" not in [p.prompt for p in added]
    total_after_first = len(load_prompts(path))

    seed_prompt_library(path, now=_DAY)
    assert len(load_prompts(path)) == total_after_first  # no duplicates on re-seed


# ── Phase 3: Candidate id ───────────────────────────────────────────────────────


def test_candidate_id_is_unique_and_sequenced() -> None:
    assert next_candidate_id([], _DAY) == "ehm-20260619-0001"
    assert next_candidate_id(["ehm-20260619-0001", "ehm-20260619-0009"], _DAY) == "ehm-20260619-0010"


# ── Phase 4: Candidate Store ────────────────────────────────────────────────────


def test_candidate_record_from_result_uses_normalized_song() -> None:
    result = _result("神々の住まう領域", progression=[{"chord": "EbMaj7", "beats": 4}], tempo=56)
    record = candidate_record_from_result(
        result, candidate_id="ehm-20260619-0001", prompt_id="prompt-20260619-0001", category="ambient", now=_DAY
    )

    assert record["candidate_id"] == "ehm-20260619-0001"
    assert record["prompt_id"] == "prompt-20260619-0001"
    assert record["category"] == "ambient"
    assert record["tempo"] == 56
    assert record["meter"] == "4/4"
    # Spelling repaired by the generation pipeline before storage.
    assert record["progression"] == [{"chord": "Ebmaj7", "beats": 4}]
    assert record["composer"] == AI_COMPOSER
    assert record["generated_json"]["progression"][0]["chord"] == "EbMaj7"


# ── Phase 2: Batch Generate ─────────────────────────────────────────────────────


def test_batch_generation_handles_multiple_prompts_and_failures(tmp_path) -> None:
    prompts_path = tmp_path / "prompts.jsonl"
    candidates_path = tmp_path / "candidates.jsonl"
    failures_path = tmp_path / "failures.jsonl"
    p1 = add_prompt(prompts_path, "元気で軽快", category="upbeat", now=_DAY)
    p2 = add_prompt(prompts_path, "壊れた遊園地", category="dark", now=_DAY)

    calls: list[str] = []

    def fake_generate(prompt, settings, *, failure_log_path=None, **kwargs):
        calls.append(prompt)
        # The second prompt always fails to exercise failure handling.
        if prompt == "壊れた遊園地":
            raise AiGenerationError("boom", stage="pipeline_validation")
        return _result(prompt)

    summary = run_batch_generation(
        load_prompts(prompts_path),
        2,
        settings=None,
        candidate_store_path=candidates_path,
        failure_log_path=failures_path,
        generate_fn=fake_generate,
        now=_DAY,
    )

    assert len(calls) == 4  # 2 prompts x 2 candidates, batch did not abort
    assert summary.total_success == 2
    assert summary.total_failed == 2
    stored = load_candidates(candidates_path)
    assert len(stored) == 2
    assert all(c["prompt_id"] == p1.prompt_id for c in stored)
    assert len({c["candidate_id"] for c in stored}) == 2  # unique ids
    _ = p2


def test_batch_generation_reports_progress(tmp_path) -> None:
    prompts_path = tmp_path / "prompts.jsonl"
    add_prompt(prompts_path, "a", now=_DAY)
    add_prompt(prompts_path, "b", now=_DAY)
    progress: list[BatchProgress] = []

    run_batch_generation(
        load_prompts(prompts_path),
        3,
        settings=None,
        candidate_store_path=tmp_path / "c.jsonl",
        generate_fn=lambda prompt, settings, **kw: _result(prompt),
        progress_cb=progress.append,
        now=_DAY,
    )

    assert len(progress) == 6  # 2 x 3
    assert progress[-1].total_success == 6
    assert progress[-1].prompt_total == 2
    assert progress[-1].candidate_total == 3


# ── Phase 5: Critic Export ──────────────────────────────────────────────────────


def _seed_candidates(tmp_path):
    prompts_path = tmp_path / "prompts.jsonl"
    candidates_path = tmp_path / "candidates.jsonl"
    add_prompt(prompts_path, "元気で軽快", category="upbeat", now=_DAY)
    add_prompt(prompts_path, "静かな湖", category="ambient", now=_DAY)
    run_batch_generation(
        load_prompts(prompts_path),
        2,
        settings=None,
        candidate_store_path=candidates_path,
        generate_fn=lambda prompt, settings, **kw: _result(prompt),
        now=_DAY,
    )
    return candidates_path


def test_critic_export_only_unreviewed_and_minimal_fields(tmp_path) -> None:
    candidates_path = _seed_candidates(tmp_path)
    all_ids = [c["candidate_id"] for c in load_candidates(candidates_path)]

    rows = critic_export_rows(candidates_path, reviewed_ids={all_ids[0]})

    assert len(rows) == 3  # one excluded as already reviewed
    assert all_ids[0] not in {r["candidate_id"] for r in rows}
    # Critic must not see internal fields.
    assert set(rows[0]) == {"candidate_id", "prompt_id", "prompt", "tempo", "meter", "progression"}
    assert "model_name" not in rows[0]
    assert "generated_json" not in rows[0]

    text = format_critic_export(rows)
    assert json.loads(text)[0]["candidate_id"] == rows[0]["candidate_id"]


def test_critic_export_filters_by_category(tmp_path) -> None:
    candidates_path = _seed_candidates(tmp_path)
    rows = critic_export_rows(candidates_path, category="ambient")
    assert rows
    assert all(r["prompt"] == "静かな湖" for r in rows)


# ── Phase 6: Critic Import ──────────────────────────────────────────────────────


def test_parse_critic_import_accepts_array_and_jsonl() -> None:
    array = '[{"candidate_id": "ehm-1", "critic_score": 4}]'
    jsonl = '{"candidate_id": "ehm-1", "critic_score": 4}\n{"candidate_id": "ehm-2", "critic_score": 2}'
    assert len(parse_critic_import(array)) == 1
    assert len(parse_critic_import(jsonl)) == 2
    assert parse_critic_import("") == []


def test_import_critic_results_links_and_detects_unknown(tmp_path) -> None:
    critic_path = tmp_path / "critic.jsonl"
    rows = [
        {"candidate_id": "ehm-20260619-0001", "critic_score": 4, "decision": "candidate", "tags": ["prompt_fit"]},
        {"candidate_id": "ehm-20260619-0002", "critic_score": 2, "decision": "reject", "tags": ["too_dense"]},
        {"candidate_id": "ehm-does-not-exist", "critic_score": 5, "decision": "candidate"},
    ]
    known = {"ehm-20260619-0001", "ehm-20260619-0002"}

    summary = import_critic_results(critic_path, rows, known_candidate_ids=known, now=_DAY)

    assert sorted(summary.imported) == ["ehm-20260619-0001", "ehm-20260619-0002"]
    assert summary.unknown == ["ehm-does-not-exist"]
    stored = latest_critic_by_candidate(critic_path)
    assert stored["ehm-20260619-0001"]["critic_score"] == 4
    assert stored["ehm-20260619-0001"]["source"] == "nyemos"
    assert stored["ehm-20260619-0001"]["tags"] == ["prompt_fit"]


def test_import_critic_results_reimport_overrides_and_flags_duplicate(tmp_path) -> None:
    critic_path = tmp_path / "critic.jsonl"
    known = {"ehm-1"}
    import_critic_results(critic_path, [{"candidate_id": "ehm-1", "critic_score": 2, "decision": "reject"}], known_candidate_ids=known, now=_DAY)
    summary = import_critic_results(critic_path, [{"candidate_id": "ehm-1", "critic_score": 5, "decision": "candidate"}], known_candidate_ids=known, now=_DAY)

    assert summary.duplicates == ["ehm-1"]
    # Latest record wins on read, but both are kept in the append-only log.
    assert latest_critic_by_candidate(critic_path)["ehm-1"]["critic_score"] == 5
    assert len([r for r in critic_path.read_text(encoding="utf-8").splitlines() if r.strip()]) == 2


# ── Phase 7 + 8: Browser join + Human queue ─────────────────────────────────────


def _full_pipeline(tmp_path):
    candidates_path = _seed_candidates(tmp_path)
    critic_path = tmp_path / "critic.jsonl"
    human_path = tmp_path / "human.jsonl"
    ids = [c["candidate_id"] for c in load_candidates(candidates_path)]
    import_critic_results(
        critic_path,
        [
            {"candidate_id": ids[0], "critic_score": 5, "decision": "candidate", "tags": ["balanced"]},
            {"candidate_id": ids[1], "critic_score": 3, "decision": "borderline", "tags": ["tempo_ok"]},
            {"candidate_id": ids[2], "critic_score": 2, "decision": "reject", "tags": ["too_dense"]},
        ],
        known_candidate_ids=set(ids),
        now=_DAY,
    )
    return candidates_path, critic_path, human_path, ids


def test_build_candidate_views_joins_all_three_streams(tmp_path) -> None:
    candidates_path, critic_path, human_path, ids = _full_pipeline(tmp_path)
    append_human_evaluation(human_path, candidate_id=ids[0], human_decision="use", human_rating=5, human_note="nice", now=_DAY)

    views = {v.candidate_id: v for v in build_candidate_views(candidates_path, critic_path, human_path)}

    assert views[ids[0]].critic_score == 5
    assert views[ids[0]].critic_decision == "candidate"
    assert views[ids[0]].critic_tags == ["balanced"]
    assert views[ids[0]].human_decision == "use"
    assert views[ids[0]].human_rating == 5
    assert views[ids[3]].critic_decision is None  # unreviewed candidate


def test_filter_candidate_views(tmp_path) -> None:
    candidates_path, critic_path, human_path, ids = _full_pipeline(tmp_path)
    views = build_candidate_views(candidates_path, critic_path, human_path)

    assert {v.candidate_id for v in filter_candidate_views(views, review_filter="unreviewed")} == {ids[3]}
    assert {v.candidate_id for v in filter_candidate_views(views, review_filter="candidate")} == {ids[0]}
    assert {v.candidate_id for v in filter_candidate_views(views, review_filter="reject")} == {ids[2]}
    assert {v.candidate_id for v in filter_candidate_views(views, review_filter="human_unreviewed")} == set(ids)


def test_human_review_queue_uses_critic_score_threshold(tmp_path) -> None:
    candidates_path, critic_path, human_path, ids = _full_pipeline(tmp_path)
    views = build_candidate_views(candidates_path, critic_path, human_path)

    # Default: score >= 3 and decision in {candidate, borderline}. The reject (2)
    # and the unreviewed candidate are excluded; logs still keep everything.
    queue_ids = {v.candidate_id for v in human_review_queue(views)}
    assert queue_ids == {ids[0], ids[1]}

    # Lowering the threshold surfaces the 1-2 scored candidate again.
    relaxed = human_review_queue(views, min_critic_score=1, allowed_decisions=("candidate", "borderline", "reject"))
    assert ids[2] in {v.candidate_id for v in relaxed}


def test_human_review_queue_excludes_already_human_reviewed(tmp_path) -> None:
    candidates_path, critic_path, human_path, ids = _full_pipeline(tmp_path)
    append_human_evaluation(human_path, candidate_id=ids[0], human_decision="use", human_rating=4, now=_DAY)
    views = build_candidate_views(candidates_path, critic_path, human_path)

    queue_ids = {v.candidate_id for v in human_review_queue(views)}
    assert queue_ids == {ids[1]}  # ids[0] dropped after human review


def test_append_human_evaluation_validates(tmp_path) -> None:
    path = tmp_path / "human.jsonl"
    with pytest.raises(ValueError):
        append_human_evaluation(path, candidate_id="x", human_decision="maybe", human_rating=3)
    with pytest.raises(ValueError):
        append_human_evaluation(path, candidate_id="x", human_decision="use", human_rating=9)

    record = append_human_evaluation(path, candidate_id="x", human_decision="use", human_rating=5, human_note="ok", now=_DAY)
    assert record["candidate_id"] == "x"
    assert record["human_decision"] == "use"
    assert record["human_rating"] == 5
