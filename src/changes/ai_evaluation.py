"""Evaluation loop for Emnyeca Harmony AI (beta).

This module owns the data layer that turns single-shot generation into a
reviewable pipeline:

    Prompt Library
        -> Batch Generate (N candidates per prompt)
        -> Candidate Store
        -> Critic Export  (JSON handed to the external critic "Nyemos"/ChatGPT)
        -> Critic Import  (Nyemos scores brought back in)
        -> Human Evaluation Queue (Emnyeca reviews the borderline cases)
        -> three accumulating log streams for future training

Intentional design notes (see eub/ai-governance):
  - No in-app critic AI. The critic is external on purpose, so we can measure the
    gap between the critic's judgement and Emnyeca's before automating anything.
  - Records are kept in plain JSONL stores keyed by stable ids. Candidate,
    critic, and human records live in *separate* append-only streams and are
    joined by ``candidate_id``; nothing mutates a candidate after creation. This
    keeps every stream a faithful, human-readable training source.
  - Functions take explicit store paths and an injectable ``generate_fn`` so the
    whole loop is testable without Streamlit or a running Ollama.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from changes.ai_generation import (
    AI_COMPOSER,
    AiGenerationError,
    AiGenerationResult,
    AiGenerationSettings,
    generate_harmony_from_ollama,
)
from changes.models.song_model import SongModel

# Seed prompts used to bootstrap a fresh Prompt Library during development. The
# library is empty by default; these are only added when explicitly requested.
SEED_PROMPTS: tuple[tuple[str, str], ...] = (
    ("元気で軽快", "upbeat"),
    ("シンプルなポップス", "pop"),
    ("雨の日のダウナーな気分", "downtempo"),
    ("神々の住まう領域", "ambient"),
    ("深夜の都会", "citypop"),
    ("暗いけど前に進む感じ", "drama"),
    ("静かな湖", "ambient"),
    ("壊れた遊園地", "dark"),
    ("少し不穏な祝祭", "dark"),
    ("夜明け前", "ambient"),
    ("機械都市", "electronic"),
    ("浮遊するジャズ", "jazz"),
    ("淡いフュージョン", "fusion"),
    ("踊れるアンビエント", "electronic"),
    ("ひとりで歩く帰り道", "drama"),
)

CRITIC_SOURCE_DEFAULT = "nyemos"


# ── JSONL helpers ──────────────────────────────────────────────────────────────


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    file = Path(path)
    if not file.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def _write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def _now(now: datetime | None) -> datetime:
    return now or datetime.now(timezone.utc)


def _timestamp(now: datetime | None) -> str:
    return _now(now).isoformat()


def _fraction_to_number(value: Any) -> float | int:
    try:
        frac = Fraction(str(value)).limit_denominator(1000)
    except Exception:
        return 0
    if frac.denominator == 1:
        return int(frac)
    return float(frac)


# ── Stable id allocation ────────────────────────────────────────────────────────

_PROMPT_ID_RE = re.compile(r"^prompt-(\d{8})-(\d{4,})$")
_CANDIDATE_ID_RE = re.compile(r"^ehm-(\d{8})-(\d{4,})$")


def _next_sequenced_id(prefix: str, pattern: re.Pattern[str], existing: Iterable[str], now: datetime | None) -> str:
    day = _now(now).strftime("%Y%m%d")
    highest = 0
    for value in existing:
        match = pattern.match(str(value))
        if match and match.group(1) == day:
            highest = max(highest, int(match.group(2)))
    return f"{prefix}-{day}-{highest + 1:04d}"


def next_prompt_id(existing: Iterable[str], now: datetime | None = None) -> str:
    return _next_sequenced_id("prompt", _PROMPT_ID_RE, existing, now)


def next_candidate_id(existing: Iterable[str], now: datetime | None = None) -> str:
    return _next_sequenced_id("ehm", _CANDIDATE_ID_RE, existing, now)


# ── Phase 1: Prompt Library ─────────────────────────────────────────────────────
#
# Small, edited store -> load-all / save-all JSONL (one prompt per line). Prompt
# records are mutable (edit / disable); ids are never reused.


@dataclass
class PromptEntry:
    prompt_id: str
    prompt: str
    category: str = ""
    enabled: bool = True
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "prompt": self.prompt,
            "category": self.category,
            "enabled": bool(self.enabled),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PromptEntry":
        return cls(
            prompt_id=str(data.get("prompt_id") or ""),
            prompt=str(data.get("prompt") or ""),
            category=str(data.get("category") or ""),
            enabled=bool(data.get("enabled", True)),
            timestamp=str(data.get("timestamp") or ""),
        )


def load_prompts(path: str | Path) -> list[PromptEntry]:
    return [PromptEntry.from_dict(record) for record in _read_jsonl(path)]


def save_prompts(path: str | Path, prompts: Sequence[PromptEntry]) -> None:
    _write_jsonl(path, (entry.to_dict() for entry in prompts))


def add_prompt(
    path: str | Path,
    prompt: str,
    *,
    category: str = "",
    enabled: bool = True,
    now: datetime | None = None,
) -> PromptEntry:
    text = str(prompt).strip()
    if not text:
        raise ValueError("Prompt text must not be empty.")
    prompts = load_prompts(path)
    entry = PromptEntry(
        prompt_id=next_prompt_id((p.prompt_id for p in prompts), now),
        prompt=text,
        category=str(category).strip(),
        enabled=bool(enabled),
        timestamp=_timestamp(now),
    )
    prompts.append(entry)
    save_prompts(path, prompts)
    return entry


def update_prompt(
    path: str | Path,
    prompt_id: str,
    *,
    prompt: str | None = None,
    category: str | None = None,
    enabled: bool | None = None,
) -> PromptEntry:
    prompts = load_prompts(path)
    target: PromptEntry | None = None
    for entry in prompts:
        if entry.prompt_id == prompt_id:
            target = entry
            break
    if target is None:
        raise KeyError(f"Unknown prompt_id: {prompt_id}")
    if prompt is not None:
        text = str(prompt).strip()
        if not text:
            raise ValueError("Prompt text must not be empty.")
        target.prompt = text
    if category is not None:
        target.category = str(category).strip()
    if enabled is not None:
        target.enabled = bool(enabled)
    save_prompts(path, prompts)
    return target


def set_prompt_enabled(path: str | Path, prompt_id: str, enabled: bool) -> PromptEntry:
    return update_prompt(path, prompt_id, enabled=enabled)


def enabled_prompts(path: str | Path) -> list[PromptEntry]:
    return [entry for entry in load_prompts(path) if entry.enabled]


def seed_prompt_library(
    path: str | Path,
    seeds: Sequence[tuple[str, str]] = SEED_PROMPTS,
    *,
    now: datetime | None = None,
) -> list[PromptEntry]:
    """Add development seed prompts. Existing prompt texts are skipped so the
    seed is idempotent and never duplicates a prompt Emnyeca already added."""
    prompts = load_prompts(path)
    existing_texts = {entry.prompt for entry in prompts}
    added: list[PromptEntry] = []
    for text, category in seeds:
        text = text.strip()
        if not text or text in existing_texts:
            continue
        entry = PromptEntry(
            prompt_id=next_prompt_id((p.prompt_id for p in prompts), now),
            prompt=text,
            category=category,
            enabled=True,
            timestamp=_timestamp(now),
        )
        prompts.append(entry)
        added.append(entry)
        existing_texts.add(text)
    if added:
        save_prompts(path, prompts)
    return added


# ── Phase 3 + 4: Candidate ids and Candidate Store ──────────────────────────────


def candidate_record_from_result(
    result: AiGenerationResult,
    *,
    candidate_id: str,
    prompt_id: str | None,
    category: str = "",
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build a Candidate Store record from a successful generation.

    The progression/tempo/meter are taken from the normalized SongModel so the
    stored candidate matches what the pipeline actually accepted, while the raw
    model output is preserved in ``generated_json`` for full traceability.
    """
    return {
        "candidate_id": candidate_id,
        "timestamp": _timestamp(now),
        "prompt_id": prompt_id,
        "category": category,
        "user_prompt": result.user_prompt,
        "tempo": _fraction_to_number(result.song.performance_tempo),
        "meter": _song_meter(result.song),
        "progression": _song_progression(result.song),
        "composer": result.song.composer or AI_COMPOSER,
        "model_name": result.model_name,
        "generated_json": result.generated_json,
    }


def _song_meter(song: SongModel) -> str:
    if song.measures:
        first = song.measures[0]
        return f"{first.meter_numerator}/{first.meter_denominator}"
    return "4/4"


def _song_progression(song: SongModel) -> list[dict[str, Any]]:
    progression: list[dict[str, Any]] = []
    for measure in song.measures:
        for harmony in measure.harmony:
            progression.append(
                {
                    "chord": harmony.symbol,
                    "beats": _fraction_to_number(harmony.duration_quarters),
                }
            )
    return progression


def append_candidate(path: str | Path, record: dict[str, Any]) -> None:
    _append_jsonl(path, record)


def load_candidates(path: str | Path) -> list[dict[str, Any]]:
    return _read_jsonl(path)


def candidate_ids(path: str | Path) -> set[str]:
    return {str(record.get("candidate_id")) for record in _read_jsonl(path) if record.get("candidate_id")}


# ── Phase 2: Batch generation ───────────────────────────────────────────────────


@dataclass
class BatchProgress:
    prompt_index: int
    prompt_total: int
    candidate_index: int
    candidate_total: int
    total_success: int
    total_failed: int
    current_prompt: str


@dataclass
class BatchSummary:
    total_success: int = 0
    total_failed: int = 0
    candidate_ids: list[str] = field(default_factory=list)


GenerateFn = Callable[..., AiGenerationResult]


def run_batch_generation(
    prompts: Sequence[PromptEntry],
    candidates_per_prompt: int,
    *,
    settings: AiGenerationSettings,
    candidate_store_path: str | Path,
    failure_log_path: str | Path | None = None,
    generate_fn: GenerateFn = generate_harmony_from_ollama,
    progress_cb: Callable[[BatchProgress], None] | None = None,
    now: datetime | None = None,
) -> BatchSummary:
    """Generate ``candidates_per_prompt`` candidates for each prompt.

    Successful candidates are appended to the Candidate Store with a fresh stable
    id. Failures are already written to the failure log by
    ``generate_harmony_from_ollama``; here they are only counted so a single bad
    prompt never aborts the batch.
    """
    per_prompt = max(0, int(candidates_per_prompt))
    summary = BatchSummary()
    used_ids = candidate_ids(candidate_store_path)
    prompt_total = len(prompts)

    for prompt_index, entry in enumerate(prompts, start=1):
        for candidate_index in range(1, per_prompt + 1):
            try:
                result = generate_fn(
                    entry.prompt,
                    settings,
                    failure_log_path=failure_log_path,
                )
            except AiGenerationError:
                summary.total_failed += 1
            except Exception:
                # Unexpected errors must not abort the batch either.
                summary.total_failed += 1
            else:
                candidate_id = next_candidate_id(used_ids, now)
                used_ids.add(candidate_id)
                record = candidate_record_from_result(
                    result,
                    candidate_id=candidate_id,
                    prompt_id=entry.prompt_id,
                    category=entry.category,
                    now=now,
                )
                append_candidate(candidate_store_path, record)
                summary.total_success += 1
                summary.candidate_ids.append(candidate_id)
            if progress_cb is not None:
                progress_cb(
                    BatchProgress(
                        prompt_index=prompt_index,
                        prompt_total=prompt_total,
                        candidate_index=candidate_index,
                        candidate_total=per_prompt,
                        total_success=summary.total_success,
                        total_failed=summary.total_failed,
                        current_prompt=entry.prompt,
                    )
                )
    return summary


# ── Phase 5: Critic Export ──────────────────────────────────────────────────────

# Fields handed to the external critic. Internal fields (model_name,
# generated_json, composer, timestamps) are deliberately withheld.
_CRITIC_EXPORT_FIELDS = ("candidate_id", "prompt_id", "prompt", "tempo", "meter", "progression")


def critic_export_rows(
    candidate_store_path: str | Path,
    *,
    reviewed_ids: Iterable[str] | None = None,
    prompt_id: str | None = None,
    category: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Return the minimal critic view for candidates needing a first pass.

    ``reviewed_ids`` (typically ``reviewed_candidate_ids(critic_log)``) are
    excluded so only unreviewed candidates are exported by default.
    """
    reviewed = {str(value) for value in (reviewed_ids or ())}
    rows: list[dict[str, Any]] = []
    for record in load_candidates(candidate_store_path):
        cid = str(record.get("candidate_id") or "")
        if not cid or cid in reviewed:
            continue
        if prompt_id is not None and record.get("prompt_id") != prompt_id:
            continue
        if category is not None and record.get("category") != category:
            continue
        timestamp = str(record.get("timestamp") or "")
        if date_from is not None and timestamp < date_from:
            continue
        if date_to is not None and timestamp > date_to:
            continue
        rows.append(
            {
                "candidate_id": cid,
                "prompt_id": record.get("prompt_id"),
                "prompt": record.get("user_prompt", ""),
                "tempo": record.get("tempo"),
                "meter": record.get("meter", "4/4"),
                "progression": record.get("progression", []),
            }
        )
        if limit is not None and len(rows) >= limit:
            break
    return rows


def format_critic_export(rows: Sequence[dict[str, Any]]) -> str:
    """Pretty JSON array, ready to paste into the external critic chat."""
    return json.dumps(list(rows), ensure_ascii=False, indent=2)


# ── Phase 6: Critic Import + Critic Log ─────────────────────────────────────────


@dataclass
class CriticImportSummary:
    imported: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)
    invalid: list[str] = field(default_factory=list)


def parse_critic_import(text: str) -> list[dict[str, Any]]:
    """Accept either a JSON array or JSONL of critic results."""
    text = (text or "").strip()
    if not text:
        return []
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        return [value]
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def reviewed_candidate_ids(critic_log_path: str | Path) -> set[str]:
    return {str(record.get("candidate_id")) for record in _read_jsonl(critic_log_path) if record.get("candidate_id")}


def latest_critic_by_candidate(critic_log_path: str | Path) -> dict[str, dict[str, Any]]:
    """Latest critic record wins (re-imports override earlier scores)."""
    latest: dict[str, dict[str, Any]] = {}
    for record in _read_jsonl(critic_log_path):
        cid = str(record.get("candidate_id") or "")
        if cid:
            latest[cid] = record
    return latest


def import_critic_results(
    critic_log_path: str | Path,
    rows: Sequence[dict[str, Any]],
    *,
    known_candidate_ids: Iterable[str],
    source: str = CRITIC_SOURCE_DEFAULT,
    now: datetime | None = None,
) -> CriticImportSummary:
    """Append critic results to the critic log, keyed by candidate_id.

    Unknown candidate_ids are reported (not written). Re-imports of an
    already-reviewed candidate are allowed and appended (latest wins on read),
    but flagged as duplicates so the human is aware.
    """
    known = {str(value) for value in known_candidate_ids}
    already = reviewed_candidate_ids(critic_log_path)
    summary = CriticImportSummary()
    for row in rows:
        cid = str(row.get("candidate_id") or "").strip()
        if not cid:
            summary.invalid.append(json.dumps(row, ensure_ascii=False))
            continue
        if cid not in known:
            summary.unknown.append(cid)
            continue
        if cid in already:
            summary.duplicates.append(cid)
        record = {
            "candidate_id": cid,
            "timestamp": _timestamp(now),
            "critic_score": row.get("critic_score"),
            "decision": row.get("decision"),
            "tags": list(row.get("tags") or []),
            "source": source,
        }
        _append_jsonl(critic_log_path, record)
        already.add(cid)
        summary.imported.append(cid)
    return summary


# ── Phase 8: Human Evaluation Log ───────────────────────────────────────────────


def append_human_evaluation(
    human_eval_log_path: str | Path,
    *,
    candidate_id: str,
    human_decision: str,
    human_rating: int | None,
    human_note: str = "",
    now: datetime | None = None,
) -> dict[str, Any]:
    if human_decision not in {"use", "reject"}:
        raise ValueError("human_decision must be 'use' or 'reject'")
    if human_rating is not None and not 1 <= int(human_rating) <= 5:
        raise ValueError("human_rating must be 1..5 or None")
    record = {
        "candidate_id": str(candidate_id),
        "timestamp": _timestamp(now),
        "human_decision": human_decision,
        "human_rating": int(human_rating) if human_rating is not None else None,
        "human_note": str(human_note or ""),
    }
    _append_jsonl(human_eval_log_path, record)
    return record


def latest_human_by_candidate(human_eval_log_path: str | Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in _read_jsonl(human_eval_log_path):
        cid = str(record.get("candidate_id") or "")
        if cid:
            latest[cid] = record
    return latest


# ── Phase 7: Candidate Browser (joined view) ────────────────────────────────────


@dataclass
class CandidateView:
    candidate_id: str
    prompt_id: str | None
    user_prompt: str
    category: str
    tempo: Any
    meter: str
    progression: list[dict[str, Any]]
    timestamp: str
    critic_score: Any = None
    critic_decision: str | None = None
    critic_tags: list[str] = field(default_factory=list)
    human_decision: str | None = None
    human_rating: int | None = None
    human_note: str = ""

    @property
    def has_critic(self) -> bool:
        return self.critic_decision is not None or self.critic_score is not None

    @property
    def has_human(self) -> bool:
        return self.human_decision is not None


def build_candidate_views(
    candidate_store_path: str | Path,
    critic_log_path: str | Path,
    human_eval_log_path: str | Path,
) -> list[CandidateView]:
    critic = latest_critic_by_candidate(critic_log_path)
    human = latest_human_by_candidate(human_eval_log_path)
    views: list[CandidateView] = []
    for record in load_candidates(candidate_store_path):
        cid = str(record.get("candidate_id") or "")
        if not cid:
            continue
        critic_record = critic.get(cid, {})
        human_record = human.get(cid, {})
        views.append(
            CandidateView(
                candidate_id=cid,
                prompt_id=record.get("prompt_id"),
                user_prompt=str(record.get("user_prompt") or ""),
                category=str(record.get("category") or ""),
                tempo=record.get("tempo"),
                meter=str(record.get("meter") or "4/4"),
                progression=list(record.get("progression") or []),
                timestamp=str(record.get("timestamp") or ""),
                critic_score=critic_record.get("critic_score"),
                critic_decision=critic_record.get("decision"),
                critic_tags=list(critic_record.get("tags") or []),
                human_decision=human_record.get("human_decision"),
                human_rating=human_record.get("human_rating"),
                human_note=str(human_record.get("human_note") or ""),
            )
        )
    return views


def filter_candidate_views(
    views: Sequence[CandidateView],
    *,
    review_filter: str = "all",
    category: str | None = None,
) -> list[CandidateView]:
    """Browser filters. ``review_filter`` is one of:
    all, unreviewed, candidate, borderline, reject, human_unreviewed, human_reviewed.
    """
    out: list[CandidateView] = []
    for view in views:
        if category is not None and view.category != category:
            continue
        if review_filter == "unreviewed" and view.has_critic:
            continue
        if review_filter in {"candidate", "borderline", "reject"} and view.critic_decision != review_filter:
            continue
        if review_filter == "human_unreviewed" and view.has_human:
            continue
        if review_filter == "human_reviewed" and not view.has_human:
            continue
        out.append(view)
    return out


def sort_candidate_views(
    views: Sequence[CandidateView],
    *,
    key: str = "timestamp",
    descending: bool = True,
) -> list[CandidateView]:
    def sort_key(view: CandidateView) -> Any:
        if key == "critic_score":
            return (view.critic_score if isinstance(view.critic_score, (int, float)) else -1)
        if key == "prompt":
            return view.user_prompt
        if key == "category":
            return view.category
        return view.timestamp

    return sorted(views, key=sort_key, reverse=descending)


# ── Phase 8: Human Review Queue ─────────────────────────────────────────────────


def human_review_queue(
    views: Sequence[CandidateView],
    *,
    min_critic_score: int = 3,
    allowed_decisions: Iterable[str] = ("candidate", "borderline"),
    exclude_human_reviewed: bool = True,
) -> list[CandidateView]:
    """Candidates worth Emnyeca's time, filtered by the external critic.

    A candidate qualifies when its critic_score is at least ``min_critic_score``
    AND (no decision is recorded OR the decision is in ``allowed_decisions``).
    Low-scored candidates stay in the logs but are kept out of the queue. Set
    ``min_critic_score`` to a low value to surface 1-2 again. We never
    auto-reject here: the goal is still to measure the critic-vs-human gap.
    """
    allowed = {str(value) for value in allowed_decisions}
    out: list[CandidateView] = []
    for view in views:
        if exclude_human_reviewed and view.has_human:
            continue
        score = view.critic_score
        if not isinstance(score, (int, float)) or score < min_critic_score:
            continue
        if view.critic_decision is not None and view.critic_decision not in allowed:
            continue
        out.append(view)
    return out
