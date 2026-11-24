# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the lane's SFT store: chat rows, per-row metadata, decontamination, manifest.

Rows follow the repo's chat conventions: ``openai_chat_document`` serializes the
conversation into Harmony messages (the trace's reasoning into the analysis channel, which
the Marin chat template renders as ``<|start_think|>…<|end_think|>``) under ``CHAT_SCHEMA``;
the lane's metadata rides beside it the way the repo's relay-trace rows carry theirs. The
decontamination pass is the lexical method the Nemotron-overthinking study used against the
RL blend prompt set: word 13-grams, stride zero, alphabetic-token filter, with boilerplate
suppression and per-prompt coverage thresholds.
"""

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
from marin.datakit.chat_normalize import CHAT_SCHEMA
from marin.datakit.download.rollout_transforms import openai_chat_document
from rigging.filesystem.factory import open_url
from rigging.filesystem.storage_path import prefix_join

from experiments.post_training.task_curation import CANONICAL_ARTIFACT_ROOT, publish_once
from experiments.post_training.task_curation.balanced_reasoning.difficulty import url_protocol
from experiments.post_training.task_curation.balanced_reasoning.lane import (
    AttemptTrace,
    TaskOutcome,
    outcome_from_document,
)
from experiments.post_training.task_curation.balanced_reasoning.selection import read_selection

logger = logging.getLogger(__name__)

BALANCED_SOURCE = "balanced-reasoning-v1"
STORE_VERSION = "2026.11.22.1"

NGRAM_LENGTH = 13
"""Ngram length in whitespace-split word tokens."""
BOILERPLATE_MIN_PROMPTS = 3
"""Features present in at least this many blend prompts are boilerplate and never match."""
MIN_MATCHED_FEATURES = 8
"""Distinct shared ngrams that constitute content overlap with one blend prompt."""
MIN_COVERAGE = 0.2
"""Fraction of a blend prompt's specific features a row must share for a hit."""


def _has_alpha(token: str) -> bool:
    return any(char.isalpha() for char in token)


def word_ngrams(text: str, n: int = NGRAM_LENGTH, stride: int = 0) -> frozenset[str]:
    """Word n-grams over alphabetic tokens, stride 0."""
    tokens = [token for token in text.split() if _has_alpha(token)]
    starts = range(0, len(tokens) - n + 1) if stride <= 0 else range(0, len(tokens) - n + 1, stride)
    return frozenset(" ".join(tokens[index : index + n]) for index in starts)


@dataclass(frozen=True, slots=True)
class BlendFilter:
    """The RL blend prompt feature set, with per-prompt specific features."""

    prompts: tuple[str, ...]
    features_by_prompt: tuple[frozenset[str], ...]

    @classmethod
    def from_prompts(cls, prompts: Iterable[str]) -> "BlendFilter":
        documents = tuple(prompts)
        counts: dict[str, int] = {}
        raw = []
        for prompt in documents:
            features = word_ngrams(prompt)
            raw.append(features)
            for feature in features:
                counts[feature] = counts.get(feature, 0) + 1
        boilerplate = {feature for feature, count in counts.items() if count >= BOILERPLATE_MIN_PROMPTS}
        specific = tuple(features - boilerplate for features in raw)
        kept = [(prompt, features) for prompt, features in zip(documents, specific, strict=True) if features]
        logger.info(
            "Blend filter: %d prompts, %d boilerplate features, %d prompts with specific features",
            len(documents),
            len(boilerplate),
            len(kept),
        )
        return cls(tuple(prompt for prompt, _ in kept), tuple(features for _, features in kept))

    @classmethod
    def from_jsonl(cls, path: str) -> "BlendFilter":
        """Read a blend prompt file: one JSON object per line with a ``prompt`` field."""
        with open_url(path, "rt") as source:
            prompts = [json.loads(line)["prompt"] for line in source if line.strip()]
        return cls.from_prompts(prompts)

    def hit(self, prompt: str) -> str | None:
        """The first blend prompt the row's prompt overlaps with, or ``None``."""
        row_features = word_ngrams(prompt)
        if not row_features:
            return None
        for name, features in zip(self.prompts, self.features_by_prompt, strict=True):
            shared = row_features & features
            if len(shared) >= MIN_MATCHED_FEATURES and len(shared) / len(features) >= MIN_COVERAGE:
                return name
        return None


def _message_role(message: dict) -> str | None:
    """The role of a serialized chat message, in either wire or Harmony shape."""
    role = message.get("role")
    if isinstance(role, str):
        return role
    author = message.get("author")
    if isinstance(author, dict):
        author_role = author.get("role")
        return author_role if isinstance(author_role, str) else None
    return None


def _message_text(message: dict) -> str:
    """The text content of a serialized chat message, in either wire or Harmony shape."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [part.get("text", "") for part in content if isinstance(part, dict)]
        return " ".join(part for part in parts if part)
    return ""


def conversation_prompt(messages: list[dict]) -> str:
    """The text a decontamination decision is made on: every non-assistant turn."""
    parts = []
    for message in messages:
        if _message_role(message) != "assistant":
            text = _message_text(message)
            if text:
                parts.append(text)
    return "\n".join(parts)


def row_metadata(outcome: TaskOutcome, attempt: AttemptTrace, receipt_uri: str, attempt_index: int) -> dict:
    """The lane's per-row metadata columns: identity, banding, budget, grade and generation."""
    return {
        "atlas_task_id": outcome.task_id,
        "task_sha256": outcome.task_sha256,
        "source_id": outcome.source_id,
        "domain": outcome.domain,
        "band": outcome.band,
        "difficulty": outcome.solve_rate,
        "difficulty_basis": outcome.difficulty_basis,
        "budget_cap_tokens": outcome.budget_cap_tokens,
        "attempts": outcome.attempts_requested,
        "attempt_index": attempt_index,
        "resolution_kind": outcome.resolution_kind,
        "grader_kind": outcome.grader_kind,
        "reward": attempt.reward,
        "verified": attempt.grade_passed is True,
        "verification_receipt": receipt_uri,
        "assistant_tokens": attempt.completion_tokens,
        "prompt_tokens": attempt.prompt_tokens,
        "reasoning_tokens": attempt.reasoning_tokens,
        "model": outcome.model,
        "temperature": outcome.temperature,
        "policy_digest": outcome.policy_digest,
    }


def outcome_rows(outcome: TaskOutcome, *, receipt_root: str, blend: BlendFilter | None) -> tuple[list[dict], str | None]:
    """The store rows one outcome contributes, and why it contributes none.

    A solved outcome contributes its selected trace; a graceful outcome its graceful trace.
    Rows whose prompt overlaps the RL blend set are dropped here and counted in the manifest.
    """
    if outcome.selected_index is None:
        return [], outcome.unsolved_reason or "no selected attempt"
    attempt = outcome.attempts[outcome.selected_index]
    receipt_uri = prefix_join(receipt_root, outcome.task_id)
    document = openai_chat_document(
        outcome.task_prompt_messages(attempt),
        BALANCED_SOURCE,
        **row_metadata(outcome, attempt, receipt_uri, outcome.selected_index),
    )
    if blend is not None:
        hit = blend.hit(conversation_prompt(document["messages"]))
        if hit is not None:
            logger.info("DROP %s: prompt overlaps blend prompt %s", outcome.task_id, hit)
            return [], f"blend overlap: {hit}"
    document["decontamination"] = "clean"
    return [document], None


_METADATA_FIELDS: tuple[tuple[str, pa.DataType], ...] = (
    ("atlas_task_id", pa.string()),
    ("task_sha256", pa.string()),
    ("domain", pa.string()),
    ("band", pa.string()),
    ("difficulty", pa.float64()),
    ("difficulty_basis", pa.string()),
    ("budget_cap_tokens", pa.int64()),
    ("attempts", pa.int64()),
    ("attempt_index", pa.int64()),
    ("resolution_kind", pa.string()),
    ("grader_kind", pa.string()),
    ("reward", pa.float64()),
    ("verified", pa.bool_()),
    ("verification_receipt", pa.string()),
    ("assistant_tokens", pa.int64()),
    ("prompt_tokens", pa.int64()),
    ("reasoning_tokens", pa.int64()),
    ("model", pa.string()),
    ("temperature", pa.float64()),
    ("policy_digest", pa.string()),
    ("decontamination", pa.string()),
)

BALANCED_CHAT_SCHEMA = pa.schema([*CHAT_SCHEMA, *(pa.field(name, kind) for name, kind in _METADATA_FIELDS)])
"""The store's row schema: the standard chat schema plus the lane's metadata columns."""


def build_store(
    run_root: str,
    *,
    selection_path: str,
    blend: BlendFilter | None = None,
) -> dict:
    """Assemble the store rows from a run root's outcomes and write a local bundle.

    Rows and receipts land under ``<run_root>/store/``; ``publish_store`` uploads them to
    the versioned artifact prefix. Returns the manifest document.
    """
    digest_value, selected, census = read_selection(selection_path)
    receipt_root = prefix_join(run_root, "tasks")
    store_dir = f"{run_root.rstrip('/')}/store"
    fsspec.filesystem(url_protocol(run_root)).makedirs(store_dir, exist_ok=True)
    rows: list[dict] = []
    dropped = 0
    drop_reasons: dict[str, int] = {}
    resolutions: dict[str, int] = {}
    for entry in selected:
        attempt_path = f"{run_root.rstrip('/')}/tasks/{entry.task.task_id}/attempt.json"
        try:
            with open_url(attempt_path, "rt") as source:
                document = source.read()
        except FileNotFoundError:
            logger.warning("SKIP %s: no attempt record at %s", entry.task.task_id, attempt_path)
            continue
        outcome = outcome_from_document(json.loads(document))
        contributed, drop_reason = outcome_rows(outcome, receipt_root=receipt_root, blend=blend)
        rows.extend(contributed)
        if drop_reason is not None:
            dropped += 1
            drop_reasons[drop_reason] = drop_reasons.get(drop_reason, 0) + 1
        resolutions[outcome.resolution_kind] = resolutions.get(outcome.resolution_kind, 0) + 1
    table = pa.Table.from_pylist(rows, schema=BALANCED_CHAT_SCHEMA)
    rows_path = f"{store_dir}/rows.parquet"
    with fsspec.open(rows_path, "wb") as sink:
        pq.write_table(table, sink)
    receipts_path = f"{store_dir}/receipts.jsonl"
    with open_url(receipts_path, "wt") as sink:
        for row in rows:
            sink.write(
                json.dumps({"id": row["id"], "verification_receipt": row["verification_receipt"]}, sort_keys=True) + "\n"
            )
    rows_bytes = read_bytes(rows_path)
    receipts_bytes = read_bytes(receipts_path)
    manifest = {
        "version": STORE_VERSION,
        "source": BALANCED_SOURCE,
        "selection_digest": digest_value,
        "rows": len(rows),
        "dropped_by_decontamination": dropped,
        "drop_reasons": drop_reasons,
        "resolutions": resolutions,
        "selection_census": census,
        "schema": "marin CHAT_SCHEMA + balanced-reasoning metadata columns (BALANCED_CHAT_SCHEMA)",
        "consumption": (
            "A versioned source repair: consume rows.parquet through the Atlas queue as a new "
            "private dataset version; do not merge the lane's code to use it."
        ),
        "artifacts": {
            "rows": {"sha256": sha256(rows_bytes).hexdigest(), "bytes": len(rows_bytes)},
            "receipts": {"sha256": sha256(receipts_bytes).hexdigest()},
        },
        "blend_filter": (
            None
            if blend is None
            else {
                "prompts": len(blend.prompts),
                "method": (
                    f"word {NGRAM_LENGTH}-grams, stride 0, alphabetic filter; "
                    f"boilerplate >= {BOILERPLATE_MIN_PROMPTS} prompts; "
                    f"hit requires >= {MIN_MATCHED_FEATURES} shared features and >= {MIN_COVERAGE:.0%} coverage"
                ),
            }
        ),
    }
    manifest_path = f"{store_dir}/manifest.json"
    with open_url(manifest_path, "wt") as sink:
        sink.write(json.dumps(manifest, indent=2, sort_keys=True))
    logger.info("STORE %s rows=%d dropped=%d resolutions=%s", rows_path, len(rows), dropped, sorted(resolutions.items()))
    return manifest


def read_bytes(path: str) -> bytes:
    """Read a small object whole (store bundles only; never a population dump)."""
    with fsspec.open(path, "rb") as source:
        return source.read()


def publish_store(store_dir: str, destination: str) -> list[str]:
    """Upload the built store files to the versioned artifact prefix; returns the URIs."""
    prefix = prefix_join(CANONICAL_ARTIFACT_ROOT, destination.strip("/"))
    files = {name: read_bytes(f"{store_dir.rstrip('/')}/{name}") for name in ("rows.parquet", "receipts.jsonl", "manifest.json")}
    publish_once(prefix, files)
    return [prefix_join(prefix, name) for name in files]
