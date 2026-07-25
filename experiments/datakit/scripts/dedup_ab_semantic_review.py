# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run restart-safe, exhaustive semantic adjudication of fuzzy-dedup pairs."""

import argparse
import asyncio
import hashlib
import inspect
import json
import logging
from collections import Counter
from functools import cache
from typing import Any, Literal

import pyarrow as pa
import pyarrow.parquet as pq
from marin.inference.iris import remote_inference
from openai import AsyncOpenAI
from pydantic import BaseModel
from rigging.filesystem import StoragePath
from rigging.log_setup import configure_logging

from experiments.datakit.scripts.dedup_ab_machine_labels import DedupMachineLabelsData
from experiments.datakit.scripts.dedup_ab_semantic_batch import _records, load_semantic_cases
from experiments.datakit.scripts.dedup_ab_semantic_judge import (
    CANONICAL_CHUNKS_PER_MEMBER,
    CHUNK_CHARS,
    CHUNK_OVERLAP_CHARS,
    MAX_CONCURRENT_REQUESTS,
    MAX_DIRECT_CHARS,
    MODEL_ID,
    SYSTEM_PROMPT,
    VERDICT_SCHEMA,
    ModelVerdict,
    _inference_config,
    _judge_prompt,
    chunk_review_units,
    direct_pair_prompt,
    normalized_verdict,
)

REVIEW_PROTOCOL_VERSION = "v1"
INITIAL_PASSES = ("loss", "duplication")
TIEBREAK_PASS = "tiebreak"
OUTCOME_STATUSES = frozenset({"resolved", "unresolved"})
LABELS = frozenset({"false_positive", "true_duplicate"})
IDENTITY_FIELDS = (
    "review_key",
    "variant",
    "member_source_main_dir",
    "member_basename",
    "member_id",
    "canonical_source_main_dir",
    "canonical_basename",
    "canonical_id",
    "raw_sha256",
    "canonical_raw_sha256",
    "pair_path",
    "pair_row_index",
)

logger = logging.getLogger(__name__)


class SemanticBatchManifest(BaseModel):
    """Completion marker for one immutable, hash-verified outcome shard."""

    version: str = "v1"
    model: str
    machine_labels_path: str
    decision_file: str
    decision_file_index: int
    semantic_offset: int
    next_semantic_offset: int
    total_semantic_in_file: int
    cases: int
    case_keys_sha256: str
    config_sha256: str
    resolved_pairs: int
    unresolved_pairs: int
    direct_pairs: int
    chunked_pairs: int
    request_attempts: int
    outcome_path: str
    outcome_bytes: int
    outcome_sha256: str


class SemanticReviewData(BaseModel):
    """Exact aggregate metadata for a complete decision-file range."""

    version: str = "v1"
    model: str
    machine_labels_path: str
    output_root: str
    decision_file_start: int
    decision_file_stop: int
    decision_files: int
    expected_pairs: int
    completed_pairs: int
    batch_manifests: list[str]
    counters: dict[str, int]


def _compact_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


@cache
def review_config_sha256(
    *,
    chunk_chars: int,
    overlap_chars: int,
    canonical_chunks_per_member: int,
) -> str:
    """Hash every prompt and review parameter that affects persisted decisions."""
    payload = {
        "protocol": REVIEW_PROTOCOL_VERSION,
        "system_prompt": SYSTEM_PROMPT,
        "verdict_schema": VERDICT_SCHEMA,
        "max_direct_chars": MAX_DIRECT_CHARS,
        "initial_passes": INITIAL_PASSES,
        "tiebreak_pass": TIEBREAK_PASS,
        "chunk_chars": chunk_chars,
        "overlap_chars": overlap_chars,
        "canonical_chunks_per_member": canonical_chunks_per_member,
        "low_confidence_is_unresolved": True,
        "direct_prompt_source": inspect.getsource(direct_pair_prompt),
        "direct_review_prompt_source": inspect.getsource(_direct_prompt),
        "chunk_plan_source": inspect.getsource(chunk_review_units),
        "chunk_prompt_source": inspect.getsource(_chunk_prompt),
        "aggregation_source": inspect.getsource(outcome_from_evidence),
    }
    return hashlib.sha256(_compact_json(payload).encode()).hexdigest()


def _chunk_prompt(unit: dict[str, Any], *, pass_name: str) -> str:
    if pass_name == "loss":
        instruction = (
            "Inspect the complete MEMBER chunk for any substantive content absent "
            "from all supplied CANONICAL candidates."
        )
    elif pass_name == "duplication":
        instruction = (
            "Try to establish that every substantive part of the MEMBER chunk is "
            "represented by the supplied CANONICAL candidates, then reject that "
            "claim if any distinct payload remains."
        )
    elif pass_name == TIEBREAK_PASS:
        instruction = (
            "Independently decide the directional deletion question for this chunk. "
            "Apply every audit boundary exactly, and use low confidence only when "
            "the supplied canonical candidates are insufficient."
        )
    else:
        raise ValueError(f"Unknown semantic pass {pass_name!r}")
    return f"{instruction}\n\n{unit['prompt']}"


def _direct_prompt(case: dict[str, Any], *, pass_name: str) -> str:
    if pass_name in INITIAL_PASSES:
        return direct_pair_prompt(case, pass_name=pass_name)
    if pass_name != TIEBREAK_PASS:
        raise ValueError(f"Unknown semantic pass {pass_name!r}")
    return (
        "Independently decide whether deleting MEMBER while keeping CANONICAL loses "
        "substantive content. Apply every audit boundary exactly, and use low "
        "confidence only if the complete texts are genuinely insufficient.\n\n"
        + direct_pair_prompt(case, pass_name="loss")
    )


def _judgment_label(judgments: list[dict[str, Any]]) -> str | None:
    pass_names = tuple(judgment.get("pass") for judgment in judgments)
    if pass_names not in {INITIAL_PASSES, (*INITIAL_PASSES, TIEBREAK_PASS)}:
        return None
    verdicts: list[dict[str, Any]] = []
    for judgment in judgments:
        verdict = judgment.get("verdict")
        if verdict is None:
            return None
        verdicts.append(verdict)
    for verdict in verdicts:
        model_verdict = ModelVerdict.model_validate(verdict)
        if normalized_verdict(model_verdict) != verdict:
            raise AssertionError("Persisted semantic verdict does not match its deletion-loss decision")
    votes = Counter(verdict["label"] for verdict in verdicts if verdict["confidence"] != "low")
    winners = [label for label, count in votes.items() if count >= 2]
    if len(winners) != 1:
        return None
    return winners[0]


def _covered_chars(ranges: list[tuple[int, int]], text_chars: int) -> int:
    if not ranges:
        raise AssertionError("Semantic evidence contains no member ranges")
    ordered = sorted(ranges)
    if ordered[0][0] != 0 or ordered[-1][1] != text_chars:
        raise AssertionError(f"Member ranges do not span 0..{text_chars}: {ordered[0]}..{ordered[-1]}")

    covered = 0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start > current_end:
            raise AssertionError(f"Gap in member coverage at {current_end}..{start}")
        current_end = max(current_end, end)
    covered += current_end - current_start
    return covered


def _joined_basis(judgments: list[dict[str, Any]]) -> str:
    bases = []
    for judgment in judgments:
        verdict = judgment.get("verdict")
        if verdict is not None and verdict["basis"] not in bases:
            bases.append(verdict["basis"])
    return " | ".join(bases)


def _identity(case: dict[str, Any]) -> dict[str, Any]:
    return {field: case[field] for field in IDENTITY_FIELDS}


def outcome_from_evidence(case: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    """Derive one pair outcome deterministically from persisted model evidence."""
    mode = evidence.get("mode")
    if mode not in {"direct", "chunked"}:
        raise AssertionError(f"Unknown semantic review mode {mode!r}")

    request_attempts = 0
    if mode == "direct":
        judgments = evidence["judgments"]
        label = _judgment_label(judgments)
        request_attempts = sum(len(judgment["attempts"]) for judgment in judgments)
        member_chunks = 1
        canonical_chunks_scanned = 1
        covered_member_chars = len(case["member_text"])
        if label is None:
            basis = "Complete-text two-pass review was invalid, low-confidence, or non-unanimous."
        else:
            basis = f"Two complete-text reviews agreed. {_joined_basis(judgments)}"
    else:
        units = evidence["units"]
        ranges = [(int(unit["member_start"]), int(unit["member_end"])) for unit in units]
        covered_member_chars = _covered_chars(ranges, len(case["member_text"]))
        member_chunks = len(units)
        canonical_chunks_scanned = int(evidence["canonical_chunks_scanned"])
        unit_labels = []
        for unit in units:
            judgments = unit["judgments"]
            request_attempts += sum(len(judgment["attempts"]) for judgment in judgments)
            unit_labels.append(_judgment_label(judgments))
        unique_units = [
            unit for unit, unit_label in zip(units, unit_labels, strict=True) if unit_label == "false_positive"
        ]
        unresolved_chunks = sum(unit_label is None for unit_label in unit_labels)
        if unique_units:
            label = "false_positive"
            examples = " | ".join(_joined_basis(unit["judgments"]) for unit in unique_units[:4])
            basis = (
                f"Exhaustive chunk review found distinct member content in "
                f"{len(unique_units)}/{member_chunks} chunks. {examples}"
            )
        elif unresolved_chunks:
            label = None
            basis = (
                f"Exhaustive chunk review left {unresolved_chunks}/{member_chunks} member chunks "
                "invalid, low-confidence, or non-unanimous."
            )
        else:
            label = "true_duplicate"
            basis = (
                f"Both passes found all {member_chunks} overlapping member chunks represented "
                f"after scanning all {canonical_chunks_scanned} canonical chunks."
            )

    status = "resolved" if label is not None else "unresolved"
    chunk_chars = int(evidence["chunk_chars"])
    overlap_chars = int(evidence["overlap_chars"])
    canonical_chunks_per_member = int(evidence["canonical_chunks_per_member"])
    return {
        **_identity(case),
        "status": status,
        "label": label or "",
        "method": "semantic",
        "basis": basis,
        "review_mode": mode,
        "config_sha256": review_config_sha256(
            chunk_chars=chunk_chars,
            overlap_chars=overlap_chars,
            canonical_chunks_per_member=canonical_chunks_per_member,
        ),
        "member_chars": len(case["member_text"]),
        "canonical_chars": len(case["canonical_text"]),
        "member_chunks": member_chunks,
        "canonical_chunks_scanned": canonical_chunks_scanned,
        "covered_member_chars": covered_member_chars,
        "request_attempts": request_attempts,
        "judgments_json": _compact_json(evidence),
    }


async def _review_case(
    client: AsyncOpenAI,
    *,
    model: str,
    case: dict[str, Any],
    semaphore: asyncio.Semaphore,
    force_mode: Literal["direct", "chunked"] | None,
    chunk_chars: int,
    overlap_chars: int,
    canonical_chunks_per_member: int,
) -> dict[str, Any]:
    combined_chars = len(case["member_text"]) + len(case["canonical_text"])
    mode = force_mode or ("direct" if combined_chars <= MAX_DIRECT_CHARS else "chunked")
    if mode == "direct" and combined_chars > MAX_DIRECT_CHARS:
        raise ValueError(f"Direct review exceeds {MAX_DIRECT_CHARS} characters for {case['review_key']}")

    if mode == "direct":
        tasks = [
            _judge_prompt(
                client,
                model=model,
                prompt=_direct_prompt(case, pass_name=pass_name),
                semaphore=semaphore,
            )
            for pass_name in INITIAL_PASSES
        ]
        judgments = list(await asyncio.gather(*tasks))
        for pass_name, judgment in zip(INITIAL_PASSES, judgments, strict=True):
            judgment["pass"] = pass_name
        if _judgment_label(judgments) is None:
            tiebreak = await _judge_prompt(
                client,
                model=model,
                prompt=_direct_prompt(case, pass_name=TIEBREAK_PASS),
                semaphore=semaphore,
            )
            tiebreak["pass"] = TIEBREAK_PASS
            judgments.append(tiebreak)
        evidence = {
            "mode": "direct",
            "chunk_chars": chunk_chars,
            "overlap_chars": overlap_chars,
            "canonical_chunks_per_member": canonical_chunks_per_member,
            "judgments": judgments,
        }
        return outcome_from_evidence(case, evidence)

    units = chunk_review_units(
        case,
        chunk_chars=chunk_chars,
        overlap_chars=overlap_chars,
        canonical_chunks_per_member=canonical_chunks_per_member,
    )
    tasks = [
        _judge_prompt(
            client,
            model=model,
            prompt=_chunk_prompt(unit, pass_name=pass_name),
            semaphore=semaphore,
        )
        for unit in units
        for pass_name in INITIAL_PASSES
    ]
    raw_judgments = list(await asyncio.gather(*tasks))
    evidence_units = []
    pass_count = len(INITIAL_PASSES)
    for unit_index, unit in enumerate(units):
        judgments = raw_judgments[unit_index * pass_count : unit_index * pass_count + pass_count]
        for pass_name, judgment in zip(INITIAL_PASSES, judgments, strict=True):
            judgment["pass"] = pass_name
        evidence_units.append(
            {
                "member_chunk_index": unit["member_chunk_index"],
                "member_start": unit["member_start"],
                "member_end": unit["member_end"],
                "canonical_chunk_indices": unit["canonical_chunk_indices"],
                "judgments": judgments,
            }
        )
    unresolved_unit_indices = [
        unit_index for unit_index, unit in enumerate(evidence_units) if _judgment_label(unit["judgments"]) is None
    ]
    tiebreaks = list(
        await asyncio.gather(
            *[
                _judge_prompt(
                    client,
                    model=model,
                    prompt=_chunk_prompt(units[unit_index], pass_name=TIEBREAK_PASS),
                    semaphore=semaphore,
                )
                for unit_index in unresolved_unit_indices
            ]
        )
    )
    for unit_index, tiebreak in zip(unresolved_unit_indices, tiebreaks, strict=True):
        tiebreak["pass"] = TIEBREAK_PASS
        evidence_units[unit_index]["judgments"].append(tiebreak)
    evidence = {
        "mode": "chunked",
        "chunk_chars": chunk_chars,
        "overlap_chars": overlap_chars,
        "canonical_chunks_per_member": canonical_chunks_per_member,
        "canonical_chunks_scanned": units[0]["canonical_chunks_scanned"],
        "units": evidence_units,
    }
    return outcome_from_evidence(case, evidence)


async def review_cases(
    client: AsyncOpenAI,
    *,
    model: str,
    cases: list[dict[str, Any]],
    force_mode: Literal["direct", "chunked"] | None = None,
    chunk_chars: int = CHUNK_CHARS,
    overlap_chars: int = CHUNK_OVERLAP_CHARS,
    canonical_chunks_per_member: int = CANONICAL_CHUNKS_PER_MEMBER,
) -> list[dict[str, Any]]:
    """Review every case with bounded inference concurrency and complete evidence."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    return list(
        await asyncio.gather(
            *[
                _review_case(
                    client,
                    model=model,
                    case=case,
                    semaphore=semaphore,
                    force_mode=force_mode,
                    chunk_chars=chunk_chars,
                    overlap_chars=overlap_chars,
                    canonical_chunks_per_member=canonical_chunks_per_member,
                )
                for case in cases
            ]
        )
    )


def validate_outcome(case: dict[str, Any], outcome: dict[str, Any]) -> None:
    """Recompute an outcome from evidence and require an exact persisted match."""
    evidence = json.loads(outcome["judgments_json"])
    if evidence["mode"] == "direct":
        combined_chars = len(case["member_text"]) + len(case["canonical_text"])
        if combined_chars > MAX_DIRECT_CHARS:
            raise AssertionError(
                f"Oversized pair used direct review for {case['review_key']}: " f"{combined_chars}/{MAX_DIRECT_CHARS}"
            )
    else:
        expected_units = chunk_review_units(
            case,
            chunk_chars=int(evidence["chunk_chars"]),
            overlap_chars=int(evidence["overlap_chars"]),
            canonical_chunks_per_member=int(evidence["canonical_chunks_per_member"]),
        )
        actual_units = evidence["units"]
        if len(actual_units) != len(expected_units):
            raise AssertionError(
                f"Chunk count differs for {case['review_key']}: " f"{len(actual_units)}/{len(expected_units)}"
            )
        if int(evidence["canonical_chunks_scanned"]) != expected_units[0]["canonical_chunks_scanned"]:
            raise AssertionError(f"Canonical scan count differs for {case['review_key']}")
        plan_fields = (
            "member_chunk_index",
            "member_start",
            "member_end",
            "canonical_chunk_indices",
        )
        for expected_unit, actual_unit in zip(expected_units, actual_units, strict=True):
            mismatches = [field for field in plan_fields if expected_unit[field] != actual_unit.get(field)]
            if mismatches:
                raise AssertionError(
                    f"Chunk plan differs for {case['review_key']} "
                    f"unit {expected_unit['member_chunk_index']}: {mismatches}"
                )
    expected = outcome_from_evidence(case, evidence)
    if outcome != expected:
        mismatches = sorted(key for key in expected.keys() | outcome.keys() if expected.get(key) != outcome.get(key))
        raise AssertionError(f"Semantic outcome differs from evidence for {case['review_key']}: {mismatches}")
    if outcome["status"] not in OUTCOME_STATUSES:
        raise AssertionError(f"Invalid outcome status for {case['review_key']}: {outcome['status']}")
    if outcome["status"] == "resolved" and outcome["label"] not in LABELS:
        raise AssertionError(f"Invalid resolved label for {case['review_key']}: {outcome['label']}")
    if outcome["covered_member_chars"] != len(case["member_text"]):
        raise AssertionError(
            f"Incomplete member coverage for {case['review_key']}: "
            f"{outcome['covered_member_chars']}/{len(case['member_text'])}"
        )


def _parquet_bytes(outcomes: list[dict[str, Any]]) -> bytes:
    sink = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pylist(outcomes), sink, compression="zstd")
    return sink.getvalue().to_pybytes()


def _parquet_rows(payload: bytes) -> list[dict[str, Any]]:
    return pq.read_table(pa.BufferReader(payload)).to_pylist()


def _case_keys_sha256(cases: list[dict[str, Any]]) -> str:
    keys = "\n".join(case["review_key"] for case in cases)
    return hashlib.sha256(keys.encode()).hexdigest()


def batch_paths(output_root: str, *, decision_file_index: int, semantic_offset: int) -> tuple[str, str]:
    """Return deterministic outcome and completion-marker paths for one batch."""
    batch_root = (
        f"{output_root.rstrip('/')}/batches/" f"decision-{decision_file_index:05d}/semantic-{semantic_offset:08d}"
    )
    return f"{batch_root}.parquet", f"{batch_root}.json"


def _manifest_for_outcomes(
    *,
    model: str,
    machine_labels_path: str,
    decision_file: str,
    decision_file_index: int,
    semantic_offset: int,
    total_semantic_in_file: int,
    cases: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
    outcome_path: str,
    outcome_bytes: bytes,
) -> SemanticBatchManifest:
    counts = Counter(outcome["status"] for outcome in outcomes)
    modes = Counter(outcome["review_mode"] for outcome in outcomes)
    config_hashes = {outcome["config_sha256"] for outcome in outcomes}
    if len(config_hashes) != 1:
        raise AssertionError(f"Batch mixes semantic configurations: {sorted(config_hashes)}")
    return SemanticBatchManifest(
        model=model,
        machine_labels_path=machine_labels_path,
        decision_file=decision_file,
        decision_file_index=decision_file_index,
        semantic_offset=semantic_offset,
        next_semantic_offset=semantic_offset + len(cases),
        total_semantic_in_file=total_semantic_in_file,
        cases=len(cases),
        case_keys_sha256=_case_keys_sha256(cases),
        config_sha256=config_hashes.pop(),
        resolved_pairs=counts["resolved"],
        unresolved_pairs=counts["unresolved"],
        direct_pairs=modes["direct"],
        chunked_pairs=modes["chunked"],
        request_attempts=sum(int(outcome["request_attempts"]) for outcome in outcomes),
        outcome_path=outcome_path,
        outcome_bytes=len(outcome_bytes),
        outcome_sha256=hashlib.sha256(outcome_bytes).hexdigest(),
    )


def write_completed_batch(
    *,
    model: str,
    machine_labels_path: str,
    decision_file: str,
    decision_file_index: int,
    semantic_offset: int,
    total_semantic_in_file: int,
    cases: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
    output_root: str,
) -> tuple[SemanticBatchManifest, str]:
    """Write an outcome shard, verify it, then publish its completion marker."""
    if len(cases) != len(outcomes):
        raise AssertionError(f"Outcome count differs from cases: {len(outcomes)}/{len(cases)}")
    for case, outcome in zip(cases, outcomes, strict=True):
        validate_outcome(case, outcome)

    outcome_path, marker_path = batch_paths(
        output_root,
        decision_file_index=decision_file_index,
        semantic_offset=semantic_offset,
    )
    marker = StoragePath(marker_path)
    if marker.exists():
        raise FileExistsError(f"Completed semantic batch already exists: {marker_path}")
    outcome_bytes = _parquet_bytes(outcomes)
    manifest = _manifest_for_outcomes(
        model=model,
        machine_labels_path=machine_labels_path,
        decision_file=decision_file,
        decision_file_index=decision_file_index,
        semantic_offset=semantic_offset,
        total_semantic_in_file=total_semantic_in_file,
        cases=cases,
        outcomes=outcomes,
        outcome_path=outcome_path,
        outcome_bytes=outcome_bytes,
    )

    outcome = StoragePath(outcome_path)
    outcome.parent.mkdirs()
    outcome.write_bytes(outcome_bytes)
    persisted = outcome.read_bytes()
    if hashlib.sha256(persisted).hexdigest() != manifest.outcome_sha256:
        raise AssertionError(f"Persisted semantic shard hash differs for {outcome_path}")
    persisted_rows = _parquet_rows(persisted)
    for case, row in zip(cases, persisted_rows, strict=True):
        validate_outcome(case, row)
    if len(persisted_rows) != len(cases):
        raise AssertionError(f"Persisted semantic shard row count differs: {len(persisted_rows)}/{len(cases)}")

    marker.write_text(manifest.model_dump_json(indent=2))
    persisted_manifest = SemanticBatchManifest.model_validate_json(marker.read_text())
    if persisted_manifest != manifest:
        raise AssertionError(f"Persisted semantic completion marker differs for {marker_path}")
    return manifest, marker_path


def completed_batch(
    *,
    model: str,
    machine_labels_path: str,
    decision_file: str,
    decision_file_index: int,
    semantic_offset: int,
    total_semantic_in_file: int,
    cases: list[dict[str, Any]],
    output_root: str,
) -> tuple[SemanticBatchManifest, list[dict[str, Any]], str] | None:
    """Return a prior batch only after revalidating its marker, bytes, and rows."""
    _, marker_path = batch_paths(
        output_root,
        decision_file_index=decision_file_index,
        semantic_offset=semantic_offset,
    )
    marker = StoragePath(marker_path)
    if not marker.exists():
        return None
    manifest = SemanticBatchManifest.model_validate_json(marker.read_text())
    expected_fields = {
        "model": model,
        "machine_labels_path": machine_labels_path,
        "decision_file": decision_file,
        "decision_file_index": decision_file_index,
        "semantic_offset": semantic_offset,
        "next_semantic_offset": semantic_offset + len(cases),
        "total_semantic_in_file": total_semantic_in_file,
        "cases": len(cases),
        "case_keys_sha256": _case_keys_sha256(cases),
    }
    mismatches = [field for field, value in expected_fields.items() if getattr(manifest, field) != value]
    if mismatches:
        raise AssertionError(f"Semantic completion marker does not bind this batch: {mismatches}")

    payload = StoragePath(manifest.outcome_path).read_bytes()
    if len(payload) != manifest.outcome_bytes:
        raise AssertionError(
            f"Semantic shard size differs for {manifest.outcome_path}: " f"{len(payload)}/{manifest.outcome_bytes}"
        )
    if hashlib.sha256(payload).hexdigest() != manifest.outcome_sha256:
        raise AssertionError(f"Semantic shard hash differs for {manifest.outcome_path}")
    outcomes = _parquet_rows(payload)
    if len(outcomes) != len(cases):
        raise AssertionError(f"Semantic shard row count differs: {len(outcomes)}/{len(cases)}")
    for case, outcome in zip(cases, outcomes, strict=True):
        validate_outcome(case, outcome)
    regenerated = _manifest_for_outcomes(
        model=model,
        machine_labels_path=machine_labels_path,
        decision_file=decision_file,
        decision_file_index=decision_file_index,
        semantic_offset=semantic_offset,
        total_semantic_in_file=total_semantic_in_file,
        cases=cases,
        outcomes=outcomes,
        outcome_path=manifest.outcome_path,
        outcome_bytes=payload,
    )
    if regenerated != manifest:
        raise AssertionError(f"Semantic completion marker counters differ for {marker_path}")
    return manifest, outcomes, marker_path


def _decision_files(machine: DedupMachineLabelsData) -> list[str]:
    paths = sorted(str(path) for path in StoragePath(f"{machine.decisions_dir.rstrip('/')}/*.parquet").glob())
    if not paths:
        raise FileNotFoundError(f"No machine decision files under {machine.decisions_dir}")
    return paths


def _add_manifest_counters(counters: Counter[str], manifest: SemanticBatchManifest) -> None:
    counters["pairs"] += manifest.cases
    counters["resolved"] += manifest.resolved_pairs
    counters["unresolved"] += manifest.unresolved_pairs
    counters["direct"] += manifest.direct_pairs
    counters["chunked"] += manifest.chunked_pairs
    counters["request_attempts"] += manifest.request_attempts


def run_semantic_review(
    *,
    machine_labels_path: str,
    output_root: str,
    model: str,
    decision_file_start: int,
    decision_file_stop: int | None,
    batch_size: int,
    instances: int,
) -> SemanticReviewData:
    """Review a deterministic decision-file range, resuming verified batches."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if instances <= 0:
        raise ValueError(f"instances must be positive, got {instances}")

    machine = DedupMachineLabelsData.model_validate_json(StoragePath(machine_labels_path).read_text())
    decision_files = _decision_files(machine)
    stop = len(decision_files) if decision_file_stop is None else decision_file_stop
    if not 0 <= decision_file_start < stop <= len(decision_files):
        raise ValueError(f"Decision-file range {decision_file_start}:{stop} outside 0:{len(decision_files)}")

    served_model, engine, iris, broker = _inference_config(model)
    counters: Counter[str] = Counter()
    marker_paths = []
    expected_pairs = 0
    with remote_inference(served_model, engine, iris, instances=instances, broker=broker) as session:
        client = AsyncOpenAI(
            base_url=session.model.endpoint.base_url,
            api_key=session.model.endpoint.api_key or "local",
        )

        async def review_all() -> None:
            nonlocal expected_pairs
            try:
                for decision_file_index in range(decision_file_start, stop):
                    decision_file = decision_files[decision_file_index]
                    cases, total_semantic = load_semantic_cases(
                        _records(decision_file),
                        semantic_offset=0,
                        limit=2**63 - 1,
                    )
                    if len(cases) != total_semantic:
                        raise AssertionError(
                            f"Loaded semantic count differs for {decision_file}: " f"{len(cases)}/{total_semantic}"
                        )
                    expected_pairs += total_semantic
                    logger.info(
                        "Loaded decision file %d/%d with %d semantic pairs",
                        decision_file_index + 1,
                        stop,
                        total_semantic,
                    )
                    for semantic_offset in range(0, total_semantic, batch_size):
                        batch_cases = cases[semantic_offset : semantic_offset + batch_size]
                        prior = completed_batch(
                            model=model,
                            machine_labels_path=machine_labels_path,
                            decision_file=decision_file,
                            decision_file_index=decision_file_index,
                            semantic_offset=semantic_offset,
                            total_semantic_in_file=total_semantic,
                            cases=batch_cases,
                            output_root=output_root,
                        )
                        if prior is None:
                            logger.info(
                                "Reviewing decision file %d semantic range %d:%d",
                                decision_file_index,
                                semantic_offset,
                                semantic_offset + len(batch_cases),
                            )
                            outcomes = await review_cases(
                                client,
                                model=session.model.endpoint.model,
                                cases=batch_cases,
                            )
                            manifest, marker_path = write_completed_batch(
                                model=model,
                                machine_labels_path=machine_labels_path,
                                decision_file=decision_file,
                                decision_file_index=decision_file_index,
                                semantic_offset=semantic_offset,
                                total_semantic_in_file=total_semantic,
                                cases=batch_cases,
                                outcomes=outcomes,
                                output_root=output_root,
                            )
                        else:
                            manifest, _, marker_path = prior
                            logger.info(
                                "Resumed verified decision file %d semantic range %d:%d",
                                decision_file_index,
                                semantic_offset,
                                semantic_offset + len(batch_cases),
                            )
                        _add_manifest_counters(counters, manifest)
                        marker_paths.append(marker_path)
                        logger.info(
                            "Semantic progress pairs=%d resolved=%d unresolved=%d",
                            counters["pairs"],
                            counters["resolved"],
                            counters["unresolved"],
                        )
            finally:
                await client.close()

        asyncio.run(review_all())

    if counters["pairs"] != expected_pairs:
        raise AssertionError(f"Semantic range coverage differs: {counters['pairs']}/{expected_pairs}")
    result = SemanticReviewData(
        model=model,
        machine_labels_path=machine_labels_path,
        output_root=output_root,
        decision_file_start=decision_file_start,
        decision_file_stop=stop,
        decision_files=stop - decision_file_start,
        expected_pairs=expected_pairs,
        completed_pairs=counters["pairs"],
        batch_manifests=marker_paths,
        counters=dict(sorted(counters.items())),
    )
    summary_name = (
        "semantic-review.json"
        if decision_file_start == 0 and stop == len(decision_files)
        else f"semantic-review-{decision_file_start:05d}-{stop:05d}.json"
    )
    summary_path = StoragePath(output_root) / summary_name
    summary_path.write_text(result.model_dump_json(indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine-labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--decision-file-start", type=int, default=0)
    parser.add_argument("--decision-file-stop", type=int)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--instances", type=int, required=True)
    args = parser.parse_args()
    configure_logging()

    result = run_semantic_review(
        machine_labels_path=args.machine_labels,
        output_root=args.output,
        model=args.model,
        decision_file_start=args.decision_file_start,
        decision_file_stop=args.decision_file_stop,
        batch_size=args.batch_size,
        instances=args.instances,
    )
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
