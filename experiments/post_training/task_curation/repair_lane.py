# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The repair-candidate lane CLI: match, comparative review and the dry-run report.

``dry-run`` maps one candidate folder onto one tracked source by content, reviews every
content-differing pair comparatively, and aggregates the typed outcomes into a report. It
performs no registry writes and no admission: a dry run's decisions are pending until the
oracle/NoP gates run, and publication is a separate ruled step.
"""

import argparse
import io
import json
import logging
import shutil
import tarfile
import tempfile
import time
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
import pyarrow.parquet as pq
from finestore.rl.licenses import LicenseReceipt
from marin.inference import iris as iris_inference
from marin.inference.openai_chat import OpenAIChatClient
from pydantic import BaseModel, ConfigDict, Field
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.identity import canonical_sha256, task_identity
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.conversion import row_source, row_task_id
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import ImportRejection, NormalizedTask, RawRow, ReviewStatus
from taskcompendium.pipeline.repair import ComparativeOutcome, RepairMatchClass

from experiments.post_training.task_curation.full_qc import LONG_REQUEST_TIMEOUT, ReviewerVersion, local_reviewer
from experiments.post_training.task_curation.images.build import EnvironmentArtifact, identity_digest
from experiments.post_training.task_curation.judges import LocalJudgeEndpoint, PinnedLocalJudgeEndpoint
from experiments.post_training.task_curation.license_binding import LICENSE_RECEIPT_INDEX, read_license_receipts
from experiments.post_training.task_curation.pipeline import (
    environment_requirements,
    planned_lock_url,
    source_recipe,
)
from experiments.post_training.task_curation.registry import read_registry_json
from experiments.post_training.task_curation.repair_candidates import (
    CANDIDATE_REPO,
    load_candidate_population,
    load_folder_rows,
    preview_task,
)
from experiments.post_training.task_curation.repair_match import (
    MatchOutcome,
    match_candidates,
    strip_template_sections_rows,
)
from experiments.post_training.task_curation.repair_review import (
    ComparativeBatchResult,
    ComparativePair,
    ComparativePairReviewer,
)
from experiments.post_training.task_curation.sources import configured_sources
from experiments.post_training.task_curation.task_neighbors import PopulationIndex, TaskContent, content_rows
from experiments.post_training.task_curation.task_spec_harbor import prepare
from infra.marina.apps.rl_data_catalog.server.licensing import require_unbanned
from infra.marina.apps.rl_data_catalog.server.source_identity import canonical_source_id, original_source_id

logger = logging.getLogger(__name__)

REPAIR_PLACEMENT_WAIT_SECONDS = 24 * 3600.0
"""How long the lane waits for a serving job to be placed.

The shared inference default is four hours, which expires when the GPU queue is congested
(the 2026-11-16 dry runs waited longer than that in the Kueue gate). The lane extends the
wait in its own process only; a proper configuration knob belongs in marin.inference.
"""

FINGERPRINT_INDEX = Path("experiments/post_training/task_curation/source_fingerprint_index.json")
POPULATION_LOCATORS = Path("experiments/post_training/task_curation/source_population_locators.json")


def resolve_tracked_uris(source_id: str) -> tuple[str, ...]:
    """Resolve a tracked source's pinned task-spec parquet inputs from the Atlas registry.

    The identity survey pins each fingerprinted source's input files; the locator manifest
    redirects their digests to co-located copies. Source IDs are tried in canonical and
    original spellings because the survey and its index key them differently.
    """
    index = read_registry_json(FINGERPRINT_INDEX)
    identity = index.get(source_id)
    if identity is None:
        raise ValueError(f"No fingerprint identity for tracked source: {source_id}")
    survey = json.loads(StoragePath(identity["uri"]).read_bytes())
    fingerprints = survey.get("fingerprints", {})
    fingerprint = fingerprints.get(source_id)
    if fingerprint is None:
        for spelling in (canonical_source_id(source_id), original_source_id(source_id)):
            fingerprint = fingerprints.get(spelling)
            if fingerprint is not None:
                break
    if not fingerprint or not fingerprint.get("inputs"):
        raise ValueError(f"Tracked source has no pinned population inputs: {source_id}")
    locations = read_registry_json(POPULATION_LOCATORS)
    spellings = {source_id, canonical_source_id(source_id), original_source_id(source_id)}
    prepared = next((entry for key, entry in locations.get("prepared", {}).items() if key in spellings), None)
    if prepared is not None and prepared.get("uri"):
        return (prepared["uri"],)
    files: dict[str, str] = locations.get("files", {})
    uris = tuple(files.get(digest, key) for key, digest in sorted(fingerprint["inputs"].items()))
    unmapped = [uri for uri in uris if not str(uri).startswith("s3://")]
    if unmapped:
        raise ValueError(f"Locator manifest lacks co-located inputs for {source_id}: {unmapped[:2]}")
    return uris


class DryRunReport(BaseModel):
    """The dry-run summary for one candidate folder against one tracked source.

    Dry-run decisions are always pending: the oracle/NoP gates and publication are
    separate ruled steps that a dry run never performs.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_source_id: str
    tracked_source_id: str
    candidate_revision: str
    census: dict[str, Any]
    advisories: int
    license: dict[str, Any] = Field(default_factory=dict)
    """The stage-0 gate evidence bound into every report: the approved per-folder lineage receipts."""
    unresolved_pairs: int = 0
    comparative: dict[str, Any] = Field(default_factory=dict)
    decisions: dict[str, Any] = Field(default_factory=dict)
    """Verdict-level per-pair decisions: ACCEPTED/REJECTED/UNADJUDICATED counts and per-pair entries."""
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    usage: dict[str, int] = Field(default_factory=dict)
    review_wall_seconds: float | None = None
    gpu_hours: float | None = None
    serving_cluster: str = ""
    """The cluster the serving job targets; recorded because reviewer placement is cluster-dependent."""
    serving_jobs: tuple[str, ...] = ()


class TrackedPopulation(BaseModel):
    """One tracked source's loaded population: a content index plus TaskSpecs when available.

    Prepared content pins carry the matching rows but no TaskSpecs; a dry run can match
    against them and reports that comparative review needs the source's configured
    conversion for the tracked side.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source_id: str
    shape: Literal["specs", "archives", "prepared"]
    tasks: dict[str, TaskSpec]
    index: PopulationIndex


def load_tracked_specs(uris: tuple[str, ...]) -> dict[str, TaskSpec]:
    """Load the tracked source's own conversion, keyed by normalized instruction hash.

    The specs pin carries full TaskSpecs for a source whose configured population pin is a
    prepared content parquet; the content join keeps matching on the current pin while the
    review sees the conversion the Atlas actually serves.
    """
    specs: dict[str, TaskSpec] = {}
    for uri in uris:
        for task in _read_spec_rows(uri):
            row = next(content_rows([task.model_dump(mode="json")]))
            specs.setdefault(row.instruction_hash, task)
    if not specs:
        raise ValueError("The tracked specs pin is empty")
    logger.info("Loaded %d tracked TaskSpecs from %d specs inputs", len(specs), len(uris))
    return specs


def _read_spec_rows(uri: str) -> Iterator[TaskSpec]:
    with tempfile.TemporaryFile() as local:
        with StoragePath(uri).open("rb") as remote:
            shutil.copyfileobj(remote, local)
        local.seek(0)
        with pq.ParquetFile(local) as handle:
            names = handle.schema_arrow.names
            if "lowered_task_spec" in names or "task_spec" in names:
                column = "lowered_task_spec" if "lowered_task_spec" in names else "task_spec"
                yield from _spec_rows(handle, column)
            else:
                raise ValueError(f"Tracked specs input has no task-spec column: {uri}")


def load_tracked_population(uris: tuple[str, ...], *, tracked_source_id: str) -> TrackedPopulation:
    """Load one tracked source's pinned inputs in any supported shape.

    Pinned inputs come in three shapes: complete TaskSpec JSON rows (``task_spec`` or
    ``lowered_task_spec``), TaskTrove task archives (``task_binary``), lowered through the
    same preview conversion the candidate loader uses so both sides hash identically, and
    prepared content parquets (the census schema), which carry the match rows directly.
    """
    tasks: dict[str, TaskSpec] = {}
    content_rows_loaded: list[TaskContent] = []
    shapes: set[str] = set()
    for uri in uris:
        with tempfile.TemporaryFile() as local:
            with StoragePath(uri).open("rb") as remote:
                shutil.copyfileobj(remote, local)
            local.seek(0)
            with pq.ParquetFile(local) as handle:
                names = handle.schema_arrow.names
                if "lowered_task_spec" in names or "task_spec" in names:
                    column = "lowered_task_spec" if "lowered_task_spec" in names else "task_spec"
                    shapes.add("specs")
                    for task in _spec_rows(handle, column):
                        if task.id in tasks:
                            raise ValueError(f"Duplicate tracked task ID: {task.id}")
                        tasks[task.id] = task
                elif "task_binary" in names:
                    shapes.add("archives")
                    for task in _archive_rows(handle, tracked_source_id):
                        if task.id in tasks:
                            raise ValueError(f"Duplicate tracked task ID: {task.id}")
                        tasks[task.id] = task
                elif {"task_id", "instruction_hash", "text", "state_hash", "tests_hash"} <= set(names):
                    shapes.add("prepared")
                    content_rows_loaded.extend(_content_rows(handle))
                else:
                    raise ValueError(f"Tracked input has no known task column: {uri}")
    if len(shapes) > 1:
        raise ValueError(f"Mixed tracked input shapes are not supported: {sorted(shapes)}")
    if not tasks and not content_rows_loaded:
        raise ValueError("The tracked population is empty")
    if tasks:
        rows = list(content_rows(task.model_dump(mode="json") for task in tasks.values()))
    else:
        rows = content_rows_loaded
    if not rows:
        raise ValueError("The tracked population has no content rows")
    index = PopulationIndex(rows, {uri: uri for uri in uris})
    shape = next(iter(shapes))
    logger.info("Loaded %d tracked tasks (%s) from %d parquet inputs", len(rows), shape, len(uris))
    return TrackedPopulation(source_id=tracked_source_id, shape=shape, tasks=tasks, index=index)


def _spec_rows(handle: pq.ParquetFile, column: str) -> Iterator[TaskSpec]:
    for batch in handle.iter_batches(batch_size=256, columns=[column]):
        for encoded in batch.column(column).to_pylist():
            yield TaskSpec.model_validate(json.loads(encoded))


def _archive_rows(handle: pq.ParquetFile, tracked_source_id: str) -> Iterator[TaskSpec]:
    for batch in handle.iter_batches(batch_size=64, columns=["task_binary"]):
        for value in batch.column("task_binary").to_pylist():
            row = value if isinstance(value, dict) else {"task_binary": value, "path": ""}
            task = preview_task(row, candidate_source_id=tracked_source_id, revision="tracked-archive-preview")
            if isinstance(task, ImportRejection):
                logger.warning("Tracked archive rejected in preview: %s", task.reason)
                continue
            yield task


def _content_rows(handle: pq.ParquetFile) -> Iterator[TaskContent]:
    for batch in handle.iter_batches(batch_size=1024):
        for row in batch.to_pylist():
            yield TaskContent(**row)


@dataclass(frozen=True)
class MatchedPairs:
    """Content-matched pairs whose content differs, ready for comparative review."""

    pairs: list[ComparativePair]
    tracked_by_candidate: dict[str, str]
    """Candidate task ID to tracked task ID; a local join key, never the match itself."""
    unresolved: int = 0
    """Repaired pairs whose tracked TaskSpec the specs pin does not carry."""


def matched_pairs(
    outcome: MatchOutcome,
    tracked_tasks: dict[str, TaskSpec],
    candidate_tasks: list[TaskSpec],
    *,
    tracked_rows: list[TaskContent] | None = None,
    tracked_specs_by_instruction: dict[str, TaskSpec] | None = None,
) -> MatchedPairs:
    """Build the review workload: every REPAIRED pair with both sides' TaskSpecs.

    ``tracked_tasks`` supplies the TaskSpecs directly when the tracked pin is a conversion;
    otherwise ``tracked_specs_by_instruction`` joins them by content hash through
    ``tracked_rows``, so a prepared content pin still reviews against its own conversion.
    """
    candidate_by_id = {task.id: task for task in candidate_tasks}
    rows_by_id = {row.task_id: row for row in tracked_rows or []}
    join: dict[str, str] = {}
    pairs: list[ComparativePair] = []
    unresolved = 0
    for receipt in outcome.receipts:
        if receipt.match_class is not RepairMatchClass.REPAIRED or receipt.tracked_task_id is None:
            continue
        candidate = candidate_by_id.get(receipt.candidate_task_id)
        if candidate is None:
            raise ValueError(f"Match receipt references a missing candidate task: {receipt.candidate_task_id}")
        original = tracked_tasks.get(receipt.tracked_task_id)
        if original is None and tracked_specs_by_instruction is not None:
            row = rows_by_id.get(receipt.tracked_task_id)
            if row is not None:
                original = tracked_specs_by_instruction.get(row.instruction_hash)
        if original is None:
            unresolved += 1
            continue
        join[candidate.id] = original.id
        pairs.append(ComparativePair(candidate=candidate, original=original))
    return MatchedPairs(pairs=pairs, tracked_by_candidate=join, unresolved=unresolved)


@dataclass(frozen=True)
class ReviewEvidence:
    """Identity and cost observations from one served review pass."""

    identity: dict[str, Any]
    serving_jobs: tuple[str, ...]
    wall_seconds: float


def review_pairs(
    pairs: list[ComparativePair], reviewer_version: ReviewerVersion
) -> tuple[ComparativeBatchResult, ReviewEvidence]:
    """Serve the pinned FULL-QC judge and comparatively review every pair."""
    # Extend the placement wait in this process only; see REPAIR_PLACEMENT_WAIT_SECONDS.
    iris_inference._ENDPOINT_PLACEMENT_TIMEOUT_SECONDS = REPAIR_PLACEMENT_WAIT_SECONDS
    spec = local_reviewer(reviewer_version)
    endpoint = spec.endpoint
    assert isinstance(endpoint, (LocalJudgeEndpoint, PinnedLocalJudgeEndpoint))
    with endpoint.serve() as session:
        model_endpoint = session.model.endpoint
        client = OpenAIChatClient(
            model_endpoint.base_url, model_endpoint.api_key or "unused", timeout=LONG_REQUEST_TIMEOUT
        )
        reviewer = ComparativePairReviewer(client, spec, max_concurrent=32)
        started = time.monotonic()
        result = reviewer.review_pairs(pairs)
        return result, ReviewEvidence(
            identity=spec.identity, serving_jobs=tuple(endpoint.job_ids), wall_seconds=time.monotonic() - started
        )


def comparative_summary(result: ComparativeBatchResult) -> dict[str, Any]:
    outcomes = Counter(
        record.verdict.outcome.value for record in result.records if record.status is ReviewStatus.REVIEWED
    )
    defects_fixed: Counter = Counter()
    defects_introduced: Counter = Counter()
    for record in result.records:
        if record.verdict is None:
            continue
        defects_fixed.update(defect.value for defect in record.verdict.defects_fixed)
        defects_introduced.update(defect.value for defect in record.verdict.defects_introduced)
    statuses = Counter(record.status.value for record in result.records)
    return {
        "statuses": dict(statuses),
        "outcomes": dict(outcomes),
        "defects_fixed": dict(defects_fixed),
        "defects_introduced": dict(defects_introduced),
    }


def match_diagnostics(
    outcome: MatchOutcome, tracked: PopulationIndex, candidates: PopulationIndex, *, sample_characters: int = 400
) -> dict[str, Any]:
    """Sample evidence for a match report: strongest matches and instruction previews.

    A zero-overlap census is not interpretable without the texts behind it; the report
    carries the strongest forward matches, the strongest tracked-side similarities, a few
    normalized instruction previews from both populations, and the complete stripped text
    of the strongest pair.
    """
    forward = sorted(outcome.receipts, key=lambda receipt: receipt.similarity, reverse=True)[:10]
    reverse_top = sorted(outcome.reverse_best.items(), key=lambda item: item[1], reverse=True)[:10]
    previews = {
        label: [row.text[:sample_characters] for row in index.raw_rows[:3]]
        for label, index in (("tracked", tracked), ("candidate", candidates))
    }
    strongest = forward[0] if forward else None
    pair_sample = {}
    if strongest is not None:
        candidate = next((row for row in candidates.raw_rows if row.task_id == strongest.candidate_task_id), None)
        original = next((row for row in tracked.raw_rows if row.task_id == strongest.tracked_task_id), None)
        pair_sample = {
            "candidate_task_id": strongest.candidate_task_id,
            "tracked_task_id": strongest.tracked_task_id,
            "similarity": strongest.similarity,
            "candidate_text": candidate.text[:6000] if candidate else None,
            "tracked_text": original.text[:6000] if original else None,
        }
    return {
        "forward_top": [
            {
                "candidate": receipt.candidate_task_id,
                "tracked": receipt.tracked_task_id,
                "similarity": receipt.similarity,
            }
            for receipt in forward
        ],
        "reverse_top": [{"tracked": task_id, "similarity": similarity} for task_id, similarity in reverse_top],
        "instruction_previews": previews,
        "pair_sample": pair_sample,
    }


def folder_license_gate(
    repo: str, folders: tuple[str, ...], *, receipts: Mapping[str, LicenseReceipt] | None = None
) -> dict[str, LicenseReceipt]:
    """Stage 0 for a repair candidate, before any download, conversion or matching.

    Under the repackager ruling (licenses follow each folder's original upstream), the gate
    binds each folder's registered lineage receipt — the checker's per-folder decision at
    the checked upstream pin — and requires an approved decision plus an unbanned pin. A
    folder without a bound receipt (FAIL, ESCALATE, or unchecked) stops the run until it
    has a ruling.
    """
    bound_receipts = read_license_receipts(LICENSE_RECEIPT_INDEX) if receipts is None else receipts
    bound: dict[str, LicenseReceipt] = {}
    for folder in folders:
        key = f"{repo.replace('/', ':')}/{folder}"
        receipt = bound_receipts.get(key)
        if receipt is None:
            raise ValueError(f"License gate failed for {key}: no bound lineage receipt; the folder needs a ruling")
        if not any(decision.approved for decision in receipt.decisions):
            raise ValueError(f"License gate failed for {key}: no approved per-folder decision")
        require_unbanned(key, receipt.dataset_id, receipt)
        bound[key] = receipt
    return bound


def dry_run(
    *,
    tracked_source_id: str,
    tracked_uris: tuple[str, ...],
    tracked_spec_uris: tuple[str, ...] = (),
    candidate_source_id: str,
    candidate_revision: str,
    folders: tuple[str, ...],
    repo: str,
    reviewer_version: ReviewerVersion,
    run: bool,
    serving_cluster: str = "",
) -> DryRunReport:
    """Match one candidate folder against one tracked source and comparatively review it."""
    gate_receipts = folder_license_gate(repo, folders)
    candidates = load_candidate_population(
        folders=folders, repo=repo, revision=candidate_revision, candidate_source_id=candidate_source_id
    )
    tracked_population = load_tracked_population(tracked_uris, tracked_source_id=tracked_source_id)
    tracked_specs = load_tracked_specs(tracked_spec_uris) if tracked_spec_uris else {}
    candidates_rows_raw = list(content_rows(candidate.task.model_dump(mode="json") for candidate in candidates.tasks))
    if not candidates_rows_raw:
        raise ValueError("The candidate population is empty")
    # Both sides are stripped symmetrically; identity hashes stay untouched, so only the
    # near-neighbor comparison sees the stripped text.
    tracked = PopulationIndex(
        strip_template_sections_rows(tracked_population.index.raw_rows), {uri: uri for uri in tracked_uris}
    )
    candidates_rows = PopulationIndex(
        strip_template_sections_rows(candidates_rows_raw), {candidates.source_id: candidates.source_id}
    )
    outcome = match_candidates(
        tracked,
        candidates_rows,
        tracked_source_id=tracked_source_id,
        candidate_source_id=candidate_source_id,
    )
    report = DryRunReport(
        candidate_source_id=candidate_source_id,
        tracked_source_id=tracked_source_id,
        candidate_revision=candidate_revision,
        census=outcome.census.model_dump(mode="json"),
        advisories=len(outcome.advisories),
        license={
            "dataset_id": repo,
            "folders": {
                key: {
                    "approved": True,
                    "upstream": receipt.resolved_source.model_dump(mode="json") if receipt.resolved_source else None,
                    "receipt_sha256": canonical_sha256(receipt.model_dump(mode="json")),
                }
                for key, receipt in gate_receipts.items()
            },
        },
    )
    workload = matched_pairs(
        outcome,
        tracked_tasks=tracked_population.tasks,
        candidate_tasks=[candidate.task for candidate in candidates.tasks],
        tracked_rows=tracked_population.index.raw_rows if tracked_specs else None,
        tracked_specs_by_instruction=tracked_specs or None,
    )
    report = report.model_copy(
        update={
            "unresolved_pairs": workload.unresolved,
            "diagnostics": {
                **match_diagnostics(outcome, tracked, candidates_rows),
                "tracked_shape": tracked_population.shape,
            },
        }
    )
    if not run or not workload.pairs:
        logger.info("Dry run without review: %d repaired pairs pending", len(workload.pairs))
        return report
    result, evidence = review_pairs(workload.pairs, reviewer_version)
    summary = comparative_summary(result)
    decisions = comparative_decisions(result, workload)
    report = report.model_copy(
        update={
            "comparative": {**summary, "judge": evidence.identity},
            "decisions": {"summary": decision_summary(decisions), "pairs": decisions},
            "usage": {
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
            },
            "review_wall_seconds": evidence.wall_seconds,
            # The serving job holds the GPU for its whole lifetime; four GB200s per job.
            "gpu_hours": evidence.wall_seconds * 4 / 3600,
            "serving_cluster": serving_cluster,
            "serving_jobs": evidence.serving_jobs,
        }
    )
    return report


def comparative_decisions(result: ComparativeBatchResult, workload: MatchedPairs) -> list[dict[str, Any]]:
    """Bind per-pair verdicts to provisional verdict-level decisions and full failure details.

    A pair is verdict-ACCEPTED when the comparison is not worse and introduces no defects,
    verdict-REJECTED when worse, incomparable, or any defect is introduced, and
    UNADJUDICATED when the judge never returned a valid verdict. The final replace still
    requires the candidate's oracle and NoP qualification; these are the lane's verdict
    counts, not admissions.
    """
    records = {record.task_id: record for record in result.records}
    decisions = []
    for pair in workload.pairs:
        record = records[pair.candidate.id]
        verdict = record.verdict
        if record.status is not ReviewStatus.REVIEWED or verdict is None:
            outcome, introduced, fixed = None, [], []
            decision = "UNADJUDICATED"
            reasons = [record.detail or record.status.value]
        else:
            outcome = verdict.outcome.value
            introduced = [defect.value for defect in verdict.defects_introduced]
            fixed = [defect.value for defect in verdict.defects_fixed]
            if verdict.outcome in (ComparativeOutcome.BETTER, ComparativeOutcome.TIE) and not introduced:
                decision = "ACCEPTED"
                reasons = [f"comparative_outcome_{outcome}"]
            else:
                decision = "REJECTED"
                reasons = [f"comparative_outcome_{outcome}"] + (["defects_introduced"] if introduced else [])
        decisions.append(
            {
                "candidate_task_id": pair.candidate.id,
                "tracked_task_id": workload.tracked_by_candidate.get(pair.candidate.id),
                "candidate_task_sha256": task_identity(pair.candidate),
                "status": record.status.value,
                "outcome": outcome,
                "defects_fixed": fixed,
                "defects_introduced": introduced,
                "decision": decision,
                "reasons": reasons,
            }
        )
    return decisions


def decision_summary(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(entry["decision"] for entry in decisions)
    reasons = Counter(reason for entry in decisions for reason in entry["reasons"] if entry["status"] != "reviewed")
    return {"decisions": dict(counts), "failure_reasons": dict(reasons)}


COHORT_SPEC_SCHEMA = pa.schema(
    [
        ("winner_candidate_task_id", pa.string()),
        ("tracked_task_id", pa.string()),
        ("task_id", pa.string()),
        ("task_spec", pa.large_string()),
        ("runtime", pa.large_string()),
    ]
)


WRAPPER_SOLVE_SH = """#!/bin/bash
set -euo pipefail
cp /solution/solution.py /app/solution.py
"""
"""The pure wrapper the FWeindel repackager adds under solution/solve.sh.

Ruling (repackager-license-follows-upstream, coordinator 2026-11-18): the winners' solve.sh
is verified to be exactly this wrapper — the tracked upstream archives carry no solve.sh —
so the cohort stage drops it and the tracked converter's oracle layout applies unchanged.
"""


def _pure_wrapper(content: str) -> bool:
    stripped = "\n".join(line for line in content.splitlines() if line.strip() and not line.strip().startswith("#"))
    if "solution.py" not in stripped or not ("cp " in stripped or "exec " in stripped or "python" in stripped):
        return False
    return not any(token in stripped for token in ("rm ", "curl", "wget", "git ", "pip ", "sudo"))


def strip_wrapper_solve_sh(task_binary: bytes) -> bytes | None:
    """Rewrite a candidate task archive without solution/solve.sh, verifying it is the pure wrapper.

    Returns ``None`` when the archive carries no ``solution/solve.sh``. Raises when the
    solve.sh does anything beyond the verified wrapper, so a variant change fails loudly
    instead of silently reclassifying the winner's oracle.
    """
    with tarfile.open(fileobj=io.BytesIO(task_binary), mode="r:*") as archive:
        members = list(archive)
        solve = next((m for m in members if m.name.removeprefix("./") == "solution/solve.sh"), None)
        if solve is None:
            return None
        content = archive.extractfile(solve).read().decode(errors="replace")
        if not _pure_wrapper(content):
            raise ValueError(
                "The winner's solution/solve.sh is not the verified pure wrapper; the normalization "
                "ruling does not cover it: " + content[:200]
            )
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as rebuilt:
            for member in members:
                if member is solve:
                    continue
                data = archive.extractfile(member).read() if member.isfile() else None
                rebuilt.addfile(member, io.BytesIO(data) if data is not None else None)
        return buffer.getvalue()


def tracked_instruction_hashes(uris: tuple[str, ...]) -> dict[str, str]:
    """Each prepared tracked task's normalized instruction hash, keyed by its task ID."""
    hashes: dict[str, str] = {}
    for uri in uris:
        with tempfile.TemporaryFile() as local:
            with StoragePath(uri).open("rb") as remote:
                shutil.copyfileobj(remote, local)
            local.seek(0)
            with pq.ParquetFile(local) as handle:
                names = handle.schema_arrow.names
                if not {"task_id", "instruction_hash"} <= set(names):
                    raise ValueError(f"Tracked population input has no id+instruction columns: {uri}")
                for batch in handle.iter_batches(batch_size=256, columns=["task_id", "instruction_hash"]):
                    for task_id, instruction_hash in zip(
                        batch.column("task_id").to_pylist(),
                        batch.column("instruction_hash").to_pylist(),
                        strict=True,
                    ):
                        hashes.setdefault(task_id, instruction_hash)
    if not hashes:
        raise ValueError("The tracked population carries no task rows")
    return hashes


def tracked_runtimes(spec_uris: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    """Each tracked TaskSpec's original runtime binding, keyed by its normalized instruction hash.

    The tracked sources' specs parquets carry the runtime binding as its own column beside the
    full serialized TaskSpec; a repair of the same task reuses that binding.
    """
    runtimes: dict[str, dict[str, Any]] = {}
    for uri in spec_uris:
        with tempfile.TemporaryFile() as local:
            with StoragePath(uri).open("rb") as remote:
                shutil.copyfileobj(remote, local)
            local.seek(0)
            with pq.ParquetFile(local) as handle:
                names = handle.schema_arrow.names
                if "runtime" not in names or "task_spec" not in names:
                    raise ValueError(f"Tracked specs input has no task_spec+runtime columns: {uri}")
                for batch in handle.iter_batches(batch_size=32, columns=["task_spec", "runtime"]):
                    for encoded, runtime in zip(
                        batch.column("task_spec").to_pylist(),
                        batch.column("runtime").to_pylist(),
                        strict=True,
                    ):
                        if runtime is None:
                            continue
                        task = TaskSpec.model_validate_json(encoded)
                        row = next(content_rows([task.model_dump(mode="json")]))
                        runtimes.setdefault(row.instruction_hash, json.loads(runtime))
    if not runtimes:
        raise ValueError("The tracked specs carry no runtime bindings")
    return runtimes


def accepted_winners(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The ACCEPTED pairs of a sealed review report, in report order."""
    return [pair for pair in report["decisions"]["pairs"] if pair["decision"] == "ACCEPTED"]


def winner_paths(winners: list[Mapping[str, Any]]) -> list[str]:
    """Each winner's archive path: the candidate task ID's final segment."""
    return [str(pair["candidate_task_id"]).rsplit(":", 1)[-1] for pair in winners]


def cohort_conversion_context(recipe) -> ConversionContext:
    """The production conversion context: the recipe's own grader environment binding.

    The binding is the artifact the tracked source's own admission built — the deterministic
    environment-artifact lock under the active prefix — so the winners' graders run under
    exactly the environment their tracked counterparts ran under.
    """
    built = EnvironmentArtifact(
        path=str(StoragePath(planned_lock_url(recipe.grader)).parent),
        identity=identity_digest(recipe.grader),
    )
    return ConversionContext({}, environment_requirements(recipe.grader, built))


def convert_cohort(
    *,
    tracked_source_id: str,
    report: Mapping[str, Any],
    folders: tuple[str, ...],
    repo: str,
    candidate_revision: str,
    tracked_uris: tuple[str, ...],
    tracked_spec_uris: tuple[str, ...],
    output_root: str,
    shard_size: int = 256,
    drop_wrapper_solve_sh: bool = False,
) -> dict[str, Any]:
    """Convert the report's ACCEPTED winners through the tracked source's own converter.

    Crash-safe: converted winners are recorded in ``task-map.jsonl`` as they finish and are
    skipped on rerun; specs land in resumable parquet shards; the manifest summarizes. A
    rejected winner keeps its original tracked task and is recorded with its rejection.
    ``drop_wrapper_solve_sh`` normalizes FWeindel winners whose archives add the verified pure
    wrapper ``solution/solve.sh`` the tracked layout lacks — ``strip_wrapper_solve_sh`` drops
    it with a typed normalization count, and any non-wrapper variant fails loudly.
    """
    recipes = {source.info.id: source.config for source in configured_sources()}
    recipe = recipes.get(tracked_source_id)
    if recipe is None:
        raise ValueError(f"No configured source recipe for {tracked_source_id}")
    context = cohort_conversion_context(recipe)
    # The production conversion path: the recipe lowered to a SourceRecipe, whose SourceFiles
    # carries the dataset identity, decode and converter that row_source and the row bind need.
    recipe_source = source_recipe(recipe, {}, context.grader_environment)
    tracked_hashes: dict[str, str] | None = None
    runtime_join: dict[str, dict[str, Any]] | None = None
    winners = accepted_winners(report)
    paths = set(winner_paths(winners))
    rows = {}
    for row in load_folder_rows(repo, folders, revision=candidate_revision):
        if row.get("path") in paths:
            rows[row["path"]] = row
    missing = paths - set(rows)
    if missing:
        raise ValueError(f"Winners missing from the candidate folders: {sorted(missing)[:3]}")
    normalized: Counter = Counter()
    if drop_wrapper_solve_sh:
        for path in paths:
            rewritten = strip_wrapper_solve_sh(rows[path]["task_binary"])
            if rewritten is not None:
                rows[path]["task_binary"] = rewritten
                normalized["dropped_wrapper_solve_sh"] += 1

    identity = canonical_sha256(
        {
            "recipe_version": recipe.version,
            "drop_wrapper_solve_sh": drop_wrapper_solve_sh,
            "report": report,
        }
    )[:16]
    root = StoragePath(output_root) / f"conversion-{identity}"
    root.mkdirs()
    done: dict[str, dict[str, Any]] = {}
    map_path = root / "task-map.jsonl"
    specs_done: set[str] = set()
    pending_dir = root / "converted-pending"
    if pending_dir.exists():
        for shard in pending_dir.ls():
            with tempfile.TemporaryFile() as local:
                with shard.open("rb") as remote:
                    shutil.copyfileobj(remote, local)
                local.seek(0)
                with pq.ParquetFile(local) as handle:
                    for batch in handle.iter_batches(batch_size=256, columns=["winner_candidate_task_id"]):
                        specs_done.update(batch.column("winner_candidate_task_id").to_pylist())
    if map_path.exists():
        for line in map_path.read_text().splitlines():
            if line.strip():
                entry = json.loads(line)
                # A winner whose map entry exists but whose spec shard was lost (a failed
                # attempt between the map append and the shard flush) must reconvert.
                if entry["status"] == "converted" and entry["candidate_task_id"] not in specs_done:
                    continue
                done[entry["candidate_task_id"]] = entry
    buffer: list[tuple[str, dict[str, Any], TaskSpec, dict[str, Any]]] = []
    counts = Counter()
    for winner in winners:
        candidate_id = winner["candidate_task_id"]
        if candidate_id in done:
            counts["already_done"] += 1
            continue
        path = winner_paths([winner])[0]
        locator = f"{candidate_revision}:{path}"
        source = row_source(recipe_source, locator)
        decoded = (
            recipe_source.source.decode(rows[path], context) if recipe_source.source.decode is not None else rows[path]
        )
        row = RawRow(row_task_id(recipe_source, source), source, decoded)
        converted = recipe_source.convert(row, context)
        task = converted.task if isinstance(converted, NormalizedTask) else converted
        entry = {
            "candidate_task_id": candidate_id,
            "tracked_task_id": winner["tracked_task_id"],
            "converted_task_id": task.id if not isinstance(task, ImportRejection) else None,
            "status": "converted" if not isinstance(task, ImportRejection) else "rejected",
            "rejection": None if not isinstance(task, ImportRejection) else task.model_dump(mode="json"),
        }
        done[candidate_id] = entry
        with (root / "task-map.jsonl").open("a", auto_mkdir=True) as stream:
            stream.write(json.dumps(entry) + "\n")
        if not isinstance(task, ImportRejection):
            if tracked_hashes is None:
                tracked_hashes = tracked_instruction_hashes(tracked_uris)
            instruction_hash = tracked_hashes.get(winner["tracked_task_id"])
            if runtime_join is None:
                runtime_join = tracked_runtimes(tracked_spec_uris)
            runtime = runtime_join.get(instruction_hash)
            if runtime is None:
                raise ValueError(
                    f"No tracked runtime binding for the winner's tracked task: {winner['tracked_task_id']} "
                    f"(instruction {instruction_hash})"
                )
            buffer.append((candidate_id, entry, task, runtime))
            counts["converted"] += 1
        else:
            counts["rejected"] += 1
    if buffer:
        pending_dir.mkdirs()
        existing = {p.name for p in pending_dir.ls()} if pending_dir.exists() else set()
        index = 0
        while f"specs-part-{index:06d}.parquet" in existing:
            index += 1
        for start in range(0, len(buffer), shard_size):
            shard = buffer[start : start + shard_size]
            table = pa.Table.from_pylist(
                [
                    {
                        "winner_candidate_task_id": candidate_id,
                        "tracked_task_id": entry["tracked_task_id"],
                        "task_id": task.id,
                        "task_spec": task.model_dump_json(),
                        "runtime": json.dumps(runtime),
                    }
                    for candidate_id, entry, task, runtime in shard
                ],
                schema=COHORT_SPEC_SCHEMA,
            )
            name = f"specs-part-{index:06d}.parquet"
            with tempfile.TemporaryDirectory() as scratch:
                local = Path(scratch) / name
                with pq.ParquetWriter(local, COHORT_SPEC_SCHEMA) as writer:
                    writer.write_table(table)
                with (pending_dir / name).open("wb", auto_mkdir=True) as stream:
                    stream.write(local.read_bytes())
            index += 1
    manifest = {
        "tracked_source_id": tracked_source_id,
        "candidate_source_id": report["candidate_source_id"],
        "candidate_revision": report["candidate_revision"],
        "winners": len(winners),
        "counts": dict(counts),
        "recipe": {"name": recipe.name, "version": recipe.version},
        "grader_environment": context.grader_environment.model_dump(mode="json"),
        "conversion_identity": identity,
        "map_uri": str(root / "task-map.jsonl"),
        "shards_root": str(root / "converted-pending"),
    }
    if normalized:
        manifest["normalization"] = dict(normalized)
    (root / "manifest.json").write_bytes((json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
    StoragePath(output_root).mkdirs()
    (StoragePath(output_root) / "manifest.json").write_bytes(
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    )
    return manifest


def prepare_harbor_export(
    *,
    tracked_source_id: str,
    cohort_root: str,
    output_root: str,
    source_revision: str,
) -> dict[str, Any]:
    """Export the converted cohort as native Harbor task packages for production oracle trials.

    Downloads the cohort shards, runs ``task_spec_harbor.prepare`` with the tracked recipe's
    own Controls (the same golden bindings its admission used), and uploads the native tasks
    directory and task map to the output root.
    """
    recipes = {source.info.id: source.config for source in configured_sources()}
    recipe = recipes.get(tracked_source_id)
    if recipe is None:
        raise ValueError(f"No configured source recipe for {tracked_source_id}")
    if recipe.controls is None:
        raise ValueError(f"The tracked recipe declares no controls: {tracked_source_id}")
    manifest = json.loads(StoragePath(f"{cohort_root}/manifest.json").read_text())
    shards_root = manifest["shards_root"]
    with tempfile.TemporaryDirectory() as scratch:
        cohort = Path(scratch) / "cohort"
        cohort.mkdir()
        pending = StoragePath(shards_root)
        for shard in pending.ls():
            with shard.open("rb") as remote, (cohort / shard.name).open("wb") as stream:
                shutil.copyfileobj(remote, stream)
        output = Path(scratch) / "prepared"
        prepare(cohort, output, source_revision=source_revision, controls=recipe.controls)
        uploaded = 0
        for path in sorted(output.rglob("*")):
            if not path.is_file():
                continue
            uri = f"{output_root}/{path.relative_to(output)}"
            with StoragePath(uri).open("wb", auto_mkdir=True) as stream:
                stream.write(path.read_bytes())
            uploaded += 1
    tasks = sum(1 for _ in (StoragePath(f"{output_root}/tasks")).ls())
    return {"tracked_source_id": tracked_source_id, "tasks": tasks, "files_uploaded": uploaded}


def write_report(report: DryRunReport, output_uri: str) -> None:
    payload = (report.model_dump_json(indent=2) + "\n").encode()
    path = StoragePath(output_uri)
    with path.open("wb", auto_mkdir=True) as stream:
        stream.write(payload)


def build_parser() -> argparse.ArgumentParser:
    """The lane CLI; the candidate args bind only to the modes that consume them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracked-source-id", required=True)
    parser.add_argument(
        "--tracked-parquet",
        action="append",
        help="Pinned tracked task-spec parquet URIs; resolved from the registry when omitted",
    )
    parser.add_argument(
        "--tracked-specs",
        action="append",
        help="Full-TaskSpec parquet URIs for the tracked conversion, content-joined for review",
    )
    # The candidate args bind to the cohort/report modes only; the prepare mode uses none.
    parser.add_argument("--candidate-source-id")
    parser.add_argument("--candidate-revision")
    parser.add_argument("--folder", action="append")
    parser.add_argument("--repo", default=CANDIDATE_REPO)
    parser.add_argument(
        "--reviewer-version", type=ReviewerVersion, choices=list(ReviewerVersion), default=ReviewerVersion.V3
    )
    parser.add_argument(
        "--serving-cluster", default="", help="Recorded in the report; the cluster the serving job targets"
    )
    parser.add_argument(
        "--cohort-report",
        help="Sealed review report URI; set with --cohort-output-root to run the winner-conversion stage",
    )
    parser.add_argument(
        "--cohort-output-root", help="Output root for the winner cohort (specs shards, task map, manifest)"
    )
    parser.add_argument(
        "--with-prepare",
        action="store_true",
        help="After the cohort conversion, export the cohort as native Harbor tasks",
    )
    parser.add_argument(
        "--drop-wrapper-solve-sh",
        action="store_true",
        help="Normalize FWeindel winners by dropping the verified pure-wrapper solution/solve.sh",
    )
    parser.add_argument(
        "--prepare-output-root",
        help="Output root for the native Harbor export; set with --cohort-root to run the prepare stage",
    )
    parser.add_argument("--cohort-root", help="Converted cohort root; set with --prepare-output-root")
    parser.add_argument("--source-revision", default="", help="Pinned code revision declaring the controls")
    parser.add_argument("--report", required=True)
    parser.add_argument("--run", action="store_true")
    return parser


def require_candidate_args(parser: argparse.ArgumentParser, args: argparse.Namespace, *, prepare_mode: bool) -> None:
    """The candidate args bind to the cohort/report modes; the prepare mode uses none."""
    if not prepare_mode and (not args.candidate_source_id or not args.candidate_revision or not args.folder):
        parser.error(
            "--candidate-source-id, --candidate-revision and --folder are required outside --prepare-output-root"
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = build_parser()
    args = parser.parse_args()
    require_candidate_args(parser, args, prepare_mode=args.prepare_output_root is not None)
    configure_coreweave_s3()
    if args.cohort_output_root is not None:
        if args.cohort_report is None:
            parser.error("--cohort-output-root requires --cohort-report")
        if args.with_prepare and (args.prepare_output_root is None or not args.source_revision):
            parser.error("--with-prepare requires --prepare-output-root and --source-revision")
        report = json.loads(StoragePath(args.cohort_report).read_text())
        cohort_tracked_uris = (
            tuple(args.tracked_parquet) if args.tracked_parquet else resolve_tracked_uris(args.tracked_source_id)
        )
        manifest = convert_cohort(
            tracked_source_id=args.tracked_source_id,
            report=report,
            folders=tuple(args.folder),
            repo=args.repo,
            candidate_revision=args.candidate_revision,
            tracked_uris=cohort_tracked_uris,
            tracked_spec_uris=tuple(args.tracked_specs or ()),
            output_root=args.cohort_output_root,
            drop_wrapper_solve_sh=args.drop_wrapper_solve_sh,
        )
        if args.with_prepare:
            manifest["harbor"] = prepare_harbor_export(
                tracked_source_id=args.tracked_source_id,
                cohort_root=args.cohort_output_root,
                output_root=args.prepare_output_root,
                source_revision=args.source_revision,
            )
        logger.info("RESULT %s", json.dumps(manifest, sort_keys=True))
        return
    if args.prepare_output_root is not None:
        if args.cohort_root is None or not args.source_revision:
            parser.error("--prepare-output-root requires --cohort-root and --source-revision")
        summary = prepare_harbor_export(
            tracked_source_id=args.tracked_source_id,
            cohort_root=args.cohort_root,
            output_root=args.prepare_output_root,
            source_revision=args.source_revision,
        )
        logger.info("RESULT %s", json.dumps(summary, sort_keys=True))
        return
    tracked_uris = tuple(args.tracked_parquet) if args.tracked_parquet else resolve_tracked_uris(args.tracked_source_id)
    logger.info("Tracked population inputs: %s", list(tracked_uris))
    report = dry_run(
        tracked_source_id=args.tracked_source_id,
        tracked_uris=tracked_uris,
        tracked_spec_uris=tuple(args.tracked_specs or ()),
        candidate_source_id=args.candidate_source_id,
        candidate_revision=args.candidate_revision,
        folders=tuple(args.folder),
        repo=args.repo,
        reviewer_version=args.reviewer_version,
        run=args.run,
        serving_cluster=args.serving_cluster,
    )
    write_report(report, args.report)
    logger.info("RESULT %s", report.model_dump_json())


if __name__ == "__main__":
    main()
