# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The repair-candidate lane CLI: match, comparative review and the dry-run report.

``dry-run`` maps one candidate folder onto one tracked source by content, reviews every
content-differing pair comparatively, and aggregates the typed outcomes into a report. It
performs no registry writes and no admission: a dry run's decisions are pending until the
oracle/NoP gates run, and publication is a separate ruled step.
"""

import argparse
import json
import logging
import shutil
import tempfile
import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pyarrow.parquet as pq
from marin.inference.openai_chat import OpenAIChatClient
from pydantic import BaseModel, ConfigDict, Field
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.models import ImportRejection, ReviewStatus
from taskcompendium.pipeline.repair import ComparativeOutcome, RepairMatchClass

from experiments.post_training.task_curation.full_qc import LONG_REQUEST_TIMEOUT, ReviewerVersion, local_reviewer
from experiments.post_training.task_curation.judges import LocalJudgeEndpoint, PinnedLocalJudgeEndpoint
from experiments.post_training.task_curation.registry import read_registry_json
from experiments.post_training.task_curation.repair_candidates import (
    CANDIDATE_REPO,
    load_candidate_population,
    preview_task,
)
from experiments.post_training.task_curation.repair_match import MatchOutcome, match_candidates
from experiments.post_training.task_curation.repair_review import (
    ComparativeBatchResult,
    ComparativePair,
    ComparativePairReviewer,
)
from experiments.post_training.task_curation.task_neighbors import PopulationIndex, TaskContent, content_rows
from infra.marina.apps.rl_data_catalog.server.source_identity import canonical_source_id, original_source_id

logger = logging.getLogger(__name__)

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
    comparative: dict[str, Any] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    usage: dict[str, int] = Field(default_factory=dict)
    review_wall_seconds: float | None = None
    gpu_hours: float | None = None
    serving_jobs: tuple[str, ...] = ()


@dataclass(frozen=True)
class MatchedPairs:
    """Content-matched pairs whose content differs, ready for comparative review."""

    pairs: list[ComparativePair]
    tracked_by_candidate: dict[str, str]
    """Candidate task ID to tracked task ID; a local join key, never the match itself."""


class TrackedPopulation(BaseModel):
    """One tracked source's loaded population: a content index plus TaskSpecs when available.

    Prepared content pins carry the matching rows but no TaskSpecs; a dry run can match
    against them and reports that comparative review needs the source's configured
    conversion for the tracked side.
    """

    source_id: str
    shape: Literal["specs", "archives", "prepared"]
    tasks: dict[str, TaskSpec]
    index: Any


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


def candidate_index(population) -> PopulationIndex:
    rows = list(content_rows(candidate.task.model_dump(mode="json") for candidate in population.tasks))
    if not rows:
        raise ValueError("The candidate population is empty")
    return PopulationIndex(rows, {population.source_id: population.source_id})


def matched_pairs(
    outcome: MatchOutcome, tracked_tasks: dict[str, TaskSpec], candidate_tasks: list[TaskSpec]
) -> MatchedPairs:
    """Build the review workload: every REPAIRED pair with both sides' TaskSpecs."""
    candidate_by_id = {task.id: task for task in candidate_tasks}
    join: dict[str, str] = {}
    pairs: list[ComparativePair] = []
    for receipt in outcome.receipts:
        if receipt.match_class is not RepairMatchClass.REPAIRED or receipt.tracked_task_id is None:
            continue
        candidate = candidate_by_id.get(receipt.candidate_task_id)
        original = tracked_tasks.get(receipt.tracked_task_id)
        if candidate is None or original is None:
            raise ValueError(f"Match receipt references a missing task: {receipt.candidate_task_id}")
        join[candidate.id] = original.id
        pairs.append(ComparativePair(candidate=candidate, original=original))
    return MatchedPairs(pairs=pairs, tracked_by_candidate=join)


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
        "outcomes_enum": [outcome.value for outcome in ComparativeOutcome],
    }


def match_diagnostics(
    outcome: MatchOutcome, tracked: PopulationIndex, candidates: PopulationIndex, *, sample_characters: int = 400
) -> dict[str, Any]:
    """Sample evidence for a match report: strongest matches and instruction previews.

    A zero-overlap census is not interpretable without the texts behind it; the report
    carries the strongest forward matches, the strongest tracked-side similarities and a
    few normalized instruction previews from both populations.
    """
    forward = sorted(outcome.receipts, key=lambda receipt: receipt.similarity, reverse=True)[:10]
    reverse_top = sorted(outcome.reverse_best.items(), key=lambda item: item[1], reverse=True)[:10]
    previews = {
        label: [row.text[:sample_characters] for row in index.raw_rows[:3]]
        for label, index in (("tracked", tracked), ("candidate", candidates))
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
    }


def dry_run(
    *,
    tracked_source_id: str,
    tracked_uris: tuple[str, ...],
    candidate_source_id: str,
    candidate_revision: str,
    folders: tuple[str, ...],
    repo: str,
    reviewer_version: ReviewerVersion,
    run: bool,
) -> DryRunReport:
    """Match one candidate folder against one tracked source and comparatively review it."""
    candidates = load_candidate_population(
        folders=folders, repo=repo, revision=candidate_revision, candidate_source_id=candidate_source_id
    )
    tracked_population = load_tracked_population(tracked_uris, tracked_source_id=tracked_source_id)
    candidates_rows = candidate_index(candidates)
    outcome = match_candidates(
        tracked_population.index,
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
    )
    if tracked_population.tasks:
        workload = matched_pairs(
            outcome,
            tracked_tasks=tracked_population.tasks,
            candidate_tasks=[candidate.task for candidate in candidates.tasks],
        )
    else:
        # A prepared content pin carries no TaskSpecs; the dry run reports matching only.
        workload = MatchedPairs(pairs=[], tracked_by_candidate={})
    report = report.model_copy(
        update={
            "diagnostics": {
                **match_diagnostics(outcome, tracked_population.index, candidates_rows),
                "tracked_shape": tracked_population.shape,
            }
        }
    )
    if not run or not workload.pairs:
        logger.info("Dry run without review: %d repaired pairs pending", len(workload.pairs))
        return report
    result, evidence = review_pairs(workload.pairs, reviewer_version)
    summary = comparative_summary(result)
    report = report.model_copy(
        update={
            "comparative": {**summary, "judge": evidence.identity},
            "usage": {
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
            },
            "review_wall_seconds": evidence.wall_seconds,
            # The serving job holds the GPU for its whole lifetime; four GB200s per job.
            "gpu_hours": evidence.wall_seconds * 4 / 3600,
            "serving_jobs": evidence.serving_jobs,
        }
    )
    return report


def write_report(report: DryRunReport, output_uri: str) -> None:
    payload = (report.model_dump_json(indent=2) + "\n").encode()
    path = StoragePath(output_uri)
    with path.open("wb", auto_mkdir=True) as stream:
        stream.write(payload)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracked-source-id", required=True)
    parser.add_argument(
        "--tracked-parquet",
        action="append",
        help="Pinned tracked task-spec parquet URIs; resolved from the registry when omitted",
    )
    parser.add_argument("--candidate-source-id", required=True)
    parser.add_argument("--candidate-revision", required=True)
    parser.add_argument("--folder", action="append", required=True)
    parser.add_argument("--repo", default=CANDIDATE_REPO)
    parser.add_argument(
        "--reviewer-version", type=ReviewerVersion, choices=list(ReviewerVersion), default=ReviewerVersion.V3
    )
    parser.add_argument("--report", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    configure_coreweave_s3()
    tracked_uris = tuple(args.tracked_parquet) if args.tracked_parquet else resolve_tracked_uris(args.tracked_source_id)
    logger.info("Tracked population inputs: %s", list(tracked_uris))
    report = dry_run(
        tracked_source_id=args.tracked_source_id,
        tracked_uris=tracked_uris,
        candidate_source_id=args.candidate_source_id,
        candidate_revision=args.candidate_revision,
        folders=tuple(args.folder),
        repo=args.repo,
        reviewer_version=args.reviewer_version,
        run=args.run,
    )
    write_report(report, args.report)
    logger.info("RESULT %s", report.model_dump_json())


if __name__ == "__main__":
    main()
