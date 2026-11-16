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
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from marin.inference.openai_chat import OpenAIChatClient
from pydantic import BaseModel, ConfigDict, Field
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.models import ReviewStatus
from taskcompendium.pipeline.repair import ComparativeOutcome, RepairMatchClass

from experiments.post_training.task_curation.full_qc import LONG_REQUEST_TIMEOUT, ReviewerVersion, local_reviewer
from experiments.post_training.task_curation.judges import LocalJudgeEndpoint, PinnedLocalJudgeEndpoint
from experiments.post_training.task_curation.registry import read_registry_json
from experiments.post_training.task_curation.repair_candidates import load_candidate_population
from experiments.post_training.task_curation.repair_match import MatchOutcome, match_candidates
from experiments.post_training.task_curation.repair_review import (
    ComparativeBatchResult,
    ComparativePair,
    ComparativePairReviewer,
)
from experiments.post_training.task_curation.task_neighbors import PopulationIndex, content_rows
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


logger = logging.getLogger(__name__)


class DryRunReport(BaseModel):
    """The dry-run summary for one candidate folder against one tracked source."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_source_id: str
    tracked_source_id: str
    candidate_revision: str
    census: dict[str, Any]
    advisories: int
    comparative: dict[str, Any] = Field(default_factory=dict)
    decisions_pending: bool = True
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


def load_tracked_tasks(uris: tuple[str, ...]) -> dict[str, TaskSpec]:
    """Decode every pinned tracked TaskSpec, keyed by its ID for local joins only."""
    tasks: dict[str, TaskSpec] = {}
    for uri in uris:
        with pq.ParquetFile(StoragePath(uri).open("rb")) as handle:
            names = handle.schema_arrow.names
            column = "lowered_task_spec" if "lowered_task_spec" in names else "task_spec"
            for batch in handle.iter_batches(batch_size=256, columns=[column]):
                for encoded in batch.column(column).to_pylist():
                    task = TaskSpec.model_validate(json.loads(encoded))
                    if task.id in tasks:
                        raise ValueError(f"Duplicate tracked task ID: {task.id}")
                    tasks[task.id] = task
    if not tasks:
        raise ValueError("The tracked population is empty")
    logger.info("Loaded %d tracked tasks from %d parquet inputs", len(tasks), len(uris))
    return tasks


def tracked_index(tasks: dict[str, TaskSpec], uris: tuple[str, ...]) -> PopulationIndex:
    rows = list(content_rows(task.model_dump(mode="json") for task in tasks.values()))
    return PopulationIndex(rows, {uri: uri for uri in uris})


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


def review_pairs(pairs: list[ComparativePair], reviewer_version: ReviewerVersion) -> tuple[ComparativeBatchResult, Any]:
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
        return result, {
            "identity": spec.identity,
            "serving_jobs": endpoint.job_ids,
            "wall_seconds": time.monotonic() - started,
        }


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
    tracked_tasks = load_tracked_tasks(tracked_uris)
    outcome = match_candidates(
        tracked_index(tracked_tasks, tracked_uris),
        candidate_index(candidates),
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
    workload = matched_pairs(
        outcome, tracked_tasks=tracked_tasks, candidate_tasks=[candidate.task for candidate in candidates.tasks]
    )
    if not run or not workload.pairs:
        logger.info("Dry run without review: %d repaired pairs pending", len(workload.pairs))
        return report
    result, evidence = review_pairs(workload.pairs, reviewer_version)
    summary = comparative_summary(result)
    wall_seconds = evidence["wall_seconds"]
    report = report.model_copy(
        update={
            "comparative": {**summary, "judge": evidence["identity"]},
            "usage": {
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
            },
            "review_wall_seconds": wall_seconds,
            # The serving job holds the GPU for its whole lifetime; four GB200s per job.
            "gpu_hours": wall_seconds * 4 / 3600,
            "serving_jobs": tuple(evidence["serving_jobs"]),
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
    parser.add_argument("--repo", default="FWeindel/validated-tasks")
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
