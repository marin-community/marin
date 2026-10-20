# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experiment-specific, reproducible selection from frozen TaskSpec populations."""

import json
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.fingerprints import semantic_digest
from taskcompendium.pipeline.sampling import seeded_order, seeded_sample

from experiments.post_training.snowball_curriculum.bundle import write_manifest, write_pool

PUBLICATION_MANIFEST = "s3://marin-us-east-02a/marin/rl-semantics/publication-008e95e5/2026.10.15.1/release.json"
PRODUCTION_CATALOG = "s3://marin-us-east-02a/marin/rl-data/catalog/production"
type Population = Literal["converted", "admitted"]


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    intended_use: str
    task_paths: tuple[str, ...]
    population: Population
    family: str | None = None
    manifest_sha256: str | None = None
    source_key: str | None = None
    catalog_snapshot_id: str | None = None
    assessment_curation_uri: str | None = None


@dataclass(frozen=True)
class TaskCandidate:
    source_id: str
    task_id: str
    contract_id: str
    public_content_id: str
    difficulty: str
    task: dict[str, Any]
    intended_use: str = "train"
    population: Population = "converted"
    prior_admission_status: str = "unknown"


@dataclass(frozen=True, order=True)
class TaskAlias:
    source_id: str
    task_id: str


@dataclass(frozen=True)
class SelectionResult:
    selected: tuple[TaskCandidate, ...]
    eligible_counts: dict[str, int]
    source_quotas: dict[str, int]
    difficulty_counts: dict[str, int]
    rejected_counts: dict[str, int]
    shortfall: int
    eligible_difficulty_counts: dict[str, dict[str, int]]
    difficulty_quotas: dict[str, dict[str, int]]
    candidate_counts: dict[str, int]
    selected_aliases: dict[str, tuple[TaskAlias, ...]]


def allocate_quotas(capacities: Mapping[str, int], target: int, weights: Mapping[str, float]) -> dict[str, int]:
    """Allocate integer quotas with capped weighted water filling and stable ties."""
    if target < 0 or any(capacity < 0 for capacity in capacities.values()):
        raise ValueError("Counts must be nonnegative")
    if any(not math.isfinite(weight) or weight < 0 for weight in weights.values()):
        raise ValueError("Weights must be finite and nonnegative")
    quotas = {key: 0 for key in sorted(capacities)}
    active = {key for key, capacity in capacities.items() if capacity and weights.get(key, 0) > 0}
    remaining = min(target, sum(capacities[key] for key in active))
    while remaining and active:
        total_weight = sum(weights[key] for key in active)
        ideals = {key: remaining * weights[key] / total_weight for key in active}
        capped = {key for key in active if ideals[key] >= capacities[key] - quotas[key]}
        if capped:
            for key in capped:
                count = capacities[key] - quotas[key]
                quotas[key] += count
                remaining -= count
            active -= capped
            continue
        whole = {key: math.floor(value) for key, value in ideals.items()}
        for key, count in whole.items():
            quotas[key] += count
            remaining -= count
        for key in sorted(active, key=lambda key: (-(ideals[key] - whole[key]), key))[:remaining]:
            quotas[key] += 1
        remaining = 0
    return quotas


def source_weights(capacities: Mapping[str, int], equal_fraction: float = 0.5) -> dict[str, float]:
    """Blend equal-source and square-root population mass over available sources."""
    if not 0 <= equal_fraction <= 1:
        raise ValueError("equal_fraction must lie in [0, 1]")
    positive = {key: count for key, count in capacities.items() if count > 0}
    total = sum(math.sqrt(count) for count in positive.values())
    return {
        key: equal_fraction / len(positive) + (1 - equal_fraction) * math.sqrt(count) / total
        for key, count in positive.items()
    }


def select_candidates(
    candidates: Callable[[], Iterable[TaskCandidate]],
    *,
    target: int,
    seed: int,
    weights: Mapping[str, float] | None = None,
    heldout_contract_ids: Iterable[str] = (),
    heldout_public_content_ids: Iterable[str] = (),
    heldout_task_ids: Iterable[str] = (),
    heldout_context_ids: Iterable[str] = (),
) -> SelectionResult:
    """Count eligible unique contracts, then sample evenly across difficulty bins.

    The factory is read twice. Only compact identity metadata for all contracts
    and selected task payloads are retained. Evaluation rows in the factory also
    establish exact public-content and contract exclusions before quotas.
    Inputs must remain frozen between passes.
    """
    heldout_contracts = set(heldout_contract_ids)
    heldout_public = set(heldout_public_content_ids)
    heldout_tasks = set(heldout_task_ids)
    heldout_contexts = set(heldout_context_ids)
    context_ids: dict[str, str | None] = {}
    winners: dict[str, tuple[str, str, str, str]] = {}
    aliases: dict[str, set[TaskAlias]] = defaultdict(set)
    rejected: Counter[str] = Counter()
    candidate_counts: Counter[str] = Counter()
    for row in candidates():
        candidate_counts[row.source_id] += 1
        if row.task_id in heldout_tasks:
            heldout_contracts.add(row.contract_id)
        context_id = canonical_sha256(row.task["context"]) if "context" in row.task else None
        if row.intended_use != "train":
            rejected["non_train"] += 1
            heldout_contracts.add(row.contract_id)
            heldout_public.add(row.public_content_id)
            heldout_tasks.add(row.task_id)
            if context_id is not None:
                heldout_contexts.add(context_id)
            continue
        context_ids[row.contract_id] = context_id
        identity = (row.source_id, row.task_id, row.difficulty, row.public_content_id)
        if row.contract_id in winners:
            rejected["duplicate_contract"] += 1
            winner = winners[row.contract_id]
            aliases[row.contract_id].update((TaskAlias(winner[0], winner[1]), TaskAlias(row.source_id, row.task_id)))
            winners[row.contract_id] = min(winners[row.contract_id], identity)
        else:
            winners[row.contract_id] = identity
    for contract, identity in list(winners.items()):
        if (
            contract in heldout_contracts
            or identity[3] in heldout_public
            or identity[1] in heldout_tasks
            or any(alias.task_id in heldout_tasks for alias in aliases.get(contract, ()))
            or context_ids[contract] in heldout_contexts
        ):
            rejected["heldout_overlap"] += 1
            del winners[contract]
    capacities: Counter[str] = Counter()
    bins: dict[str, Counter[str]] = defaultdict(Counter)
    for source, _task, difficulty, _public in winners.values():
        capacities[source] += 1
        bins[source][difficulty] += 1
    resolved_weights = weights if weights is not None else source_weights(capacities)
    quotas = allocate_quotas(capacities, target, resolved_weights)
    bin_quotas = {
        (source, difficulty): count
        for source, capacity in bins.items()
        for difficulty, count in allocate_quotas(capacity, quotas[source], dict.fromkeys(capacity, 1.0)).items()
    }
    # Each bin is sampled in one pass without retaining rejected task payloads.
    # seeded_sample merges at most twice the required payloads per bin per batch.
    pending: dict[tuple[str, str], list[TaskCandidate]] = defaultdict(list)
    selected_bins: dict[tuple[str, str], list[TaskCandidate]] = defaultdict(list)
    seen_second_pass: set[str] = set()

    def flush(key: tuple[str, str]) -> None:
        _, selected_bins[key] = seeded_sample(
            (*selected_bins[key], *pending[key]),
            size=bin_quotas[key],
            key=lambda row: seeded_order(row.contract_id, seed),
        )
        pending[key].clear()

    for row in candidates():
        identity = (row.source_id, row.task_id, row.difficulty, row.public_content_id)
        key = (row.source_id, row.difficulty)
        if (
            row.intended_use != "train"
            or winners.get(row.contract_id) != identity
            or row.contract_id in seen_second_pass
        ):
            continue
        seen_second_pass.add(row.contract_id)
        if not bin_quotas.get(key):
            continue
        pending[key].append(row)
        if len(pending[key]) >= bin_quotas[key]:
            flush(key)
    if len(seen_second_pass) != len(winners):
        raise ValueError("Candidate identities changed between selection passes")
    for key in pending:
        flush(key)
    selected = tuple(
        sorted((row for rows in selected_bins.values() for row in rows), key=lambda r: seeded_order(r.contract_id, seed))
    )
    return SelectionResult(
        selected=selected,
        eligible_counts=dict(sorted(capacities.items())),
        source_quotas=quotas,
        difficulty_counts=dict(Counter(f"{row.source_id}/{row.difficulty}" for row in selected)),
        rejected_counts=dict(rejected),
        shortfall=max(0, target - len(selected)),
        eligible_difficulty_counts={source: dict(counts) for source, counts in bins.items()},
        difficulty_quotas={
            source: {difficulty: bin_quotas[source, difficulty] for difficulty in counts}
            for source, counts in bins.items()
        },
        candidate_counts=dict(sorted(candidate_counts.items())),
        selected_aliases={
            row.contract_id: tuple(sorted(aliases.get(row.contract_id, {TaskAlias(row.source_id, row.task_id)})))
            for row in selected
        },
    )


def load_sources(manifest_uri: str, population: Population = "converted") -> tuple[SourceSpec, ...]:
    """Resolve explicit source snapshots or Atlas release sources into fixed paths."""
    manifest_path = StoragePath(manifest_uri)
    document = json.loads(manifest_path.read_bytes())
    entries = document if isinstance(document, list) else document["sources"]
    sources = []
    for entry in entries:
        declared = entry.get("population", document.get("population") if isinstance(document, dict) else None)
        if declared is not None and declared != population:
            raise ValueError(f"Source population {declared} differs from requested {population}")
        paths = entry.get("task_paths")
        manifest_sha256 = entry.get("manifest_sha256")
        if paths is None:
            root = StoragePath(entry["curation_root"])
            manifest_sha256 = canonical_sha256(json.loads((root / "manifest.json").read_bytes()))
            if entry.get("source_key") is not None and entry["source_key"] != manifest_sha256:
                raise ValueError(f"Curation manifest differs from frozen source_key for {entry['source_id']}")
            view = "final" if population == "admitted" else "normalize"
            paths = sorted(str(path) for path in (root / f"{view}/*.parquet").glob())
        else:
            paths = [
                str(manifest_path.parent / path) if not (StoragePath(path).scheme or StoragePath(path).rooted) else path
                for path in paths
            ]
        if not paths:
            raise ValueError(f"No TaskSpec paths for {entry['source_id']}")
        sources.append(
            SourceSpec(
                entry["source_id"],
                entry["intended_use"],
                tuple(paths),
                population,
                entry.get("family"),
                manifest_sha256,
                entry.get("source_key"),
                entry.get("catalog_snapshot_id"),
                entry.get("assessment_curation_uri"),
            )
        )
    return tuple(sorted(sources, key=lambda source: source.source_id))


def read_candidates(
    sources: Sequence[SourceSpec], difficulty_by_source: Mapping[str, str] | None = None
) -> Iterator[TaskCandidate]:
    """Read JSONL or Parquet task_json records through Marin storage routing."""
    for source in sources:
        for uri in source.task_paths:
            path = StoragePath(uri)
            with path.open("rb") as stream:
                if uri.endswith(".parquet"):
                    rows = (
                        row for batch in pq.ParquetFile(stream).iter_batches(batch_size=256) for row in batch.to_pylist()
                    )
                else:
                    rows = (json.loads(line) for line in stream if line.strip())
                for row in rows:
                    task_json = row.get("task_json")
                    if task_json is None:
                        continue
                    task = (
                        TaskSpec.model_validate_json(task_json)
                        if isinstance(task_json, str)
                        else TaskSpec.model_validate(task_json)
                    )
                    # The source prior is explicitly authored; unsupported native scales remain unknown.
                    difficulty = (difficulty_by_source or {}).get(source.source_id, "unknown")
                    yield TaskCandidate(
                        source.source_id,
                        task.id,
                        semantic_digest(task, include_reference=True),
                        semantic_digest(task, include_reference=False),
                        difficulty,
                        task.model_dump(mode="json"),
                        source.intended_use,
                        source.population,
                        "admitted" if source.population == "admitted" else row.get("prior_admission_status", "unknown"),
                    )


def write_selection(
    result: SelectionResult, sources: Sequence[SourceSpec], output: str, *, seed: int, target: int
) -> None:
    """Write a replayable input snapshot, TaskSpecs, and selection accounting."""
    root = StoragePath(output)
    if root.exists():
        raise FileExistsError(f"Frozen selection output already exists: {output}")
    root.mkdirs(exist_ok=True)
    snapshot = {"sources": [asdict(source) for source in sources]}
    (root / "sources.json").write_text(json.dumps(snapshot, sort_keys=True, indent=2))

    def records() -> Iterator[dict[str, Any]]:
        for row in result.selected:
            record = asdict(row)
            record.pop("task")
            record["task_json"] = json.dumps(row.task)
            record["contract_kind"] = "task_spec"
            record["inclusion_probability"] = (
                result.difficulty_quotas[row.source_id][row.difficulty]
                / result.eligible_difficulty_counts[row.source_id][row.difficulty]
            )
            yield record

    write_pool(records(), root)
    with (root / "selected-aliases.jsonl").open("w") as stream:
        for contract_id, task_aliases in result.selected_aliases.items():
            stream.write(
                json.dumps({"contract_id": contract_id, "aliases": [asdict(alias) for alias in task_aliases]}) + "\n"
            )
    report = {
        "eligible_counts": result.eligible_counts,
        "source_quotas": result.source_quotas,
        "difficulty_counts": result.difficulty_counts,
        "rejected_counts": result.rejected_counts,
        "shortfall": result.shortfall,
        "eligible_difficulty_counts": result.eligible_difficulty_counts,
        "difficulty_quotas": result.difficulty_quotas,
        "candidate_counts": result.candidate_counts,
    }
    report.update(
        seed=seed, target=target, selected_count=len(result.selected), source_snapshot_sha256=canonical_sha256(snapshot)
    )
    (root / "selection.json").write_text(json.dumps(report, sort_keys=True, indent=2))
    write_manifest(root)
