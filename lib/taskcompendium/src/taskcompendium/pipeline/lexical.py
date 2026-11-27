# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lexical decontamination of a source's kept population against a frozen eval corpus.

The stage renders every kept task's conversation as text, marks it against the
caller's frozen eval-corpus artifacts, and records the measurement. Training
sources reject contaminated rows before verification and export, so a FULL
campaign writes the retained population to ``final/`` itself. Evaluation and
benchmark sources keep every task and record the measurement only.

The marking engine is injected through :class:`DecontaminationEngine`, the same
way model review receives a ``Reviewer`` and verification receives
``GradingMachines``: the pipeline owns the stage, the caller owns the frozen
corpus artifacts (bloom filter, hash index, corpus-common drop sets) and the
ngram machinery that marks text against them.
"""

import json
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple, Protocol

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from finestore.rl.curation import LexicalContaminationEvidence
from rigging.filesystem.storage_path import StoragePath
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset, ShardInfo

from taskcompendium.models import TaskSpec, format_conversation
from taskcompendium.pipeline.execution_telemetry import PhaseTelemetry, SourceTelemetry, execute_phase
from taskcompendium.pipeline.models import IntendedUse
from taskcompendium.pipeline.stages import combine_manifest_counts, write_filtered_shard

TEXT_RENDERER_VERSION = "taskcompendium-format-conversation-v1"
"""The conversation renderer whose output the engine marks; bump when format_conversation changes meaning."""

LEXICAL_CONTAMINATED_REASON = "lexical:contaminated"
"""Filter reason recorded on a kept training row whose text matched the frozen eval corpus."""

POPULATION_SCHEMA = pa.schema([("id", pa.string()), ("text", pa.large_string()), ("partition_id", pa.int64())])
TASK_FLAGS_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("max_overlap", pa.float64()),
        ("contaminated", pa.bool_()),
        ("matched_hashes", pa.list_(pa.uint64())),
        ("matched_eval_ids", pa.list_(pa.string())),
        ("matched_eval_families", pa.list_(pa.string())),
    ]
)
DROP_SCHEMA = pa.schema(
    [("id", pa.string()), ("max_overlap", pa.float64()), ("matched_eval_ids", pa.list_(pa.string()))]
)
EVIDENCE_FILENAME = "measurement.json"


@dataclass(frozen=True)
class EngineIdentity:
    """The frozen-corpus facts every mark depends on; the evidence schema validates their ranges."""

    eval_corpus_version: str
    eval_bloom_uri: str
    eval_hash_index_uri: str
    feature_filter_version: int
    ngram_length: int
    overlap_threshold: float
    min_matched_features: int


class MarkedPopulation(NamedTuple):
    """One engine mark of a text population.

    ``flags_dir`` holds parquet files in :data:`TASK_FLAGS_SCHEMA`, one per
    population partition; ``drop_set_uris`` records the corpus-common mask
    directories the engine marked against.
    """

    flags_dir: str
    drop_set_uris: tuple[str, ...]


class DecontaminationEngine(Protocol):
    """Marks task text against one frozen eval corpus.

    ``identity`` names the corpus version and every frozen artifact the marks
    depend on. ``mark_population`` runs inside the caller's Zephyr context,
    builds whatever source-local corpus-common mask it needs, attributes every
    match through the corpus hash index, and must not observe or modify the
    audit.
    """

    @property
    def identity(self) -> EngineIdentity: ...

    def mark_population(
        self, population: "TextPopulation", output_path: str, *, context: ZephyrContext
    ) -> MarkedPopulation: ...


@dataclass(frozen=True)
class TextPopulation:
    """The rendered text of one source's kept population, partitioned for marking."""

    source: str
    path: str
    task_count: int


def rendered_population_row(row: dict[str, Any]) -> dict[str, Any]:
    """Render one kept audit row to a population text record; reject rows pass through unmapped."""
    if row["filter_status"] != "keep":
        return {"id": row["task_id"], "text": None, "partition_id": None}
    task = TaskSpec.model_validate_json(row["task_json"])
    return {"id": task.id, "text": format_conversation(task.context.events), "partition_id": None}


def write_population_shard(rows: Iterator[dict[str, Any]], shard: ShardInfo, *, output_path: str) -> Iterator[dict]:
    """Write one population partition of rendered kept rows and report its row count."""
    texts = [row for row in rows if row["text"] is not None]
    path = StoragePath(output_path) / f"part-{shard.shard_idx:05d}.parquet"
    table = pa.Table.from_pylist(
        [{"id": row["id"], "text": row["text"], "partition_id": shard.shard_idx} for row in texts],
        schema=POPULATION_SCHEMA,
    )
    with path.open("wb", auto_mkdir=True) as stream:
        pq.write_table(table, stream, compression="zstd")
    yield {"population_rows": len(texts)}


@dataclass(frozen=True)
class LexicalStageResult:
    """One stage run: the measurement and the audit the caller's next stage should read."""

    evidence: LexicalContaminationEvidence
    checked: str


def _flag_files(directory: str) -> list[StoragePath]:
    """The sorted parquet files of one flags or population directory."""
    return [StoragePath(name) for name in sorted(str(path) for path in (StoragePath(directory) / "*.parquet").glob())]


@dataclass(frozen=True)
class PopulationSummary:
    """The measurement statistics of one set of per-task flags."""

    measured_tasks: int
    contaminated_tasks: int
    contaminated_fraction: float
    max_overlap: float
    mean_max_overlap: float
    p95_max_overlap: float
    matched_evals: dict[str, int]


def summarize_flags(rows: Iterable[Mapping[str, Any]]) -> PopulationSummary:
    """Summarize per-task flag rows: counts, overlap statistics and matched eval families."""
    overlaps: list[float] = []
    contaminated = 0
    matched_evals: Counter[str] = Counter()
    for row in rows:
        overlaps.append(row["max_overlap"])
        if not row["contaminated"]:
            continue
        contaminated += 1
        matched_evals.update(row["matched_eval_families"])
    if not overlaps:
        raise ValueError("The flags hold no rows")
    return PopulationSummary(
        measured_tasks=len(overlaps),
        contaminated_tasks=contaminated,
        contaminated_fraction=contaminated / len(overlaps),
        max_overlap=max(overlaps),
        mean_max_overlap=float(np.mean(overlaps)),
        p95_max_overlap=float(np.quantile(overlaps, 0.95)),
        matched_evals=dict(matched_evals),
    )


def population_summary(flags_dir: str) -> PopulationSummary:
    """Summarize the per-task flags written below one directory."""

    def rows() -> Iterator[dict[str, Any]]:
        for path in _flag_files(flags_dir):
            with path.open("rb") as stream:
                for batch in pq.ParquetFile(stream).iter_batches(
                    columns=["max_overlap", "contaminated", "matched_eval_families"]
                ):
                    yield from batch.to_pylist()

    return summarize_flags(rows())


def contaminated_ids(flags_dir: str) -> frozenset[str]:
    """Collect the marked task IDs whose text matched the frozen eval corpus."""
    contaminated: set[str] = set()
    for path in _flag_files(flags_dir):
        with path.open("rb") as stream:
            for batch in pq.ParquetFile(stream).iter_batches(columns=["id", "contaminated"]):
                for row_id, flag in zip(
                    batch.column("id").to_pylist(), batch.column("contaminated").to_pylist(), strict=True
                ):
                    if flag:
                        contaminated.add(row_id)
    return frozenset(contaminated)


def copy_flags(flags_dir: str, target_dir: str) -> None:
    """Publish the engine's per-task flags verbatim as the ``lexical/task_flags`` view."""
    source = StoragePath(flags_dir)
    target = StoragePath(target_dir)
    if target.exists():
        target.rmtree()
    for path in _flag_files(str(source)):
        with path.open("rb") as reader, (target / path.name).open("wb", auto_mkdir=True) as writer:
            while chunk := reader.read(1024 * 1024):
                writer.write(chunk)


def write_drop_manifest(flags_dir: str, view_path: str) -> None:
    """Write the contaminated rows' drop manifest as the ``lexical/drop_manifest.parquet`` view."""
    rows = []
    for path in _flag_files(flags_dir):
        with path.open("rb") as stream:
            for batch in pq.ParquetFile(stream).iter_batches(
                columns=["id", "max_overlap", "contaminated", "matched_eval_ids"]
            ):
                for row_id, overlap, flag, eval_ids in zip(
                    batch.column("id").to_pylist(),
                    batch.column("max_overlap").to_pylist(),
                    batch.column("contaminated").to_pylist(),
                    batch.column("matched_eval_ids").to_pylist(),
                    strict=True,
                ):
                    if flag:
                        rows.append({"id": row_id, "max_overlap": overlap, "matched_eval_ids": eval_ids})
    with StoragePath(view_path).open("wb", auto_mkdir=True) as stream:
        pq.write_table(pa.Table.from_pylist(rows, schema=DROP_SCHEMA), stream, compression="zstd")


def lexical_evidence(
    summary: PopulationSummary,
    identity: EngineIdentity,
    *,
    dataset_id: str,
    dataset_revision: str,
    converter_revision: str,
    drop_set_uris: list[str],
    task_flags_uri: str,
    evidence_url: str,
) -> LexicalContaminationEvidence:
    """Build the typed contamination evidence; the evidence schema validates the identity's ranges."""
    return LexicalContaminationEvidence(
        version=1,
        input_dataset_id=dataset_id,
        input_dataset_revision=dataset_revision,
        converter_revision=converter_revision,
        text_renderer_version=TEXT_RENDERER_VERSION,
        measured_tasks=summary.measured_tasks,
        contaminated_tasks=summary.contaminated_tasks,
        contaminated_fraction=summary.contaminated_fraction,
        max_overlap=summary.max_overlap,
        mean_max_overlap=summary.mean_max_overlap,
        p95_max_overlap=summary.p95_max_overlap,
        matched_evals=summary.matched_evals,
        eval_corpus_version=identity.eval_corpus_version,
        eval_bloom_uri=identity.eval_bloom_uri,
        eval_hash_index_uri=identity.eval_hash_index_uri,
        feature_filter_version=identity.feature_filter_version,
        ngram_length=identity.ngram_length,
        overlap_threshold=identity.overlap_threshold,
        min_matched_features=identity.min_matched_features,
        drop_set_uris=drop_set_uris,
        task_flags_uri=task_flags_uri,
        evidence_url=evidence_url,
        receipt=None,
    )


def reject_contaminated_row(row: dict[str, Any], contaminated: frozenset[str]) -> dict[str, Any]:
    """Reject one kept audit row whose text matched the frozen eval corpus; every other row passes."""
    if row["filter_status"] != "keep" or row["task_id"] not in contaminated:
        return row
    return {
        **row,
        "filter_status": "reject",
        "filter_reasons": [*(row["filter_reasons"] or []), LEXICAL_CONTAMINATED_REASON],
    }


def _kept_row_count(audit_path: str) -> int:
    """The kept disposition count of one filtered audit, from its stage manifest."""
    with (StoragePath(audit_path) / "manifest.json").open("rt") as stream:
        dispositions = json.load(stream)["dispositions"]
    return int(dispositions.get("keep", 0))


def _build_population(
    audit_path: str, output_path: str, *, source_name: str, context: ZephyrContext, telemetry: PhaseTelemetry
) -> TextPopulation | None:
    """Render the kept rows of one checked audit into the engine's text population."""
    storage = StoragePath(output_path)
    if storage.exists():
        storage.rmtree()
    dataset = (
        Dataset.from_files(str(StoragePath(audit_path) / "audit/part-*.parquet"))
        .load_parquet(columns=["task_id", "task_json", "filter_status"])
        .map(rendered_population_row)
        .map_shard(partial(write_population_shard, output_path=str(storage)))
    )
    results = execute_phase(context, dataset, telemetry=telemetry, operation="lexical-text").results
    count = sum(result["population_rows"] for result in results)
    if count == 0:
        return None
    return TextPopulation(source=source_name, path=str(storage), task_count=count)


def _gate_contaminated(
    audit_path: str,
    output_path: str,
    contaminated: frozenset[str],
    *,
    context: ZephyrContext,
    telemetry: PhaseTelemetry,
) -> str:
    """Rewrite one checked audit with contaminated kept rows rejected; return the gated audit path."""
    storage = StoragePath(output_path)
    if storage.exists():
        storage.rmtree()
    dataset = (
        Dataset.from_files(str(StoragePath(audit_path) / "audit/part-*.parquet"))
        .load_parquet()
        .map(partial(reject_contaminated_row, contaminated=contaminated))
        .map_shard(partial(write_filtered_shard, output=storage))
    )
    counts = combine_manifest_counts(
        execute_phase(context, dataset, telemetry=telemetry, operation="lexical-gate").results
    )
    with (storage / "manifest.json").open("wt", auto_mkdir=True) as stream:
        json.dump({**counts, "lexical_contaminated": len(contaminated)}, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return str(storage)


def decontaminate_population(
    audit_path: str,
    view_path: str,
    scratch_path: str,
    *,
    source_name: str,
    dataset_id: str,
    dataset_revision: str,
    converter_revision: str,
    intended_use: IntendedUse,
    engine: DecontaminationEngine,
    context: ZephyrContext,
    telemetry: SourceTelemetry,
) -> LexicalStageResult | None:
    """Run the lexical stage over one checked audit and return its result, or ``None`` with no kept rows.

    The population, marks and gated audit live below ``scratch_path``; the
    measurement, per-task flags and drop manifest are published below
    ``view_path``. A training source's checked audit is rewritten with
    contaminated rows rejected; every other use keeps the audit unchanged.
    """
    kept = _kept_row_count(audit_path)
    if kept == 0:
        return None
    scratch = StoragePath(scratch_path)
    with telemetry.phase("lexical_text") as phase:
        population = _build_population(
            audit_path, str(scratch / "population"), source_name=source_name, context=context, telemetry=phase
        )
        if population is None:
            return None
    with telemetry.phase("lexical_mark") as phase:
        marked = engine.mark_population(population, str(scratch / "marks"), context=context)
    summary = population_summary(marked.flags_dir)
    if summary.measured_tasks != population.task_count:
        raise ValueError(f"The engine flagged {summary.measured_tasks} of {population.task_count} population rows")
    if not marked.drop_set_uris:
        raise ValueError("The engine recorded no drop-set directories for the marked population")
    flags_view = str(StoragePath(view_path) / "task_flags")
    copy_flags(marked.flags_dir, flags_view)
    write_drop_manifest(marked.flags_dir, str(StoragePath(view_path) / "drop_manifest.parquet"))
    evidence = lexical_evidence(
        summary,
        engine.identity,
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
        converter_revision=converter_revision,
        drop_set_uris=list(marked.drop_set_uris),
        task_flags_uri=flags_view,
        evidence_url=str(StoragePath(view_path) / EVIDENCE_FILENAME),
    )
    with (StoragePath(view_path) / EVIDENCE_FILENAME).open("wt", auto_mkdir=True) as stream:
        json.dump(evidence.model_dump(mode="json"), stream, indent=2, sort_keys=True)
        stream.write("\n")
    contaminated = contaminated_ids(marked.flags_dir)
    checked = audit_path
    if intended_use is IntendedUse.TRAIN and contaminated:
        with telemetry.phase("lexical_gate") as phase:
            checked = _gate_contaminated(
                audit_path, str(scratch / "audit"), contaminated, context=context, telemetry=phase
            )
    return LexicalStageResult(evidence=evidence, checked=checked)
