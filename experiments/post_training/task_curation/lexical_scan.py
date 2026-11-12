# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind Atlas populations to Datakit DF filtering, lexical marks, and full attribution."""

import argparse
import json
import logging
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from fray.types import ResourceConfig
from marin.datakit.decon import (
    FEATURE_FILTER_VERSION,
    NGramConfig,
    build_all_source_drop_sets,
    decon_to_parquet,
)
from marin.datakit.normalize import NormalizedData
from marin.execution.lazy import ArtifactStep, StepContext, run
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath, prefix_join
from taskcompendium.identity import canonical_sha256
from zephyr.context import ZephyrContext

from experiments.datakit.decontam.config import (
    GLOBAL_DF_COMMON_MIN_ABS,
    GLOBAL_DF_COMMON_MIN_SOURCES,
    GLOBAL_DF_SAMPLE_DOCS,
    SOURCE_DF_COMMON_FRAC,
    SOURCE_DF_COMMON_MIN_ABS,
    SOURCE_DF_SAMPLE_DOCS,
)
from experiments.datakit.decontam.viewer.export_reference_run import _eval_family, _hash_to_evals
from experiments.post_training.task_curation.campaign import CampaignRuntime
from experiments.post_training.task_curation.lexical_artifacts import (
    LexicalDropSets,
    LexicalMeasurements,
    SourceLexicalMarks,
    SourceTextPopulation,
)
from experiments.post_training.task_curation.local import artifact_prefix

logger = logging.getLogger(__name__)
NGRAM = NGramConfig(ngram_length=13, stride=0, overlap_threshold=0.5, min_matched_features=2)
WORKER_RESOURCES = ResourceConfig(cpu=2, ram="4g")
DROP_SCHEMA = pa.schema(
    [("id", pa.string()), ("max_overlap", pa.float64()), ("matched_eval_ids", pa.list_(pa.string()))]
)


@dataclass(frozen=True)
class DropConfig:
    population_paths: list[str]
    bloom_path: str
    output_path: str
    max_workers: int


@dataclass(frozen=True)
class MarkConfig:
    population_path: str
    drop_path: str
    bloom_path: str
    output_path: str
    max_workers: int


@dataclass(frozen=True)
class ReportConfig:
    mark_paths: list[str]
    drop_path: str
    bloom_path: str
    eval_corpus_version: str
    population_set: str
    output_path: str


def population_key(source_id: str) -> str:
    return canonical_sha256(source_id)[:20]


def build_drop_sets(config: DropConfig, *, campaign: CampaignRuntime) -> LexicalDropSets:
    populations = [SourceTextPopulation.raw_load(path) for path in config.population_paths]
    result = build_all_source_drop_sets(
        sources=[(population_key(p.source_id), p.main_output_dir) for p in populations],
        prebuilt_bloom_dir=config.bloom_path,
        output_path=config.output_path,
        ngram=NGRAM,
        sample_docs=SOURCE_DF_SAMPLE_DOCS,
        common_frac=SOURCE_DF_COMMON_FRAC,
        common_min_abs=SOURCE_DF_COMMON_MIN_ABS,
        global_sample_docs=GLOBAL_DF_SAMPLE_DOCS,
        global_common_min_abs=GLOBAL_DF_COMMON_MIN_ABS,
        global_common_min_sources=GLOBAL_DF_COMMON_MIN_SOURCES,
        worker_resources=WORKER_RESOURCES,
        max_workers=config.max_workers,
        zephyr_context=campaign.context,
    )
    return LexicalDropSets(path=config.output_path, attributes=result)


def mark_population(config: MarkConfig, *, campaign: CampaignRuntime) -> SourceLexicalMarks:
    population = SourceTextPopulation.raw_load(config.population_path)
    drops = LexicalDropSets.raw_load(config.drop_path).attributes
    result = decon_to_parquet(
        normalized_data=NormalizedData(main_output_dir=population.main_output_dir, dup_output_dir="", counters={}),
        prebuilt_bloom_dir=config.bloom_path,
        output_path=config.output_path,
        ngram=NGRAM,
        drop_set_dirs=[
            prefix_join(str(drops.output_dir), population_key(population.source_id)),
            str(drops.global_output_dir),
        ],
        flagged_sample_size=0,
        worker_resources=WORKER_RESOURCES,
        max_workers=config.max_workers,
        zephyr_context=campaign.context,
    )
    return SourceLexicalMarks(
        path=config.output_path,
        source_id=population.source_id,
        population_path=config.population_path,
        attributes=result,
    )


def mark_files(mark: SourceLexicalMarks) -> list[str]:
    return sorted(str(p) for p in StoragePath(prefix_join(str(mark.attributes.main_output_dir), "*.parquet")).glob())


def measure_marks(config: ReportConfig) -> LexicalMeasurements:
    """Attribute every marked task through the complete causal hash index, without sampling."""
    marks = [SourceLexicalMarks.raw_load(path) for path in config.mark_paths]
    index_paths = {str(mark.attributes.eval_hash_index_path) for mark in marks}
    if len(index_paths) != 1:
        raise ValueError(f"Marks use different eval indices: {index_paths}")
    index_path = next(iter(index_paths))
    needed_hashes = set()
    for mark in marks:
        for path in mark_files(mark):
            with StoragePath(path).open("rb") as stream:
                for batch in pq.ParquetFile(stream).iter_batches(columns=["matched_hashes"]):
                    for hashes in batch.column(0).to_pylist():
                        needed_hashes.update(hashes)
    logger.info("Joining %d distinct causal hashes against the complete eval index", len(needed_hashes))
    # Reuse the reference report's full streaming index join, never its sampled gallery.
    hash_evals = _hash_to_evals(index_path, needed_hashes) if needed_hashes else {}
    drops = LexicalDropSets.raw_load(config.drop_path).attributes
    measurements = {}
    for mark in marks:
        population = SourceTextPopulation.raw_load(mark.population_path)
        output = prefix_join(config.output_path, population_key(mark.source_id))
        flags_uri = prefix_join(output, "task_flags")
        drop_uri = prefix_join(output, "drop_manifest.parquet")
        overlaps = []
        contaminated = 0
        matched_evals = Counter()
        with StoragePath(drop_uri).open("wb", auto_mkdir=True) as drop_stream:
            with pq.ParquetWriter(drop_stream, DROP_SCHEMA, compression="zstd") as drop_writer:
                for part, path in enumerate(mark_files(mark)):
                    flags_path = prefix_join(flags_uri, f"part-{part:06d}.parquet")
                    with StoragePath(path).open("rb") as stream:
                        with StoragePath(flags_path).open("wb", auto_mkdir=True) as flags_stream:
                            writer = None
                            try:
                                for batch in pq.ParquetFile(stream).iter_batches(batch_size=4096):
                                    rows = batch.to_pylist()
                                    eval_ids = [
                                        sorted({e for h in row["matched_hashes"] for e in hash_evals.get(h, ())})
                                        for row in rows
                                    ]
                                    families = [sorted({_eval_family(e) for e in ids}) for ids in eval_ids]
                                    table = pa.Table.from_batches([batch])
                                    table = table.append_column(
                                        "matched_eval_ids", pa.array(eval_ids, pa.list_(pa.string()))
                                    )
                                    table = table.append_column(
                                        "matched_eval_families", pa.array(families, pa.list_(pa.string()))
                                    )
                                    if writer is None:
                                        writer = pq.ParquetWriter(flags_stream, table.schema, compression="zstd")
                                    writer.write_table(table)
                                    dropped = []
                                    for row, ids, family in zip(rows, eval_ids, families, strict=True):
                                        overlaps.append(row["max_overlap"])
                                        if not row["contaminated"]:
                                            continue
                                        contaminated += 1
                                        matched_evals.update(family)
                                        dropped.append(
                                            {"id": row["id"], "max_overlap": row["max_overlap"], "matched_eval_ids": ids}
                                        )
                                    if dropped:
                                        drop_writer.write_table(pa.Table.from_pylist(dropped, schema=DROP_SCHEMA))
                            finally:
                                if writer is not None:
                                    writer.close()
        if len(overlaps) != population.task_count:
            raise ValueError(f"{mark.source_id}: {len(overlaps)} marks for {population.task_count} converted tasks")
        # Publication readback must establish private TaskCompendium output before admission.
        # Both clean raw inputs and flagged populations are measurements at this stage.
        measurements[mark.source_id] = dict(
            version=1,
            input_dataset_id=population.dataset_id,
            input_dataset_revision=population.dataset_revision,
            converter_revision=population.converter_revision,
            text_renderer_version=population.text_renderer_version,
            measured_tasks=population.task_count,
            contaminated_tasks=contaminated,
            contaminated_fraction=contaminated / population.task_count,
            max_overlap=max(overlaps),
            mean_max_overlap=float(np.mean(overlaps)),
            p95_max_overlap=float(np.quantile(overlaps, 0.95)),
            matched_evals=dict(matched_evals),
            eval_corpus_version=config.eval_corpus_version,
            eval_bloom_uri=config.bloom_path,
            eval_hash_index_uri=index_path,
            feature_filter_version=FEATURE_FILTER_VERSION,
            ngram_length=NGRAM.ngram_length,
            overlap_threshold=NGRAM.overlap_threshold,
            min_matched_features=NGRAM.min_matched_features,
            drop_set_uris=[
                prefix_join(str(drops.output_dir), population_key(mark.source_id)),
                str(drops.global_output_dir),
            ],
            task_flags_uri=flags_uri,
            drop_manifest_uri=drop_uri,
            evidence_url=prefix_join(output, "measurement.json"),
            receipt=None,
        )
        with StoragePath(prefix_join(output, "measurement.json")).open("w", auto_mkdir=True) as stream:
            json.dump(measurements[mark.source_id], stream, indent=2, sort_keys=True)
        logger.info("%s: %d/%d contaminated", mark.source_id, contaminated, population.task_count)
    return LexicalMeasurements(
        path=config.output_path,
        population_set=config.population_set,
        source_ids=[mark.source_id for mark in marks],
        measurements=measurements,
    )


def scan_steps(
    populations: list[SourceTextPopulation],
    bloom_path: str,
    corpus_version: str,
    population_set: str,
    version: str,
    max_workers: int,
    runtime: CampaignRuntime,
) -> ArtifactStep:
    """Bind all DF inputs before any mark and one complete attribution after every mark."""
    inputs = [
        ArtifactStep.adopt(
            f"rl-data/atlas/lexical/input/{population_key(p.source_id)}", version, p.path, kind=SourceTextPopulation
        )
        for p in populations
    ]
    bloom = ArtifactStep.adopt("rl-data/atlas/lexical/eval-bloom", version, bloom_path)
    cohort = canonical_sha256(
        sorted((p.source_id, p.dataset_revision, p.converter_revision, p.text_sha256) for p in populations)
    )[:20]
    root = f"rl-data/atlas/lexical/scan/{cohort}"
    drops = ArtifactStep(
        name=f"{root}/df",
        version=version,
        artifact_type=LexicalDropSets,
        deps=[*inputs, bloom],
        runtime_args={"max_workers": max_workers},
        run=partial(build_drop_sets, campaign=runtime),
        build_config=lambda ctx: DropConfig(
            [ctx.artifact_path(p) for p in inputs],
            ctx.artifact_path(bloom),
            ctx.output_path,
            ctx.runtime_arg("max_workers"),
        ),
    )
    marks = []
    for population, handle in zip(populations, inputs, strict=True):

        def config(ctx: StepContext, handle=handle):
            return MarkConfig(
                ctx.artifact_path(handle),
                ctx.artifact_path(drops),
                ctx.artifact_path(bloom),
                ctx.output_path,
                ctx.runtime_arg("max_workers"),
            )

        marks.append(
            ArtifactStep(
                name=f"{root}/marks/{population_key(population.source_id)}",
                version=version,
                artifact_type=SourceLexicalMarks,
                deps=[handle, drops, bloom],
                runtime_args={"max_workers": max_workers},
                run=partial(mark_population, campaign=runtime),
                build_config=config,
            )
        )
    return ArtifactStep(
        name=f"{root}/report",
        version=version,
        artifact_type=LexicalMeasurements,
        deps=[*marks, drops, bloom],
        run=measure_marks,
        build_config=lambda ctx: ReportConfig(
            [ctx.artifact_path(mark) for mark in marks],
            ctx.artifact_path(drops),
            ctx.artifact_path(bloom),
            corpus_version,
            population_set,
            ctx.output_path,
        ),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--populations", type=Path, required=True)
    parser.add_argument("--source-ids", type=Path, required=True)
    parser.add_argument("--bloom-path", required=True)
    parser.add_argument("--eval-corpus-version", required=True)
    parser.add_argument("--population-set", choices=["pilot", "retained"], required=True)
    parser.add_argument("--artifact-version", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--max-workers", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    configure_coreweave_s3()
    populations = [SourceTextPopulation.model_validate(row) for row in json.loads(args.populations.read_text())]
    expected = json.loads(args.source_ids.read_text())
    if sorted(expected) != sorted(p.source_id for p in populations) or len(expected) != len(set(expected)):
        raise ValueError("Normalized populations must cover every requested source exactly once")
    runtime = CampaignRuntime()
    with artifact_prefix(args.artifact_root):
        report = scan_steps(
            populations,
            args.bloom_path,
            args.eval_corpus_version,
            args.population_set,
            args.artifact_version,
            args.max_workers,
            runtime,
        )
        if not args.run:
            args.output.write_text(
                json.dumps({"report": report.path(), "fingerprint": report.fingerprint()}, indent=2) + "\n"
            )
            return
        with ExitStack() as cleanup:
            client = LocalClient(max_threads=max(16, args.max_workers * 2))
            cleanup.callback(client.shutdown)
            cleanup.enter_context(set_current_client(client))
            with ZephyrContext(client=client, max_workers=args.max_workers, resources=WORKER_RESOURCES) as context:
                with runtime.activate(context):
                    result = run(report, max_concurrent=2)[0]
        args.output.write_text(json.dumps(result.model_dump(mode="json"), indent=2) + "\n")


if __name__ == "__main__":
    main()
