# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run raw data through DataKit, fast-track training, and in-run evaluation.

The DataKit pass uses the production DAG at its smoke scale. The second pass
reads the resulting store and builds the training mixture. This split keeps the
store cache reusable when only the model or mixture changes.
"""

import argparse
import hashlib
import logging
from dataclasses import replace
from enum import StrEnum

import pyarrow as pa
import pyarrow.parquet as pq
from fray.cluster import ResourceConfig
from levanter.tokenizers import load_tokenizer, tokenizer_content_hash
from marin.datakit.normalize import normalize_step
from marin.execution.lazy import run
from marin.execution.step_spec import StepSpec
from rigging.filesystem.storage_path import StoragePath
from rigging.log_setup import configure_logging
from zephyr.context import ZephyrContext
from zephyr.runners import SubprocessRunner

from experiments.datakit.reference_pipeline import (
    QUALITY_MODEL_VERSION,
    SAMPLE_PREFIX,
    SAMPLE_SOURCES,
    SMOKE_SCALE,
    ClusterConfig,
    PipelineScale,
    PoolConfig,
    TokenizerSpec,
    materialize_reference_store,
    quality_model_path,
    sample_sources,
    select_sources,
)
from experiments.datakit.store.mixture import MixtureWeighting, log_store_summary, store_mixture
from experiments.grug.fast_track.launch import (
    H100_LADDER_SIZES,
    SEQ_LEN,
    V16384_TOKENIZER,
    V16384_VOCAB,
    MatchMode,
    build_h100_ladder_run,
    submit_to_cluster,
)

logger = logging.getLogger(__name__)

_REPEATED_DOCUMENT_PARAGRAPH = """Data pipelines must preserve the intended distribution of source documents.
Each stage records its inputs and outputs so a later run can use the same data.
Exact duplicate removal keeps one copy of a repeated document. Domain and quality
labels then describe the remaining content. This text includes enough separate words
for tokenization, embedding, and MinHash processing in a small integration test."""
REPEATED_DOCUMENT = "\n\n".join([_REPEATED_DOCUMENT_PARAGRAPH] * 96)


class SourceMode(StrEnum):
    SAMPLE = "sample"
    REGISTRY = "registry"
    REPEATED_DOCUMENT = "repeated_document"


class Stage(StrEnum):
    DATAKIT = "datakit"
    TRAIN = "train"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return parsed


def repeated_document_sources(count: int, scale: PipelineScale) -> dict[str, StepSpec]:
    """Build a raw source with one document repeated under different source IDs."""
    if count < 2:
        raise ValueError(f"repeated-document count must be at least 2, got {count}")

    document_hash = hashlib.sha256(REPEATED_DOCUMENT.encode()).hexdigest()

    def write_raw(output_path: str) -> None:
        destination = StoragePath(output_path) / "part-00000.parquet"
        destination.parent.mkdirs()
        table = pa.table(
            {
                "id": [f"repeat-{index:06d}" for index in range(count)],
                "text": [REPEATED_DOCUMENT] * count,
            }
        )
        with destination.open("wb") as output:
            pq.write_table(table, output, compression="zstd")

    raw = StepSpec(
        name="datakit/fast_track/repeated_document/raw",
        hash_attrs={"count": count, "document_sha256": document_hash},
        fn=write_raw,
    )
    normalized = normalize_step(
        name="datakit/fast_track/repeated_document/normalize",
        download=raw,
        file_extensions=(".parquet",),
        target_partition_bytes=1024 * 1024,
        max_workers=scale.pool.n_workers,
        worker_resources=scale.pool.worker,
    )
    return {"repeated_document": normalized}


def repeated_document_scale(pool_workers: int) -> PipelineScale:
    """Return the production pipeline shape for a one-document result."""
    return replace(
        SMOKE_SCALE,
        cluster=ClusterConfig(k_train=1, k_views=(), cluster_view=1),
        pool=PoolConfig(n_workers=pool_workers, worker=SMOKE_SCALE.pool.worker),
        n_per_source_for_sample=1,
        dedup_max_parallelism=1,
        train_centroids_resources=ResourceConfig.with_cpu(cpu=1, ram="2g"),
    )


def _selected_sources(
    *,
    mode: SourceMode,
    source_names: list[str] | None,
    all_sources: bool,
    sample_prefix: str,
    repeated_document_count: int,
    scale: PipelineScale,
) -> dict[str, StepSpec]:
    if mode is SourceMode.REPEATED_DOCUMENT:
        return repeated_document_sources(repeated_document_count, scale)
    if mode is SourceMode.REGISTRY:
        if all_sources:
            raise ValueError("--sources=all is not allowed in registry mode; select explicit source names")
        if not source_names:
            raise ValueError("--sources is required in registry mode")
        return select_sources(source_names)
    return sample_sources(sample_prefix, None if all_sources else source_names or list(SAMPLE_SOURCES))


def run_pipeline(args: argparse.Namespace) -> None:
    """Run the two pipeline passes from parsed command-line arguments."""
    pool_workers = args.pool_workers
    if args.source_mode is SourceMode.REPEATED_DOCUMENT:
        scale = repeated_document_scale(pool_workers)
    else:
        scale = replace(SMOKE_SCALE, pool=PoolConfig(n_workers=pool_workers, worker=SMOKE_SCALE.pool.worker))

    names = None
    if args.sources and args.sources != "all":
        names = [name.strip() for name in args.sources.split(",") if name.strip()]
    sources = _selected_sources(
        mode=args.source_mode,
        source_names=names,
        all_sources=args.sources == "all",
        sample_prefix=args.sample_prefix,
        repeated_document_count=args.repeated_document_count,
        scale=scale,
    )
    tokenizer = load_tokenizer(V16384_TOKENIZER)
    if len(tokenizer) != V16384_VOCAB:
        raise ValueError(
            f"tokenizer {V16384_TOKENIZER!r} has {len(tokenizer)} entries; "
            f"the fast-track model requires {V16384_VOCAB}"
        )
    if args.source_mode is SourceMode.REPEATED_DOCUMENT and len(tokenizer.encode(REPEATED_DOCUMENT)) < SEQ_LEN:
        raise ValueError("the repeated document must contain at least one fast-track training sequence")
    tokenizer_spec = TokenizerSpec(V16384_TOKENIZER, tokenizer_content_hash(V16384_TOKENIZER))

    with ZephyrContext(
        name=f"fast-track-data-{args.run_id}",
        resources=scale.pool.worker,
        max_workers=scale.pool.n_workers,
        stage_runner_factory=SubprocessRunner,
    ) as zephyr_context:
        store = materialize_reference_store(
            sources,
            quality_model=args.quality_model,
            quality_model_version=args.quality_model_version,
            scale=scale,
            zephyr_context=zephyr_context,
            tokenizer=tokenizer_spec,
            max_concurrent=args.max_concurrent,
        )
    log_store_summary(store)
    if args.stop_after is Stage.DATAKIT:
        return

    training_data = store_mixture(store, weighting=args.weighting, min_tokens_per_component=SEQ_LEN)
    training = build_h100_ladder_run(
        run_id=args.run_id,
        size=args.size,
        match=args.match,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        version=args.version,
        tokenizer=store.tokenizer,
        vocab_size=V16384_VOCAB,
        training_data=training_data,
        no_eval=args.no_eval,
        dense=args.dense,
        save_checkpoints=args.save_checkpoints,
    )
    run(training, max_concurrent=args.max_concurrent)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="Artifact and W&B run identifier")
    parser.add_argument("--size", choices=H100_LADDER_SIZES, default="d512", help="Fast-track model size")
    parser.add_argument("--version", default="dev", help="Training artifact version")
    parser.add_argument(
        "--source-mode",
        type=SourceMode,
        choices=list(SourceMode),
        default=SourceMode.SAMPLE,
        help="Input source type",
    )
    parser.add_argument("--sample-prefix", default=SAMPLE_PREFIX, help="Normalized test sample root")
    parser.add_argument("--sources", help="Comma-separated names; 'all' is valid only in sample mode")
    parser.add_argument(
        "--repeated-document-count",
        type=_positive_int,
        default=1_000,
        help="Number of raw copies in repeated-document mode",
    )
    parser.add_argument("--quality-model", default=quality_model_path(), help="DataKit quality model directory")
    parser.add_argument(
        "--quality-model-version",
        default=QUALITY_MODEL_VERSION,
        help="Stable identity for the quality model bytes",
    )
    parser.add_argument(
        "--weighting",
        type=MixtureWeighting,
        choices=list(MixtureWeighting),
        default=MixtureWeighting.TOKEN_PROPORTIONAL,
    )
    parser.add_argument("--match", type=MatchMode, choices=list(MatchMode), default=MatchMode.DATA, help="Run budget")
    parser.add_argument("--batch-size", type=_positive_int, help="Global sequence batch")
    parser.add_argument("--num-steps", type=_positive_int, help="Explicit training step count")
    parser.add_argument("--dense", action="store_true", help="Use the dense fast-track model")
    parser.add_argument("--no-eval", action="store_true", help="Disable in-run evaluation")
    parser.add_argument("--save-checkpoints", action="store_true", help="Keep the final training checkpoint")
    parser.add_argument(
        "--pool-workers",
        type=_positive_int,
        default=16,
        help="Worker count for DataKit derived stages and repeated-document normalization",
    )
    parser.add_argument("--max-concurrent", type=_positive_int, default=8, help="Maximum concurrent steps")
    parser.add_argument(
        "--stop-after",
        type=Stage,
        choices=list(Stage),
        default=Stage.TRAIN,
        help="Last pipeline stage to run",
    )
    parser.add_argument("--run", action="store_true", help="Run the pipeline in the current environment")
    parser.add_argument("--submit", action="store_true", help="Submit an Iris coordinator job")
    return parser


def main() -> None:
    args = _parser().parse_args()
    configure_logging(logging.INFO)
    if args.submit:
        submit_to_cluster(
            args.run_id,
            module="experiments.grug.fast_track.data_pipeline",
            job_name=f"{args.run_id}-data-coord",
            dependency_groups=("cpu", "datakit"),
            coordinator_args=("--cpu", "2", "--memory", "8GB", "--disk", "32GB"),
            require_wandb=args.stop_after is Stage.TRAIN,
            allow_disabled_wandb=True,
        )
    if not args.run:
        logger.info("No work started. Add --submit for Iris or --run in an Iris environment.")
        return
    run_pipeline(args)


if __name__ == "__main__":
    main()
