# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical end-to-end reference pipeline: raw sample -> datakit -> pretrain -> eval.

One blessed, minimal path that runs a testbed sample through full datakit processing, a
tiny pretraining run over the resulting per-(cluster, quality) store, and an eval readout.
It is the "does the whole path still work, and what does this change do to the numbers"
harness: swap a data source or a model knob, re-run, and read a comparable eval report with
everything else held fixed.

Two ``StepRunner`` passes joined by a dynamic handoff, because the two halves live on
different execution primitives and the store's non-empty bucket set is only known after
datakit runs:

1. Datakit pass -- :func:`reference_datakit_steps` builds the content-addressed ``StepSpec``
   DAG over ``sample_sources(SAMPLE_PREFIX)`` at ``SMOKE_SCALE``; the driver runs it and reads
   the terminal :class:`ClusteredStoreData`.
2. Train + eval pass -- :func:`store_mixture` turns that store into the Levanter
   ``LmDataConfig`` (one ``flat_cache=True`` component per non-empty bucket), the Grug launch
   trainer produces an ``ArtifactStep[LevanterCheckpoint]``, and ``eval_steps`` / ``eval_report``
   produce the readout.

Both passes resume from their own caches: a training-config edit re-fingerprints only
train+eval and reuses the datakit store; a repeated run with unchanged config mints no new
copies. Single-edit points: ``SAMPLE_PREFIX`` (the data source) and ``REFERENCE_MODEL`` (the
model/train config).

Submit on iris (datakit fans out its own Zephyr fleets; the driver is a small CPU job)::

    uv run iris --cluster=cw-rno2a job run --cpu 2 --memory 8GB \\
        --enable-extra-resources --extra datakit \\
        -- python -m experiments.references.reference_training_pipeline \\
            --version dev --stop-after datakit
"""

import argparse
import logging
from dataclasses import replace
from enum import StrEnum

from fray.cluster import ResourceConfig
from fray.types import ANY_REGION
from levanter.data.text.datasets import DatasetComponent, LmDataConfig
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.optim.config import AdamConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.artifact import read_artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext, run
from marin.execution.step_runner import StepRunner
from marin.experiment.namespacing import user_namespaced_name
from marin.training.training import LevanterCheckpoint
from rigging.log_setup import configure_logging

from experiments.datakit.reference_pipeline import (
    QUALITY_MODEL,
    SAMPLE_PREFIX,
    SAMPLE_SOURCES,
    SMOKE_SCALE,
    PoolConfig,
    reference_datakit_steps,
    sample_sources,
)
from experiments.datakit.store.datakit_store import ClusteredStoreData
from experiments.evals.evalchemy.serve_and_eval import ServeSpec
from experiments.evals.evals import core_evals, eval_report, eval_steps
from experiments.grug.base.launch import GrugBaseLaunchConfig, run_grug_base_trial
from experiments.grug.base.model import GrugModelConfig

logger = logging.getLogger(__name__)

# Shared name for the train + eval + report handles. The datakit steps keep their own
# ``datakit/*`` content-addressed names.
REF_NAME = "references/reference-pipeline"

# The datakit quality scorer is region-specific, so its identity enters the datakit hash as
# a stable tag, not the path (see reference_pipeline.py). ``pooled-junkgate2`` is the tag for
# the default ``QUALITY_MODEL`` bytes.
QUALITY_MODEL_VERSION = "pooled-junkgate2"

# A nano model: this harness measures path-liveness and delta-vs-baseline, not absolute
# quality, so it is sized for a fast smoke on a single accelerator, not for signal. vocab_size
# matches the marin/llama3 tokenizer the datakit store is tokenized with.
REFERENCE_MODEL = GrugModelConfig(
    vocab_size=128_256,
    hidden_dim=256,
    intermediate_dim=768,
    num_layers=4,
    num_heads=8,
    num_kv_heads=8,
    max_seq_len=1024,
    head_dim=None,
)
REFERENCE_STEPS = 40
REFERENCE_BATCH_SIZE = 16
REFERENCE_SEED = 0

# Grug's trainer requires an accelerator, so the tiny pretrain runs on one GPU (a run-arg,
# off the checkpoint's identity). CPU-only is not an option for training here; datakit itself
# runs on CPU Zephyr fleets. ``regions=[ANY_REGION]`` is required: the CoreWeave GPU fleet
# advertises no region, so an inherited region would leave the job unschedulable (see
# experiments/tutorials/train_tiny_model.py).
REFERENCE_TRAIN_RESOURCES = ResourceConfig.with_gpu("H100", count=1, cpu=8, disk="128G", ram="64G", regions=[ANY_REGION])

# Eval serves the checkpoint via marin-serve (vLLM) on one GPU, then runs the core MCQ suite
# against its OpenAI endpoint.
REFERENCE_SERVE = ServeSpec(gpu_type="H100", gpu_count=1, tpu_type=None)


class MixtureWeighting(StrEnum):
    """How :func:`store_mixture` weights the per-(cluster, quality) buckets."""

    TOKEN_PROPORTIONAL = "token_proportional"
    UNIFORM = "uniform"


def _bucket_name(cluster_id: int, quality_bucket: int) -> str:
    """The ``cXXqY`` component key, matching the datakit-store bucket convention."""
    return f"c{cluster_id:02d}q{quality_bucket}"


def store_mixture(
    store: ClusteredStoreData,
    *,
    weighting: MixtureWeighting = MixtureWeighting.TOKEN_PROPORTIONAL,
) -> LmDataConfig:
    """One ``DatasetComponent`` per non-empty store bucket, as the Levanter training data config.

    Weights are token-proportional (``bucket.total_tokens``) or uniform; Levanter renormalizes.
    Raises ``ValueError`` if the store has no non-empty buckets, or (under ``TOKEN_PROPORTIONAL``)
    if any bucket has ``total_tokens <= 0``.
    """
    if not store.buckets:
        raise ValueError(f"store at {store.cache_path} has no non-empty buckets; datakit produced no data")

    components: dict[str, DatasetComponent] = {}
    weights: dict[str, float] = {}
    for bucket in store.buckets:
        name = _bucket_name(bucket.cluster_id, bucket.quality_bucket)
        components[name] = DatasetComponent(
            source=None,
            # Absolute s3:// path: Levanter resolves a relative cache_dir against the worker CWD
            # (/app), not the object store, so a relativized path fails to load.
            cache_dir=bucket.path,
            format=TextLmDatasetFormat(),
            tags=[name],
            # The store writes flat caches (part-* + shard_ledger.json at the bucket root, no
            # train/ subdir); TokenizedCache.as_component omits flat_cache, so Levanter would look
            # for <bucket>/train/ and silently drop the component.
            flat_cache=True,
        )
        if weighting is MixtureWeighting.TOKEN_PROPORTIONAL:
            if bucket.total_tokens <= 0:
                # A 0-weight component is silently dropped by Levanter -> a broken store, not a mixture.
                raise ValueError(f"bucket {name} at {bucket.path} has total_tokens={bucket.total_tokens}; expected > 0")
            weights[name] = float(bucket.total_tokens)
        else:
            weights[name] = 1.0

    logger.info(
        "store_mixture: %d buckets, %s weighting, tokenizer=%s",
        len(components),
        weighting.value,
        store.tokenizer,
    )
    return LmDataConfig(
        tokenizer=store.tokenizer,
        cache_dir=None,
        components=components,
        train_weights=weights,
        auto_build_caches=False,
    )


def reference_train_on_store(
    store: ClusteredStoreData,
    *,
    model: GrugModelConfig = REFERENCE_MODEL,
    version: str | None = None,
    weighting: MixtureWeighting = MixtureWeighting.TOKEN_PROPORTIONAL,
    resources: ResourceConfig = REFERENCE_TRAIN_RESOURCES,
) -> ArtifactStep[LevanterCheckpoint]:
    """The reference pretrain over a resolved datakit store, as an ``ArtifactStep[LevanterCheckpoint]``.

    The store's bucket paths and weights are config literals, so they bear identity in the
    fingerprint: a datakit change re-fingerprints training. Uses a fixed ``REFERENCE_STEPS`` step
    count; epochs are unsupported for the store's flat caches (there is no ``<bucket>/train/.stats.json``).
    """
    version = resolve_version(REF_NAME, version)

    def build_config(ctx: StepContext) -> GrugBaseLaunchConfig:
        return GrugBaseLaunchConfig(
            model=model,
            data=store_mixture(store, weighting=weighting),
            output_path=ctx.output_path,
            run_id="reference-pipeline",
            resources=ctx.runtime_arg("train_resources"),
            steps=REFERENCE_STEPS,
            batch_size=REFERENCE_BATCH_SIZE,
            seed=REFERENCE_SEED,
            mp="params=float32,compute=bfloat16,output=bfloat16",
            tracker=WandbConfig(
                project="marin",
                tags=["reference", "pipeline", "e2e"],
                group="reference-pipeline",
                name=None,
                replicate_path=ctx.output_path,
            ),
            optimizer=AdamConfig(
                learning_rate=3e-3,
                weight_decay=0.1,
                warmup=0.05,
                decay=0.2,
            ),
            eval_batch_size=None,  # no in-loop perplexity; the readout is the downstream eval
            steps_per_eval=REFERENCE_STEPS,
        )

    return ArtifactStep(
        name=user_namespaced_name(REF_NAME, version),
        version=version,
        artifact_type=LevanterCheckpoint,
        run=run_grug_base_trial,
        build_config=build_config,
        runtime_args={"train_resources": resources},
    )


class Stage(StrEnum):
    """How far the driver runs."""

    DATAKIT = "datakit"
    TRAIN = "train"
    EVAL = "eval"


def _log_store_summary(store: ClusteredStoreData) -> None:
    total_docs = sum(b.total_elements for b in store.buckets)
    total_tokens = sum(b.total_tokens for b in store.buckets)
    logger.info(
        "datakit store: %d non-empty buckets, %d docs, %d tokens, sources=%s, tokenizer=%s",
        len(store.buckets),
        total_docs,
        total_tokens,
        store.source_names,
        store.tokenizer,
    )
    for bucket in sorted(store.buckets, key=lambda b: (b.cluster_id, b.quality_bucket)):
        logger.info(
            "  %s: docs=%d tokens=%d shards=%d",
            _bucket_name(bucket.cluster_id, bucket.quality_bucket),
            bucket.total_elements,
            bucket.total_tokens,
            bucket.n_shards,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="dev", help="artifact version for the train/eval handles (or 'dev')")
    parser.add_argument("--sample-prefix", default=SAMPLE_PREFIX, help="testbed sample root")
    parser.add_argument(
        "--sources",
        default=None,
        help="comma-separated source names, or 'all' to discover every source; default = curated SAMPLE_SOURCES subset",
    )
    parser.add_argument("--quality-model", default=QUALITY_MODEL, help="pooled fast-transformer scorer + calib dir")
    parser.add_argument(
        "--quality-model-version", default=QUALITY_MODEL_VERSION, help="stable identity tag for --quality-model"
    )
    parser.add_argument(
        "--weighting", type=MixtureWeighting, choices=list(MixtureWeighting), default=MixtureWeighting.TOKEN_PROPORTIONAL
    )
    parser.add_argument(
        "--stop-after",
        type=Stage,
        choices=list(Stage),
        default=Stage.TRAIN,
        help="run through this stage; default 'train' (eval needs an HF export the Grug orbax "
        "checkpoint does not yet produce -- pass '--stop-after eval' once that is wired)",
    )
    parser.add_argument("--pool-workers", type=int, default=None, help="datakit per-stage worker count (override scale)")
    parser.add_argument("--max-concurrent", type=int, default=8, help="max steps a StepRunner walks at once")
    args = parser.parse_args()

    configure_logging(logging.INFO)

    scale = SMOKE_SCALE
    if args.pool_workers is not None:
        scale = replace(scale, pool=PoolConfig(n_workers=args.pool_workers, worker=scale.pool.worker))

    # Default to the curated diverse subset (matching the datakit --mode sample default), so the
    # smallest testbed is a quick end-to-end run; "all" discovers every source in the sample prefix.
    if args.sources is None:
        names = list(SAMPLE_SOURCES)
    elif args.sources == "all":
        names = None
    else:
        names = [s.strip() for s in args.sources.split(",") if s.strip()]
    sources = sample_sources(args.sample_prefix, names)

    # --- Pass 1: datakit ----------------------------------------------------------------
    datakit = reference_datakit_steps(
        sources,
        quality_model=args.quality_model,
        quality_model_version=args.quality_model_version,
        scale=scale,
    )
    StepRunner().run(datakit.all_steps, max_concurrent=args.max_concurrent)
    store = read_artifact(datakit.output_buckets.output_path, ClusteredStoreData)
    _log_store_summary(store)
    if args.stop_after is Stage.DATAKIT:
        logger.info("stop-after=datakit; store at %s", store.cache_path)
        return

    # --- Pass 2: train (+ eval) ---------------------------------------------------------
    model = reference_train_on_store(store, model=REFERENCE_MODEL, version=args.version, weighting=args.weighting)
    if args.stop_after is Stage.TRAIN:
        run(model, max_concurrent=args.max_concurrent)
        logger.info("stop-after=train; checkpoint at %s", model.path())
        return

    results = eval_steps(model, core_evals(serve=REFERENCE_SERVE), version=args.version)
    report = eval_report(results, name=REF_NAME, version=args.version)
    run(report, max_concurrent=args.max_concurrent)
    logger.info("eval report at %s", report.path())


if __name__ == "__main__":
    main()
