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
   ``LmDataConfig`` (one ``flat_cache=True`` component per bucket with one model
   sequence), the Grug launch trainer produces an ``ArtifactStep[LevanterCheckpoint]``,
   and ``eval_steps`` / ``eval_report`` produce the readout.

Both passes resume from their own caches: a training-config edit re-fingerprints only
train+eval and reuses the datakit store; a repeated run with unchanged config mints no new
copies. Single-edit points: ``SAMPLE_PREFIX`` (the data source) and ``REFERENCE_MODEL`` (the
model/train config).

Submit on iris (datakit fans out its own Zephyr fleets; the driver is a small CPU job)::

    uv run iris --cluster=cw-rno2a job run --cpu 2 --memory 8GB \\
        --enable-extra-resources --extra cpu --extra datakit \\
        -- python -m experiments.references.reference_training_pipeline \\
            --version dev --stop-after datakit
"""

import argparse
import logging
from dataclasses import replace
from enum import StrEnum

from fray.cluster import ResourceConfig
from fray.types import ANY_REGION
from levanter.optim.config import AdamConfig
from levanter.tracker.wandb import WandbConfig
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext, run
from marin.experiment.evaluation import eval_report, eval_steps
from marin.experiment.namespacing import user_namespaced_name
from marin.training.training import LevanterCheckpoint
from rigging.log_setup import configure_logging

from experiments.datakit.reference_pipeline import (
    DEFAULT_MAX_CONCURRENT,
    QUALITY_MODEL_VERSION,
    SAMPLE_PREFIX,
    SAMPLE_SOURCES,
    SMOKE_SCALE,
    PoolConfig,
    materialize_reference_store,
    quality_model_path,
    sample_sources,
)
from experiments.datakit.store.datakit_store import ClusteredStoreData
from experiments.datakit.store.mixture import MixtureWeighting, log_store_summary, store_mixture
from experiments.evals.evals import core_evals
from experiments.grug.base.launch import GrugBaseLaunchConfig, run_grug_base_trial
from experiments.grug.base.model import GrugModelConfig

logger = logging.getLogger(__name__)

# Shared name for the train + eval + report handles. The datakit steps keep their own
# ``datakit/*`` content-addressed names.
REF_NAME = "references/reference-pipeline"

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
REFERENCE_EVAL_ACCELERATOR = AcceleratorChoice(platform=Platform.GPU, gpu_type="H100", gpu_count=1)


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
            data=store_mixture(
                store,
                weighting=weighting,
                min_tokens_per_component=model.max_seq_len,
            ),
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="dev", help="artifact version for the train/eval handles (or 'dev')")
    parser.add_argument("--sample-prefix", default=SAMPLE_PREFIX, help="testbed sample root")
    parser.add_argument(
        "--sources",
        default=None,
        help="comma-separated source names, or 'all' to discover every source; default = curated SAMPLE_SOURCES subset",
    )
    parser.add_argument(
        "--quality-model", default=quality_model_path(), help="pooled fast-transformer scorer + calib dir"
    )
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
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=DEFAULT_MAX_CONCURRENT,
        help="max steps a StepRunner walks at once",
    )
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
    store = materialize_reference_store(
        sources,
        quality_model=args.quality_model,
        quality_model_version=args.quality_model_version,
        scale=scale,
        max_concurrent=args.max_concurrent,
    )
    log_store_summary(store)
    if args.stop_after is Stage.DATAKIT:
        logger.info("stop-after=datakit; store at %s", store.cache_path)
        return

    # --- Pass 2: train (+ eval) ---------------------------------------------------------
    model = reference_train_on_store(store, model=REFERENCE_MODEL, version=args.version, weighting=args.weighting)
    if args.stop_after is Stage.TRAIN:
        run(model, max_concurrent=args.max_concurrent)
        logger.info("stop-after=train; checkpoint at %s", model.path())
        return

    results = eval_steps(model, core_evals(accelerator=REFERENCE_EVAL_ACCELERATOR), version=args.version)
    report = eval_report(results, name=REF_NAME, version=args.version)
    run(report, max_concurrent=args.max_concurrent)
    logger.info("eval report at %s", report.path())


if __name__ == "__main__":
    main()
