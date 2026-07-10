# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scaling ladder for the Completed AdamH heuristic on Nemotron.

End-to-end demonstration of the IsoFLOP scaling suite workflow:

  1. **Analysis** — read metrics from 78 existing isoflop sweep checkpoints,
     fit quadratic loss curves at each budget, extract optimal-token minima,
     fit D* ~ A * C^alpha, save ``isoflop_analysis_result.json``.
  2. **Optimal training** — for each target budget (1e21, 1e22, 1e23 FLOPs) use
     the fitted D*(C) to predict model size, then train with the
     :data:`~experiments.references.completed_adamh.completed_adamh_heuristic`
     to set LR, beta2, etc.  Multiple seeds at smaller budgets.

The isoflop input runs live at ``gs://marin-us-central2/checkpoints/isoflop/``
(created by the old executor from :data:`_ADAMH_V6_ISOFLOP_RUNS`).
Metrics are read via ``tracker_metrics.jsonl`` with WandB fallback.

Run (analysis + optimal chain):
    uv run iris --cluster=marin job run --no-wait --cpu=1 --memory=2G --extra=cpu \\
      -- python -m experiments.references.reference_scaling_suite
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import timedelta

import fsspec
import jmp
from fray.cluster import ResourceConfig
from haliax.partitioning import ResourceAxis
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text.datasets import LmDataConfig
from levanter.main import train_lm
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import remote
from marin.execution.step_runner import StepRunner
from marin.experiment.data import mixture
from marin.experiment.namespacing import user_namespaced_name
from marin.processing.tokenize.tokenize import TokenizedCache
from marin.scaling_laws import (
    IsoFlopRecord,
    ScalingFit,
    fit_scaling_laws,
    predict_optimal_config,
    round_flops_to_bucket,
)
from marin.scaling_laws.eval_metrics_reader import read_eval_records
from marin.training.training import LevanterCheckpoint, TrainLmOnPodConfig, run_levanter_train_lm
from marin.utilities.wandb_utils import WANDB_ENTITY, WANDB_PROJECT
from pydantic import Field
from rigging.filesystem import prefix_join

from experiments.datasets.nemotron import nemotron_datasets
from experiments.datasets.paloma import paloma_datasets
from experiments.datasets.proofpile import proofpile_dataset
from experiments.datasets.starcoder import starcoder_dataset
from experiments.datasets.uncheatable import uncheatable_datasets
from experiments.llama import llama3_tokenizer
from experiments.references.completed_adamh import SEQ_LEN, completed_adamh_heuristic

logger = logging.getLogger(__name__)

# --- Existing isoflop sweep checkpoints (Delphi AdamH v6 on Nemotron) ---
# These 78 runs were produced by the old executor; metrics are read via WandB.
_ADAMH_V6_ISOFLOP_RUNS: tuple[str, ...] = (
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-3e+18-d512-L6-B32-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-3e+18-d640-L7-B16-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-3e+18-d768-L8-B16-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-3e+18-d896-L10-B8-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-3e+18-d1024-L11-B8-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-9e+18-d512-L6-B64-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-9e+18-d640-L7-B64-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-9e+18-d768-L8-B32-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-9e+18-d896-L10-B32-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-9e+18-d1024-L11-B16-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-9e+18-d1152-L12-B16-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-9e+18-d1280-L13-B16-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-9e+18-d1408-L15-B8-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-9e+18-d1536-L16-B8-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-9e+18-d1664-L17-B8-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-2e+19-d512-L6-B128-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-2e+19-d640-L7-B128-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-2e+19-d768-L8-B64-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-2e+19-d896-L10-B64-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-2e+19-d1024-L11-B32-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-2e+19-d1152-L12-B32-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-2e+19-d1280-L13-B32-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-2e+19-d1408-L15-B16-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-2e+19-d1536-L16-B16-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-2e+19-d1664-L17-B16-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-2e+19-d1792-L18-B8-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-2e+19-d1920-L19-B8-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-2e+19-d2048-L21-B8-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-2e+19-d2176-L22-B8-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-2e+19-d2304-L23-B8-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-3e+19-d768-L8-B128-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-3e+19-d1024-L11-B64-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-3e+19-d1280-L13-B32-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-3e+19-d1536-L16-B32-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-3e+19-d1792-L18-B16-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-3e+19-d2048-L21-B16-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-3e+19-d2304-L23-B8-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-3e+19-d2560-L26-B8-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-3e+19-d2816-L28-B8-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-9e+19-d768-L8-B256-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-9e+19-d1024-L11-B256-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-9e+19-d1280-L13-B128-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-9e+19-d1536-L16-B64-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-9e+19-d1792-L18-B64-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-9e+19-d2048-L21-B32-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-9e+19-d2304-L23-B32-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-9e+19-d2560-L26-B16-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-9e+19-d2816-L28-B16-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-9e+19-d3072-L30-B16-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-9e+19-d3328-L33-B8-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-9e+19-d3584-L35-B8-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-9e+19-d3840-L37-B8-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-9e+19-d4096-L40-B8-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-2e+20-d1024-L11-B512-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-2e+20-d1280-L13-B256-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-2e+20-d1536-L16-B128-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-2e+20-d1792-L18-B128-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-2e+20-d2048-L21-B64-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-2e+20-d2304-L23-B64-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-2e+20-d2560-L26-B32-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-2e+20-d2816-L28-B32-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-2e+20-d3072-L30-B32-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-2e+20-d3328-L33-B16-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-2e+20-d3584-L35-B16-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-2e+20-d3840-L37-B16-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-2e+20-d4096-L40-B16-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-3e+20-d1280-L13-B512-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-3e+20-d1536-L16-B256-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-3e+20-d1792-L18-B256-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-3e+20-d2048-L21-B128-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-3e+20-d2304-L23-B128-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-3e+20-d2560-L26-B64-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-3e+20-d2816-L28-B64-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-3e+20-d3072-L30-B64-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-3e+20-d3328-L33-B32-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-3e+20-d3584-L35-B32-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-3e+20-d3840-L37-B32-adamh_scaling_v6",
    "gs://marin-us-central2/checkpoints/isoflop/isoflop-3e+20-d4096-L40-B16-adamh_scaling_v6",
)

# --- Metric extraction keys ---
_THROUGHPUT_TOKENS_KEY = "throughput/total_tokens"
_THROUGHPUT_GFLOPS_KEY = "throughput/total_gflops"
_PARAMETER_COUNT_KEY = "parameter_count"
_DEFAULT_METRIC_KEY = "eval/paloma/c4_en/bpb"

# --- Scaling suite constants ---
EXPERIMENT_NAME = "adamh-scaling-ladder-nemotron"
LABEL = "adamh_scaling_v6"

TARGET_BUDGETS: dict[float, tuple[str, int]] = {
    1e21: ("v4-128", 512),
    1e22: ("v4-512", 1024),
    1e23: ("v4-1024", 2048),
}
SEEDS_PER_BUDGET: dict[float, list[int]] = {
    1e21: [0, 42, 62746],
    1e22: [0, 42, 62746],
    1e23: [0],
}

# --- Training data handles ---
_NEMOTRON_WEIGHTS = {
    "hq_actual": 0.91351,
    "hq_synth": 2.72,
    "medium_high": 0.82471,
    "medium": 3.38,
    "medium_low": 1.54,
    "low_actual": 0.70123,
    "low_synth": 0.62771,
}
_STARCODER_WEIGHT = 0.25
_PROOFPILE_WEIGHT = 0.055

_nem = nemotron_datasets(tokenizer=llama3_tokenizer)
_NEMOTRON_TRAIN: dict[ArtifactStep[TokenizedCache], float] = {
    _nem[split]: weight for split, weight in _NEMOTRON_WEIGHTS.items()
}
_NEMOTRON_TRAIN[starcoder_dataset(tokenizer=llama3_tokenizer)] = _STARCODER_WEIGHT
_NEMOTRON_TRAIN[proofpile_dataset(tokenizer=llama3_tokenizer)] = _PROOFPILE_WEIGHT
_VALIDATION: list[ArtifactStep[TokenizedCache]] = [
    *paloma_datasets().values(),
    *uncheatable_datasets().values(),
]
_ALL_DATA_DEPS: tuple[ArtifactStep[TokenizedCache], ...] = (*_NEMOTRON_TRAIN, *_VALIDATION)


# --- IsoFLOP analysis helpers (inlined from the deleted experiments/isoflop_sweep.py) ---


def _parse_isoflop_run_name(run_name: str) -> str | None:
    run_name = re.sub(r"-[0-9a-fA-F]{6}$", "", run_name)
    for pattern in [
        r"isoflop-(?:[0-9.e+]+)-N(?:[0-9.e+]+)-B(?:\d+)-(.+)",
        r"isoflop-(?:[0-9.e+]+)-d(?:\d+)-L(?:\d+)-B(?:\d+)-(.+)",
    ]:
        match = re.match(pattern, run_name)
        if match:
            return match.group(1)
    return None


def _transform_levanter_metrics(
    raw_records: list[dict],
    metric_key: str,
    label_map: dict[str, str] | None = None,
    min_flops: float = 1e18,
) -> list[IsoFlopRecord]:
    records = []
    for raw in raw_records:
        run_path = raw.get("run_path", "")
        run_name = os.path.basename(run_path.rstrip("/"))
        summary = raw.get("summary", {}) or {}

        tokens = summary.get(_THROUGHPUT_TOKENS_KEY)
        total_gflops = summary.get(_THROUGHPUT_GFLOPS_KEY)
        metric = summary.get(metric_key)
        params = summary.get(_PARAMETER_COUNT_KEY)

        if any(v is None for v in [tokens, total_gflops, metric, params]):
            continue

        flops = round_flops_to_bucket(total_gflops * 1e9)
        if flops < min_flops:
            continue

        exp_name = _parse_isoflop_run_name(run_name) or run_name
        label = (label_map or {}).get(exp_name, exp_name)

        records.append(
            IsoFlopRecord(
                tokens=float(tokens),
                metric=float(metric),
                flops=float(flops),
                params=float(params),
                label=label,
            )
        )

    logger.info(f"Transformed {len(records)} records from {len(raw_records)} raw records")
    return records


class IsoFlopAnalysisArtifact(Artifact):
    """Typed artifact for IsoFLOP analysis results.

    ``scaling_fits`` is persisted in ``record.result`` so downstream steps resolve
    it via ``ctx.resolved()`` without reading any sidecar files.  The quadratic
    fit coefficients are still written to ``fit_curves.json`` for human inspection.
    """

    scaling_fits: dict[str, tuple[float, float]] = Field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class IsoFlopAnalysisConfig:
    training_runs: tuple[str, ...]
    output_path: str
    metric_key: str = _DEFAULT_METRIC_KEY
    label_map: tuple[tuple[str, str], ...] | None = None
    metrics_filename: str = "tracker_metrics.jsonl"
    wandb_entity_project: str = f"{WANDB_ENTITY}/{WANDB_PROJECT}"


def run_isoflop_analysis(config: IsoFlopAnalysisConfig) -> IsoFlopAnalysisArtifact:
    raw_records = read_eval_records(
        training_runs=list(config.training_runs),
        metrics_filename=config.metrics_filename,
        wandb_entity_project=config.wandb_entity_project,
    )
    label_map = dict(config.label_map) if config.label_map else None
    records = _transform_levanter_metrics(raw_records, config.metric_key, label_map)

    if not records:
        raise RuntimeError("No valid isoflop records found after reading and transforming metrics")

    result = fit_scaling_laws(records)
    logger.info(f"Found {len(result.minima_records)} optimal configurations")
    for label, scaling_fit in result.scaling_fits.items():
        logger.info(f"  {label}: D* = {scaling_fit.A:.2e} * C^{scaling_fit.alpha:.3f}")

    # Write fit_curves.json as a human-readable sidecar (not consumed by downstream steps).
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    fit_curves_json = {f"{label}|{flops}": list(coeffs) for (label, flops), coeffs in result.fit_curves.items()}
    with fs.open(prefix_join(config.output_path, "fit_curves.json"), "w") as f:
        json.dump(fit_curves_json, f, indent=2)

    return IsoFlopAnalysisArtifact(scaling_fits={k: (v.alpha, v.A) for k, v in result.scaling_fits.items()})


# --- Optimal training ---


@dataclass(frozen=True)
class OptimalTrainingConfig:
    scaling_fits: dict[str, tuple[float, float]]
    target_budget: float
    resources: ResourceConfig
    batch_size: int
    label: str
    output_path: str
    data: LmDataConfig
    seed: int = 0


def run_optimal_training(config: OptimalTrainingConfig) -> None:
    """Predict compute-optimal config from scaling fits and dispatch TPU training."""
    scaling_fits = {k: ScalingFit(*v) for k, v in config.scaling_fits.items()}

    candidate = predict_optimal_config(
        scaling_fits=scaling_fits,
        target_flops=config.target_budget,
        label=config.label,
        heuristic=completed_adamh_heuristic,
        seq_len=SEQ_LEN,
    )
    if candidate is None:
        raise RuntimeError(f"Could not find optimal config for budget {config.target_budget:.2e} label '{config.label}'")

    model_config = candidate.model_config
    params = model_config.total_trainable_params(completed_adamh_heuristic.vocab_size)
    hidden_dim = model_config.hidden_dim
    chips = config.resources.chip_count()

    tp = 1
    while hidden_dim % (chips // tp) != 0 and tp < 8:
        tp *= 2

    optimizer_config = completed_adamh_heuristic.build_optimizer_config(config.batch_size, candidate.tokens)
    train_steps = round(candidate.tokens / (config.batch_size * SEQ_LEN))

    logger.info(
        "Optimal config: budget=%.2e hidden=%d layers=%d params=%.2e tokens=%.2e batch=%d steps=%d tp=%d",
        config.target_budget,
        hidden_dim,
        model_config.num_layers,
        params,
        candidate.tokens,
        config.batch_size,
        train_steps,
        tp,
    )

    inner_config = train_lm.TrainLmConfig(
        data=config.data,
        trainer=TrainerConfig(
            tracker=WandbConfig(
                entity=WANDB_ENTITY,
                project=WANDB_PROJECT,
                tags=[
                    "optimal-training",
                    "completed-adamh",
                    f"FLOPs={config.target_budget:.1e}",
                    f"label={config.label}",
                    f"N={params:.1e}",
                    f"seed={config.seed}",
                ],
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=config.batch_size,
            per_device_parallelism=-1,
            num_train_steps=train_steps,
            steps_per_eval=1000,
            checkpointer=CheckpointerConfig(
                save_interval=timedelta(minutes=10),
                keep=[dict(every=5000)],
            ),
            mesh=MeshConfig(
                axes={"data": -1, "replica": 1, "model": tp},
                compute_mapping={
                    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                },
            ),
            seed=config.seed,
            allow_nondivisible_batch_size=True,
        ),
        train_seq_len=SEQ_LEN,
        model=model_config,
        optimizer=optimizer_config,
    )

    pod_config = TrainLmOnPodConfig(
        train_config=inner_config,
        resources=config.resources,
        output_path=config.output_path,
    )
    remote(run_levanter_train_lm, resources=config.resources)(pod_config)


# --- ArtifactStep builders ---


def _analysis_step(*, version: str) -> ArtifactStep[IsoFlopAnalysisArtifact]:
    def build_config(ctx: StepContext) -> IsoFlopAnalysisConfig:
        return IsoFlopAnalysisConfig(
            training_runs=_ADAMH_V6_ISOFLOP_RUNS,
            output_path=ctx.output_path,
        )

    return ArtifactStep(
        name=user_namespaced_name(f"{EXPERIMENT_NAME}-analysis", version),
        version=version,
        artifact_type=IsoFlopAnalysisArtifact,
        run=remote(run_isoflop_analysis, resources=ResourceConfig.with_cpu()),
        build_config=build_config,
        deps=(),
    )


def _optimal_step(
    *,
    analysis: ArtifactStep[IsoFlopAnalysisArtifact],
    budget: float,
    tpu_type: str,
    batch_size: int,
    seed: int,
    version: str,
) -> ArtifactStep[LevanterCheckpoint]:
    resources = ResourceConfig.with_tpu(tpu_type)
    suffix = f"-seed{seed}" if seed != 0 else ""
    name = user_namespaced_name(f"{EXPERIMENT_NAME}-optimal-{budget:.0e}{suffix}", version)

    def build_config(ctx: StepContext) -> OptimalTrainingConfig:
        artifact = ctx.resolved(analysis)
        return OptimalTrainingConfig(
            scaling_fits=artifact.scaling_fits,
            target_budget=budget,
            resources=resources,
            batch_size=batch_size,
            label=LABEL,
            output_path=ctx.output_path,
            data=mixture(ctx, _NEMOTRON_TRAIN, validation=_VALIDATION),
            seed=seed,
        )

    return ArtifactStep(
        name=name,
        version=version,
        artifact_type=LevanterCheckpoint,
        run=remote(run_optimal_training, resources=ResourceConfig.with_cpu()),
        build_config=build_config,
        deps=(analysis, *_ALL_DATA_DEPS),
    )


def build(*, version: str = "dev") -> list[ArtifactStep[IsoFlopAnalysisArtifact] | ArtifactStep[LevanterCheckpoint]]:
    """Return all steps: [analysis, optimal-1e21-seed0, ..., optimal-1e23]."""
    analysis = _analysis_step(version=version)
    optimal_runs: list[ArtifactStep[LevanterCheckpoint]] = []
    for budget, (tpu_type, batch_size) in TARGET_BUDGETS.items():
        for seed in SEEDS_PER_BUDGET[budget]:
            optimal_runs.append(
                _optimal_step(
                    analysis=analysis,
                    budget=budget,
                    tpu_type=tpu_type,
                    batch_size=batch_size,
                    seed=seed,
                    version=version,
                )
            )
    return [analysis, *optimal_runs]


if __name__ == "__main__":
    StepRunner().run([s.lower() for s in build()])
