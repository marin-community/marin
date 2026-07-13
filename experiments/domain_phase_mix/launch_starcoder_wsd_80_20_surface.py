# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the fixed 64-point StarCoder surface with an 80/20 WSD schedule."""

from __future__ import annotations

import argparse
import logging
import math
import os
from dataclasses import dataclass, replace

from fray.types import ResourceConfig
from levanter.models.llama import LlamaConfig
from levanter.optim.muonh import MuonHConfig
from marin.execution.lazy import ArtifactStep, StepContext, lower, run
from marin.experiment.data import mixture
from marin.experiment.train import train_lm
from marin.processing.tokenize.tokenize import TokenizedCache
from marin.training.training import LevanterCheckpoint, TrainLmOnPodConfig

from experiments.datasets.dolma import dolma_datasets
from experiments.datasets.nemotron import nemotron_datasets
from experiments.datasets.paloma import paloma_datasets
from experiments.datasets.uncheatable import uncheatable_datasets
from experiments.llama import llama3_tokenizer

logger = logging.getLogger(__name__)

NAME = "pinlin_calvin_xu/data_mixture/two_phase_starcoder_wsd80_20_surface64_20260711"
WANDB_EXPERIMENT_TAG = "starcoder_wsd80_20_surface64"
OBJECTIVE_METRIC = "eval/paloma/dolma_100_programing_languages/bpb"
EXPERIMENT_BUDGET = 1_000_000_000
TARGET_BUDGET = 5_729_908_864_777
BATCH_SIZE = 128
SEQ_LEN = 2048
PHASE_BOUNDARY = 0.8
MIXTURE_BLOCK_SIZE = 2048
WARMUP_FRACTION = 0.01
ANALYSIS_METRICS = ["eval/loss", OBJECTIVE_METRIC]
VERSION = "2026.07.11"
DEFAULT_TPU_TYPE = "v5p-8"
DEFAULT_TPU_REGION = "us-central1"
DEFAULT_TPU_ZONE = "us-central1-a"
DEFAULT_MARIN_PREFIX = "gs://marin-us-central1"
DEFAULT_MAX_CONCURRENT = 64
DEFAULT_DATA_SEED = 20_260_711

NEMOTRON_TOKEN_COUNTS = {
    "hq_actual": 537_620_495_374,
    "hq_synth": 1_497_529_159_716,
    "medium_high": 489_053_720_257,
    "medium": 1_960_603_657_130,
    "medium_low": 860_999_424_951,
    "low_actual": 384_102_407_349,
}

# Original RegMix 60M proxy geometry. It is stated here rather than imported
# through the removed executor-era experiment framework.
REGMIX_60M_PROXY = LlamaConfig(
    max_seq_len=SEQ_LEN,
    hidden_dim=768,
    intermediate_dim=1536,
    num_layers=10,
    num_heads=8,
    num_kv_heads=8,
    tie_word_embeddings=True,
    gradient_checkpointing=True,
    scan_layers=True,
)

# Ordered exactly as selected_coordinates_64.csv in the reviewed design artifact.
SURFACE_COORDINATES: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (0.0, 0.025),
    (0.0, 0.05),
    (0.0, 0.10),
    (0.0, 0.15),
    (0.0, 0.20),
    (0.0, 0.25),
    (0.0, 0.2814080848021789),
    (0.0, 0.30),
    (0.0, 0.35),
    (0.0, 0.40),
    (0.0, 0.50),
    (0.0, 0.60),
    (0.0, 0.65),
    (0.0, 0.7035202120054471),
    (0.0, 0.75),
    (0.0, 0.80),
    (0.0, 0.85025),
    (0.0, 0.90),
    (0.0, 1.0),
    (0.0275, 0.3126),
    (0.0364194347695976, 0.0364194347695976),
    (0.05, 0.05),
    (0.10, 0.10),
    (0.1407040424010894, 0.1407040424010894),
    (0.15, 0.15),
    (0.17005, 0.17005),
    (0.1758800530013617, 0.0),
    (0.20, 0.20),
    (0.2125625, 0.0),
    (0.228, 0.253),
    (0.2405, 0.2405),
    (0.25, 0.25),
    (0.30, 0.30),
    (0.40, 0.40),
    (0.50, 0.50),
    (0.60, 0.60),
    (0.70, 0.70),
    (0.80, 0.80),
    (0.90, 0.90),
    (1.0, 0.0),
    (1.0, 1.0),
    (0.6650742704116693, 0.0),
    (0.4428605356866863, 1.0),
    (1.0, 0.4301969940239051),
    (0.4377258697981811, 0.0),
    (0.2049245327065826, 0.9084203215117188),
    (0.7691273202087807, 0.3128835565078506),
    (0.3021801240283029, 0.6792695917153253),
    (0.6034807337817608, 0.2294881147889508),
    (0.1452468603730965, 0.517364768878253),
    (0.5610815842347233, 0.8263897321179244),
    (0.897047152691042, 0.1208640528987332),
    (0.7509762931249057, 1.0),
    (1.0, 0.6595645994862139),
    (0.4925267800219105, 0.2350626567801249),
    (1.0, 0.2264066237521311),
    (0.3159208592002993, 0.0),
    (0.7687545544819003, 0.0586430611181535),
    (0.5402527425571795, 0.0),
    (0.3924905441894449, 0.1590707052543056),
    (0.1275858678360516, 1.0),
    (0.5542247188441423, 1.0),
    (0.3091737148518361, 0.5137617308892801),
)


@dataclass(frozen=True)
class SurfaceRunSpec:
    """One fixed coordinate in the WSD surface."""

    rank: int
    phase_0_starcoder: float
    phase_1_starcoder: float
    run_name_override: str | None = None
    data_seed_override: int | None = None

    @property
    def run_id(self) -> int:
        return self.rank - 1

    @property
    def run_name(self) -> str:
        if self.run_name_override is not None:
            return self.run_name_override
        return (
            f"surface64_r{self.rank:02d}_p0_{_weight_slug(self.phase_0_starcoder)}"
            f"_p1_{_weight_slug(self.phase_1_starcoder)}"
        )


def _weight_slug(weight: float) -> str:
    return f"{weight:.4f}".replace(".", "p")


def build_run_specs() -> tuple[SurfaceRunSpec, ...]:
    """Return and validate the immutable 64-coordinate manifest."""
    if len(SURFACE_COORDINATES) != 64:
        raise ValueError(f"Expected 64 coordinates, got {len(SURFACE_COORDINATES)}")
    if len(set(SURFACE_COORDINATES)) != len(SURFACE_COORDINATES):
        raise ValueError("Surface coordinates must be unique")
    for p0, p1 in SURFACE_COORDINATES:
        if not 0.0 <= p0 <= 1.0 or not 0.0 <= p1 <= 1.0:
            raise ValueError(f"Coordinate outside simplex interval: {(p0, p1)}")
    return tuple(
        SurfaceRunSpec(rank=index, phase_0_starcoder=p0, phase_1_starcoder=p1)
        for index, (p0, p1) in enumerate(SURFACE_COORDINATES, start=1)
    )


def _schedule_summary() -> dict[str, int | float]:
    total_steps = EXPERIMENT_BUDGET // (BATCH_SIZE * SEQ_LEN)
    step_alignment = MIXTURE_BLOCK_SIZE // math.gcd(BATCH_SIZE, MIXTURE_BLOCK_SIZE)
    boundary_step = (int(total_steps * PHASE_BOUNDARY) // step_alignment) * step_alignment
    return {
        "total_steps": total_steps,
        "boundary_step": boundary_step,
        "realized_phase_0_fraction": boundary_step / total_steps,
        "warmup_steps": int(total_steps * WARMUP_FRACTION),
        "decay_steps": total_steps - boundary_step,
        "materialized_tokens": total_steps * BATCH_SIZE * SEQ_LEN,
    }


def _optimizer() -> MuonHConfig:
    """Return the historical Muon optimizer with WSD aligned to phase 1."""
    schedule = _schedule_summary()
    return MuonHConfig(
        learning_rate=0.02,
        adam_lr=0.008,
        min_lr_ratio=0.0,
        momentum=0.95,
        beta1=0.9,
        beta2=0.98,
        epsilon=1e-15,
        muon_epsilon=1e-5,
        max_grad_norm=1.0,
        warmup=schedule["warmup_steps"],
        decay=schedule["decay_steps"],
        rewarmup=0.0,
        lr_schedule="cosine",
        cycle_length=None,
    )


def _phase_leaf_weights(
    starcoder_weight: float,
    *,
    nemotron: dict[str, ArtifactStep[TokenizedCache]],
    starcoder: ArtifactStep[TokenizedCache],
) -> dict[str, float]:
    """Expand one broad-vs-code weight into the historical flat leaf mixture."""
    total_nemotron_tokens = sum(NEMOTRON_TOKEN_COUNTS.values())
    broad_weight = 1.0 - starcoder_weight
    weights = {
        nemotron[split].name: broad_weight * token_count / total_nemotron_tokens
        for split, token_count in NEMOTRON_TOKEN_COUNTS.items()
    }
    weights[starcoder.name] = starcoder_weight
    return weights


def _with_varying_mixture(
    base: ArtifactStep[LevanterCheckpoint],
    *,
    train_datasets: dict[ArtifactStep[TokenizedCache], float],
    validation_datasets: tuple[ArtifactStep[TokenizedCache], ...],
    phase_weights: list[tuple[int, dict[str, float]]],
    data_seed: int,
) -> ArtifactStep[LevanterCheckpoint]:
    """Replace a standard training handle's static data with the reviewed schedule."""

    def build_config(ctx: StepContext) -> TrainLmOnPodConfig:
        pod_config = base.build_config(ctx)
        data_config = mixture(ctx, train_datasets, validation=validation_datasets)
        data_config = replace(
            data_config,
            train_weights=phase_weights,
            mixture_block_size=MIXTURE_BLOCK_SIZE,
            experiment_budget=_schedule_summary()["materialized_tokens"],
            target_budget=TARGET_BUDGET,
            simulated_epoch_subset_seed=data_seed,
        )
        trainer = replace(pod_config.train_config.trainer, seed=data_seed)
        train_config = replace(
            pod_config.train_config,
            data=data_config,
            data_seed=data_seed,
            trainer=trainer,
        )
        return replace(pod_config, train_config=train_config)

    return replace(base, build_config=build_config)


def build_training_steps(
    *,
    name_prefix: str,
    tpu_type: str,
    tpu_region: str,
    tpu_zone: str,
    data_seed: int,
    run_specs: tuple[SurfaceRunSpec, ...] | None = None,
    wandb_experiment_tag: str = WANDB_EXPERIMENT_TAG,
    panel_tag: str = "surface64",
) -> tuple[ArtifactStep[LevanterCheckpoint], ...]:
    """Build resumable training handles on shared pinned datasets."""
    nemotron = nemotron_datasets(tokenizer=llama3_tokenizer)
    starcoder = dolma_datasets(tokenizer=llama3_tokenizer)["dolma/starcoder"]
    training_handles = (*tuple(nemotron[split] for split in NEMOTRON_TOKEN_COUNTS), starcoder)
    validation_handles = (
        *paloma_datasets(tokenizer=llama3_tokenizer).values(),
        *uncheatable_datasets(tokenizer=llama3_tokenizer).values(),
    )
    resources = ResourceConfig.with_tpu(tpu_type, regions=(tpu_region,), zone=tpu_zone)
    schedule = _schedule_summary()
    if run_specs is None:
        run_specs = build_run_specs()
    steps: list[ArtifactStep[LevanterCheckpoint]] = []
    for spec in run_specs:
        run_seed = spec.data_seed_override if spec.data_seed_override is not None else data_seed
        phase_0_weights = _phase_leaf_weights(
            spec.phase_0_starcoder,
            nemotron=nemotron,
            starcoder=starcoder,
        )
        phase_1_weights = _phase_leaf_weights(
            spec.phase_1_starcoder,
            nemotron=nemotron,
            starcoder=starcoder,
        )
        static_weights = {handle: phase_0_weights[handle.name] for handle in training_handles}
        base = train_lm(
            name=f"checkpoints/{name_prefix}/{spec.run_name}",
            version=VERSION,
            model=REGMIX_60M_PROXY,
            optimizer=_optimizer(),
            datasets=static_weights,
            validation=validation_handles,
            batch_size=BATCH_SIZE,
            seq_len=SEQ_LEN,
            num_train_steps=schedule["total_steps"],
            z_loss_weight=None,
            evals=None,
            resources=resources,
            steps_per_eval=1_000,
            wandb_project="marin",
            wandb_group=name_prefix,
            run_id=spec.run_name,
            tags=(wandb_experiment_tag, spec.run_name, "starcoder", "wsd80_20", panel_tag),
            env_vars={"HF_ALLOW_CODE_EVAL": "1"},
        )
        steps.append(
            _with_varying_mixture(
                base,
                train_datasets=static_weights,
                validation_datasets=validation_handles,
                phase_weights=[
                    (0, phase_0_weights),
                    (schedule["boundary_step"], phase_1_weights),
                ],
                data_seed=run_seed,
            )
        )
    return tuple(steps)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name-prefix", default=NAME)
    parser.add_argument("--marin-prefix", default=DEFAULT_MARIN_PREFIX)
    parser.add_argument("--tpu-type", default=DEFAULT_TPU_TYPE)
    parser.add_argument("--tpu-region", default=DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=DEFAULT_TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--data-seed", type=int, default=DEFAULT_DATA_SEED)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if os.getenv("CI") is not None:
        logger.info("Skipping StarCoder WSD surface in CI")
        return
    if args.marin_prefix != DEFAULT_MARIN_PREFIX:
        raise ValueError(f"This historical StarCoder experiment is central1-local: got {args.marin_prefix!r}")
    if args.tpu_region != DEFAULT_TPU_REGION or args.tpu_zone != DEFAULT_TPU_ZONE:
        raise ValueError(
            "StarCoder child TPU placement must remain central1-local: "
            f"got region={args.tpu_region!r}, zone={args.tpu_zone!r}"
        )
    if args.max_concurrent < 1 or args.max_concurrent > len(SURFACE_COORDINATES):
        raise ValueError(f"max_concurrent must be in [1, 64], got {args.max_concurrent}")

    schedule = _schedule_summary()
    logger.info(
        "Prepared %d original-architecture runs: total_steps=%d boundary_step=%d "
        "realized_phase_0_fraction=%.6f warmup=%d decay=%d shared_data_seed=%d",
        len(SURFACE_COORDINATES),
        schedule["total_steps"],
        schedule["boundary_step"],
        schedule["realized_phase_0_fraction"],
        schedule["warmup_steps"],
        schedule["decay_steps"],
        args.data_seed,
    )
    os.environ["MARIN_PREFIX"] = args.marin_prefix
    steps = build_training_steps(
        name_prefix=args.name_prefix,
        tpu_type=args.tpu_type,
        tpu_region=args.tpu_region,
        tpu_zone=args.tpu_zone,
        data_seed=args.data_seed,
    )
    if args.dry_run:
        for step in steps:
            lower(step)
        logger.info("Dry-run lowering passed for all %d training handles", len(steps))
        return
    run(*steps, max_concurrent=args.max_concurrent)


if __name__ == "__main__":
    main()
