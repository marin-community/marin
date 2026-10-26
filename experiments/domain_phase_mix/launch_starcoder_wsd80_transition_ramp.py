# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Launch the StarCoder WSD80 transition-ramp intervention on the 4x subset.

The confirmed 4x two-phase winner (``dsopt_r3d28260_m400_untied_c007``) switches StarCoder from
0.00 to 0.28 at step 22,608, the decay onset. Three linear ramps of 512, 2,048, and 5,632 steps
centred on that step replace the switch as staircases of one-block (16-step) stages. Each ramp's
realized StarCoder sequence total is matched exactly to the abrupt schedule of the same data
seed through the real ``MixtureDataset`` counts, so StarCoder epochs, sequence order, and the
learning-rate schedule are unchanged. The five confirmation seeds pair each ramp with the
August abrupt run; one abrupt replicate on the first seed is a drift canary for the training
stack, which moved from JAX 0.10.1 to 0.11.1 since August. The ``quick`` design variant adds
128- and 256-step ramps on the same seeds, shorter than the transition's own relaxation time, and
the ``abrupt`` variant retrains the abrupt switch on the four remaining seeds so that every ramp
pairs with an abrupt run from the same software.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import jax
import numpy as np
from fray.types import ResourceConfig
from levanter.data.mixture import MixtureDataset
from levanter.main.train_lm import TrainLmConfig
from marin.execution.lazy import ArtifactStep, lower, materialized_config, run
from marin.experiment.train import train_lm
from marin.processing.tokenize.tokenize import TokenizedCache
from marin.training.training import LevanterCheckpoint, TrainLmOnPodConfig

from experiments.datasets.dolma import dolma_datasets
from experiments.datasets.nemotron import nemotron_datasets
from experiments.datasets.paloma import paloma_datasets
from experiments.datasets.uncheatable import uncheatable_datasets
from experiments.domain_phase_mix import (
    launch_starcoder_wsd80_dense_support_empirical_optimum_confirmation as confirmation,
)
from experiments.domain_phase_mix import launch_starcoder_wsd80_dense_support_surfaces as source
from experiments.domain_phase_mix import launch_starcoder_wsd_80_20_surface as base
from experiments.llama import llama3_tokenizer

logger = logging.getLogger(__name__)

NAME = "pinlin_calvin_xu/data_mixture/starcoder_wsd80_transition_ramp_20261006"
WANDB_EXPERIMENT_TAG = "starcoder_wsd80_transition_ramp_20261006"
PANEL_TAG = "wsd80_transition_ramp"


@dataclass(frozen=True)
class Variant:
    """One frozen design file: a set of ramp widths on the confirmation seeds."""

    name: str
    design_version: str
    design_path: Path
    expected_sha256: str
    ramp_widths: tuple[int, ...]
    abrupt_seeds: tuple[int, ...]

    @property
    def expected_run_count(self) -> int:
        return len(SEEDS) * len(self.ramp_widths) + len(self.abrupt_seeds)


VARIANTS = {
    "main": Variant(
        name="main",
        design_version="2026-10-06-v1",
        design_path=Path(__file__).with_name("starcoder_wsd80_transition_ramp_design_20261006.json"),
        expected_sha256="784220261871d5d790febd58c0f868fe7975a73f9203d7d010dab6ef75c3fb3c",
        ramp_widths=(512, 2048, 5632),
        abrupt_seeds=(20260821,),
    ),
    "quick": Variant(
        name="quick",
        design_version="2026-10-07-quick-v1",
        design_path=Path(__file__).with_name("starcoder_wsd80_transition_ramp_quick_design_20261007.json"),
        expected_sha256="b65267c15ba243c9bc0dfcb46cef61cff8b357236b78e5b798427e281e793daa",
        ramp_widths=(128, 256),
        abrupt_seeds=(),
    ),
    "abrupt": Variant(
        name="abrupt",
        design_version="2026-10-07-abrupt-v1",
        design_path=Path(__file__).with_name("starcoder_wsd80_transition_ramp_abrupt_design_20261007.json"),
        expected_sha256="97a6ffc7908989510bb126c62f49ae2ec5a3f3e3f9b19714c0dac25b9f6c765d",
        ramp_widths=(),
        abrupt_seeds=(20260822, 20260823, 20260824, 20260825),
    ),
}
SUPPORT_ID = "m400"
SOURCE_RUN_TEMPLATE = "dsopt_r3d28260_m400_untied_c007_s{seed_suffix}"
TIED_RUN_TEMPLATE = "dsopt_r3d28260_m400_tied_c043_s{seed_suffix}"
RUN_PREFIX = "tramp_r3d28260_m400"
SEEDS = (20260821, 20260822, 20260823, 20260824, 20260825)
STEPS_PER_BLOCK = base.MIXTURE_BLOCK_SIZE // base.BATCH_SIZE
MAX_RAMP_NUDGE_ITERATIONS = 20_000
WEIGHT_TOLERANCE = 1e-12
DEFAULT_MAX_CONCURRENT = 16
MAX_WANDB_TAG_LENGTH = 64


@dataclass(frozen=True)
class RampRun:
    """One arm of the intervention: a stage schedule on one confirmation seed."""

    run_name: str
    arm: str
    ramp_width: int | None
    data_seed: int
    source_run_name: str
    tied_run_name: str
    stages: list[tuple[int, float]]
    realized_starcoder_sequences: int
    target_starcoder_sequences: int
    pre_boundary_starcoder_sequences: int
    max_weight_nudge: float


def _canonical_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _component_names() -> list[str]:
    names = [f"nemotron_{index}" for index in range(len(base.NEMOTRON_TOKEN_COUNTS))]
    names.append("starcoder")
    return names


def _leaf_weights(starcoder_weight: float) -> dict[str, float]:
    total_nemotron_tokens = sum(base.NEMOTRON_TOKEN_COUNTS.values())
    broad = [(1.0 - starcoder_weight) * count / total_nemotron_tokens for count in base.NEMOTRON_TOKEN_COUNTS.values()]
    return dict(zip(_component_names(), (*broad, starcoder_weight), strict=True))


def realized_starcoder_sequences(
    stages: list[tuple[int, float]], *, total_steps: int, data_seed: int
) -> tuple[int, list[int]]:
    """Count the StarCoder sequences the loader draws per stage, including the final partial block."""
    names = _component_names()
    weights = [(step * base.BATCH_SIZE, _leaf_weights(weight)) for step, weight in stages]
    mix_key, _ = jax.random.split(jax.random.PRNGKey(data_seed))
    mixture = MixtureDataset(
        {name: object() for name in names}, weights, block_size=base.MIXTURE_BLOCK_SIZE, key=mix_key
    )
    starcoder_index = list(mixture.dataset_index).index("starcoder")
    starts = [step for step, _ in stages] + [total_steps]
    per_stage = []
    for stage, (start, stop) in enumerate(itertools.pairwise(starts)):
        first_block, first_offset = divmod(start * base.BATCH_SIZE, base.MIXTURE_BLOCK_SIZE)
        if first_offset != 0:
            raise ValueError(f"Stage {stage} starts off a mixture-block boundary at step {start}")
        full_blocks, remainder = divmod((stop - start) * base.BATCH_SIZE, base.MIXTURE_BLOCK_SIZE)
        count = full_blocks * int(mixture._counts_per_block_per_stage[stage][starcoder_index])
        if remainder:
            block = mixture._get_block(first_block + full_blocks)
            count += int(np.count_nonzero((block[:remainder] >> 16) == starcoder_index))
        per_stage.append(count)
    return sum(per_stage), per_stage


def linear_ramp_stages(p0: float, p1: float, *, width: int, boundary_step: int) -> list[tuple[int, float]]:
    """A linear ramp from p0 to p1 centred on the boundary, sampled at the midpoint of each block."""
    half = width // 2
    start, end = boundary_step - half, boundary_step + half
    if width % (2 * STEPS_PER_BLOCK) != 0 or start % STEPS_PER_BLOCK != 0:
        raise ValueError(f"Ramp width {width} is not centred on a block boundary")
    stages = [(0, p0)]
    for step in range(start, end, STEPS_PER_BLOCK):
        midpoint = step + STEPS_PER_BLOCK / 2
        stages.append((step, p0 + (p1 - p0) * (midpoint - start) / (end - start)))
    stages.append((end, p1))
    return stages


def matched_ramp_stages(
    p0: float,
    p1: float,
    *,
    width: int,
    boundary_step: int,
    total_steps: int,
    data_seed: int,
    target_total: int,
) -> tuple[list[tuple[int, float]], int, list[int], float]:
    """Nudge ramp-stage weights by one sequence per block until the realized StarCoder total matches."""
    stages = linear_ramp_stages(p0, p1, width=width, boundary_step=boundary_step)
    linear = [weight for _, weight in stages]
    total, per_stage = realized_starcoder_sequences(stages, total_steps=total_steps, data_seed=data_seed)
    ramp_stage_count = len(stages) - 2
    iteration = 0
    while total != target_total:
        index = 1 + (iteration % ramp_stage_count)
        step, weight = stages[index]
        stages[index] = (step, weight + math.copysign(1.0 / base.MIXTURE_BLOCK_SIZE, target_total - total))
        total, per_stage = realized_starcoder_sequences(stages, total_steps=total_steps, data_seed=data_seed)
        iteration += 1
        if iteration > MAX_RAMP_NUDGE_ITERATIONS:
            raise RuntimeError(f"Ramp of width {width} did not reach {target_total} StarCoder sequences")
    nudge = max(abs(weight - reference) for (_, weight), reference in zip(stages, linear, strict=True))
    if any(weight < 0.0 or weight > 1.0 for _, weight in stages):
        raise ValueError(f"Ramp of width {width} left the unit interval")
    return stages, total, per_stage, nudge


def _source_requests() -> dict[int, source.SurfaceRun]:
    names = frozenset(SOURCE_RUN_TEMPLATE.format(seed_suffix=str(seed)[4:]) for seed in SEEDS)
    _, requests = confirmation.load_design(names)
    by_seed = {request.data_seed: request for request in requests}
    if set(by_seed) != set(SEEDS):
        raise ValueError(f"Confirmation rows cover seeds {sorted(by_seed)}, expected {SEEDS}")
    return by_seed


def _run_name(arm: str, seed: int) -> str:
    return f"{RUN_PREFIX}_{arm}_s{str(seed)[4:]}"


def build_design(variant: Variant) -> dict[str, Any]:
    """Build the frozen stage schedules of one variant from the confirmation rows."""
    requests = _source_requests()
    first = requests[SEEDS[0]]
    runs: list[dict[str, Any]] = []
    for seed in SEEDS:
        request = requests[seed]
        p0, p1 = request.phase_0_starcoder, request.phase_1_starcoder
        abrupt = [(0, p0), (request.boundary_step, p1)]
        target, abrupt_per_stage = realized_starcoder_sequences(abrupt, total_steps=request.total_steps, data_seed=seed)
        if abrupt_per_stage != [request.starcoder_phase_0_sequences, request.starcoder_phase_1_sequences]:
            raise ValueError(f"{request.run_name}: abrupt counts {abrupt_per_stage} differ from the frozen confirmation")
        arms: list[tuple[str, int | None, list[tuple[int, float]], int, list[int], float]] = []
        if seed in variant.abrupt_seeds:
            arms.append(("abrupt", None, abrupt, target, abrupt_per_stage, 0.0))
        for width in variant.ramp_widths:
            stages, total, per_stage, nudge = matched_ramp_stages(
                p0,
                p1,
                width=width,
                boundary_step=request.boundary_step,
                total_steps=request.total_steps,
                data_seed=seed,
                target_total=target,
            )
            arms.append((f"ramp{width:04d}", width, stages, total, per_stage, nudge))
        for arm, width, stages, total, per_stage, nudge in arms:
            pre_boundary = sum(c for (step, _), c in zip(stages, per_stage, strict=True) if step < request.boundary_step)
            runs.append(
                {
                    "run_name": _run_name(arm, seed),
                    "arm": arm,
                    "ramp_width": width,
                    "data_seed": seed,
                    "source_run_name": request.run_name,
                    "tied_run_name": TIED_RUN_TEMPLATE.format(seed_suffix=str(seed)[4:]),
                    "stages": [[step, weight] for step, weight in stages],
                    "realized_starcoder_sequences": total,
                    "target_starcoder_sequences": target,
                    "pre_boundary_starcoder_sequences": pre_boundary,
                    "max_weight_nudge": nudge,
                }
            )
    payload = {
        "design_version": variant.design_version,
        "variant": variant.name,
        "source_design_sha256": confirmation.EXPECTED_DESIGN_SHA256,
        "support_id": SUPPORT_ID,
        "cell_id": first.cell_id,
        "total_steps": first.total_steps,
        "boundary_step": first.boundary_step,
        "phase_0_starcoder": first.phase_0_starcoder,
        "phase_1_starcoder": first.phase_1_starcoder,
        "starcoder_support_batches": first.starcoder_support_batches,
        "ramp_widths": list(variant.ramp_widths),
        "seeds": list(SEEDS),
        "abrupt_seeds": list(variant.abrupt_seeds),
        "steps_per_block": STEPS_PER_BLOCK,
        "design_environment": source._runtime_scientific_environment(),
        "august_training_environment": source.EXPECTED_TRAINING_ENVIRONMENT,
        "runs": runs,
    }
    payload["design_sha256"] = _canonical_sha256(payload)
    return payload


def _load_payload(variant: Variant) -> dict[str, Any]:
    with variant.design_path.open() as handle:
        payload = json.load(handle)
    if payload.get("design_version") != variant.design_version:
        raise ValueError(f"Unexpected design version in {variant.design_path}")
    claimed = payload.pop("design_sha256", None)
    observed = _canonical_sha256(payload)
    if claimed != variant.expected_sha256 or observed != variant.expected_sha256:
        raise ValueError(f"Design self-hash mismatch: {observed} != {claimed}")
    payload["design_sha256"] = claimed
    return payload


def load_design(
    variant: Variant, selected_runs: frozenset[str] | None = None
) -> tuple[dict[str, Any], tuple[RampRun, ...]]:
    """Load and structurally audit one frozen ramp manifest."""
    payload = _load_payload(variant)
    if payload["source_design_sha256"] != confirmation.EXPECTED_DESIGN_SHA256:
        raise ValueError("Ramp design was built from a different confirmation design")
    runs = tuple(
        RampRun(**{**row, "stages": [(int(step), float(weight)) for step, weight in row["stages"]]})
        for row in payload["runs"]
    )
    expected = variant.expected_run_count
    if len(runs) != expected or len({run.run_name for run in runs}) != expected:
        raise ValueError(f"Expected {expected} unique ramp runs, got {len(runs)}")
    if {run.ramp_width for run in runs if run.arm != "abrupt"} != set(variant.ramp_widths):
        raise ValueError(f"{variant.name}: ramp widths drifted from {variant.ramp_widths}")
    boundary = int(payload["boundary_step"])
    for run_ in runs:
        starts = [step for step, _ in run_.stages]
        if starts[0] != 0 or starts != sorted(set(starts)) or any(step % STEPS_PER_BLOCK for step in starts):
            raise ValueError(f"{run_.run_name}: stage starts are not increasing block boundaries")
        if run_.realized_starcoder_sequences != run_.target_starcoder_sequences:
            raise ValueError(f"{run_.run_name}: realized StarCoder total differs from the abrupt schedule")
        weights = [weight for _, weight in run_.stages]
        if any(later < earlier - 2.0 / base.MIXTURE_BLOCK_SIZE for earlier, later in itertools.pairwise(weights)):
            raise ValueError(f"{run_.run_name}: ramp weights are not monotone")
        if run_.arm == "abrupt":
            if run_.stages != [(0, payload["phase_0_starcoder"]), (boundary, payload["phase_1_starcoder"])]:
                raise ValueError(f"{run_.run_name}: canary schedule differs from the confirmed winner")
        elif run_.ramp_width is None or len(run_.stages) != run_.ramp_width // STEPS_PER_BLOCK + 2:
            raise ValueError(f"{run_.run_name}: ramp stage count does not match its width")
        elif (
            run_.stages[1][0] != boundary - run_.ramp_width // 2 or run_.stages[-1][0] != boundary + run_.ramp_width // 2
        ):
            raise ValueError(f"{run_.run_name}: ramp is not centred on the boundary")
    if selected_runs is not None:
        unknown = selected_runs - {run_.run_name for run_ in runs}
        if unknown:
            raise ValueError(f"Unknown ramp runs: {sorted(unknown)}")
        runs = tuple(run_ for run_ in runs if run_.run_name in selected_runs)
    if not runs:
        raise ValueError("Launch filters selected no ramp runs")
    return payload, runs


def audit_runtime_sequence_counts(payload: dict[str, Any], runs: tuple[RampRun, ...]) -> int:
    """Recompute every realized StarCoder count in the runtime JAX environment."""
    for run_ in runs:
        total, _ = realized_starcoder_sequences(
            run_.stages, total_steps=int(payload["total_steps"]), data_seed=run_.data_seed
        )
        if total != run_.realized_starcoder_sequences:
            raise ValueError(
                f"{run_.run_name}: runtime StarCoder count {total} != frozen {run_.realized_starcoder_sequences}"
            )
    logger.info("Reproduced all %d ramp sequence totals", len(runs))
    return len(runs)


def _wandb_tags(run_: RampRun) -> tuple[str, ...]:
    tags = (
        WANDB_EXPERIMENT_TAG,
        PANEL_TAG,
        "r3d28260",
        SUPPORT_ID,
        run_.arm,
        "transition_ramp",
        "starcoder",
        "wsd80_20",
    )
    invalid = [tag for tag in tags if not 1 <= len(tag) <= MAX_WANDB_TAG_LENGTH]
    if invalid:
        raise ValueError(f"W&B tags must contain 1-{MAX_WANDB_TAG_LENGTH} characters: {invalid}")
    return tags


def build_training_steps(
    *,
    variant: Variant,
    name_prefix: str,
    tpu_type: str,
    tpu_region: str,
    tpu_zone: str,
    selected_runs: frozenset[str] | None = None,
) -> tuple[tuple[RampRun, ...], tuple[ArtifactStep[LevanterCheckpoint], ...]]:
    """Build resumable training handles for the selected ramp rows of one variant."""
    _, runs = load_design(variant, selected_runs)
    requests = _source_requests()
    cells, _ = confirmation.load_design(frozenset(request.run_name for request in requests.values()))
    model = source._validate_runtime_model(cells)
    nemotron = nemotron_datasets(tokenizer=llama3_tokenizer)
    starcoder = dolma_datasets(tokenizer=llama3_tokenizer)["dolma/starcoder"]
    training_handles: tuple[ArtifactStep[TokenizedCache], ...] = (
        *tuple(nemotron[split] for split in base.NEMOTRON_TOKEN_COUNTS),
        starcoder,
    )
    validation_handles = (
        *paloma_datasets(tokenizer=llama3_tokenizer).values(),
        *uncheatable_datasets(tokenizer=llama3_tokenizer).values(),
    )
    resources = ResourceConfig.with_tpu(tpu_type, regions=(tpu_region,), zone=tpu_zone)

    steps: list[ArtifactStep[LevanterCheckpoint]] = []
    for run_ in runs:
        request = requests[run_.data_seed]
        if request.run_name != run_.source_run_name:
            raise ValueError(f"{run_.run_name}: confirmation source row drifted")
        phase_weights = [
            (step, base._phase_leaf_weights(weight, nemotron=nemotron, starcoder=starcoder))
            for step, weight in run_.stages
        ]
        static_weights = {handle: phase_weights[0][1][handle.name] for handle in training_handles}
        training = train_lm(
            name=f"checkpoints/{name_prefix}/{run_.run_name}",
            version=base.VERSION,
            model=model,
            optimizer=base._optimizer(request.materialized_tokens),
            datasets=static_weights,
            validation=validation_handles,
            batch_size=base.BATCH_SIZE,
            seq_len=base.SEQ_LEN,
            num_train_steps=request.total_steps,
            z_loss_weight=None,
            evals=None,
            resources=resources,
            steps_per_eval=1_000,
            wandb_project="marin",
            wandb_group=name_prefix,
            run_id=run_.run_name,
            tags=_wandb_tags(run_),
            env_vars={"HF_ALLOW_CODE_EVAL": "1"},
        )
        steps.append(
            source._with_source_support(
                training,
                train_datasets=static_weights,
                validation_datasets=validation_handles,
                phase_weights=phase_weights,
                starcoder=starcoder,
                starcoder_name=starcoder.name,
                request=request,
            )
        )
    return runs, tuple(steps)


def audit_materialized_runtime_configs(
    payload: dict[str, Any],
    runs: tuple[RampRun, ...],
    steps: tuple[ArtifactStep[LevanterCheckpoint], ...],
    *,
    marin_prefix: str,
) -> int:
    """Materialize every run config and check its stage schedule, cap, seeds, and optimizer."""
    if len(runs) != len(steps):
        raise ValueError(f"Run/step cardinality mismatch: {len(runs)} != {len(steps)}")
    expected_cap = {"dolma/starcoder": int(payload["starcoder_support_batches"])}
    total_steps = int(payload["total_steps"])
    expected_optimizer = asdict(base._optimizer(total_steps * base.BATCH_SIZE * base.SEQ_LEN))
    for run_, step in zip(runs, steps, strict=True):
        config = materialized_config(step, marin_prefix)
        if not isinstance(config, TrainLmOnPodConfig):
            raise TypeError(f"{run_.run_name}: unexpected materialized config {type(config)}")
        train_config = cast(TrainLmConfig, config.train_config)
        if train_config.data.max_train_batches != expected_cap:
            raise ValueError(f"{run_.run_name}: StarCoder support cap drifted: {train_config.data.max_train_batches}")
        if train_config.data_seed != run_.data_seed or train_config.trainer.seed != run_.data_seed:
            raise ValueError(f"{run_.run_name}: data/trainer seed drifted")
        if train_config.data.experiment_budget is not None or train_config.data.target_budget is not None:
            raise ValueError(f"{run_.run_name}: simulated-epoch budget leaked into the intervention")
        if train_config.data.simulated_epoch_subset_seed is not None:
            raise ValueError(f"{run_.run_name}: simulated subset seed leaked into the intervention")
        if train_config.trainer.num_train_steps != total_steps:
            raise ValueError(f"{run_.run_name}: horizon drifted to {train_config.trainer.num_train_steps}")
        if asdict(train_config.optimizer) != expected_optimizer:
            raise ValueError(f"{run_.run_name}: optimizer or learning-rate schedule drifted")
        weights = train_config.data.train_weights
        if not isinstance(weights, list) or [step_ for step_, _ in weights] != [step_ for step_, _ in run_.stages]:
            raise ValueError(f"{run_.run_name}: materialized stage boundaries drifted")
        for (_, leaf), (_, expected) in zip(weights, run_.stages, strict=True):
            if abs(leaf["dolma/starcoder"] - expected) > WEIGHT_TOLERANCE:
                raise ValueError(f"{run_.run_name}: materialized StarCoder weight drifted")
            if abs(sum(leaf.values()) - 1.0) > 1e-9:
                raise ValueError(f"{run_.run_name}: materialized stage weights do not sum to one")
    logger.info("Materialized and audited %d ramp runtime configs", len(runs))
    return len(runs)


def _parse_runs(value: str | None) -> frozenset[str] | None:
    if value is None:
        return None
    runs = frozenset(item.strip() for item in value.split(",") if item.strip())
    if not runs:
        raise argparse.ArgumentTypeError("--runs must contain at least one exact run name")
    return runs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", choices=sorted(VARIANTS), default="main")
    parser.add_argument("--name-prefix", default=NAME)
    parser.add_argument("--marin-prefix", default=base.DEFAULT_MARIN_PREFIX)
    parser.add_argument("--tpu-type", default=base.DEFAULT_TPU_TYPE)
    parser.add_argument("--tpu-region", default=base.DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=base.DEFAULT_TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--runs", help="Comma-separated exact run names for an idempotent partial retry")
    parser.add_argument("--write-design", action="store_true", help="Regenerate the frozen design JSON and exit")
    parser.add_argument("--audit-manifest", action="store_true")
    parser.add_argument("--audit-runtime-identities", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    if os.getenv("CI") is not None:
        logger.info("Skipping transition-ramp launch in CI")
        return
    variant = VARIANTS[args.design]
    if args.write_design:
        payload = build_design(variant)
        with variant.design_path.open("w") as handle:
            json.dump(payload, handle, indent=1)
        logger.info("Wrote %s with sha256 %s", variant.design_path, payload["design_sha256"])
        return
    if args.name_prefix != NAME:
        raise ValueError(f"Ramp checkpoint identity is frozen: {args.name_prefix!r} != {NAME!r}")
    if args.marin_prefix != base.DEFAULT_MARIN_PREFIX:
        raise ValueError(f"Historical StarCoder work must remain central1-local: {args.marin_prefix!r}")
    if args.tpu_type != base.DEFAULT_TPU_TYPE:
        raise ValueError(f"Historical StarCoder accelerator is frozen: {args.tpu_type!r}")
    if args.tpu_region != base.DEFAULT_TPU_REGION or args.tpu_zone != base.DEFAULT_TPU_ZONE:
        raise ValueError("StarCoder child placement must remain us-central1-a")
    if args.max_concurrent < 1:
        raise ValueError("max_concurrent must be positive")

    environment = source._runtime_scientific_environment()
    if environment != source.EXPECTED_TRAINING_ENVIRONMENT:
        logger.warning(
            "Training environment differs from the August pins (%s vs %s); the abrupt canary measures the drift",
            environment,
            source.EXPECTED_TRAINING_ENVIRONMENT,
        )
    selected_runs = _parse_runs(args.runs)
    payload, runs = load_design(variant, selected_runs)
    max_concurrent = min(args.max_concurrent, len(runs))
    logger.info("Prepared %d ramp arms of variant %s", len(runs), variant.name)
    if args.audit_manifest:
        return

    os.environ["MARIN_PREFIX"] = args.marin_prefix
    source._validate_starcoder_source(args.marin_prefix, tuple(_source_requests().values()))
    audit_runtime_sequence_counts(payload, runs)
    runs, steps = build_training_steps(
        variant=variant,
        name_prefix=args.name_prefix,
        tpu_type=args.tpu_type,
        tpu_region=args.tpu_region,
        tpu_zone=args.tpu_zone,
        selected_runs=selected_runs,
    )
    audit_materialized_runtime_configs(payload, runs, steps, marin_prefix=args.marin_prefix)
    if args.audit_runtime_identities:
        return
    if args.dry_run:
        for step in steps:
            lower(step)
        logger.info("Dry-run lowering passed for all %d ramp handles", len(steps))
        return
    run(*steps, max_concurrent=max_concurrent)


if __name__ == "__main__":
    main()
