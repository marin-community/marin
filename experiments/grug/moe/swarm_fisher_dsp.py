# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grug-MoE swarm: 840 Fisher DSP-aligned (tau=20, lambda=0.25) 2-phase mixtures.

Loads (phase_0, phase_1) bucket weights from
``gs://marin-us-central2/grug-moe/swarm_data/production_swarm_168p_uscentral2_d_optimal_mixtures.csv``
(840 candidates x 168 buckets/phase; column names ``phase_{0,1}/c{cc}q{q}`` and
``phase_{0,1}/tail``) and emits one ``ExecutorStep`` per candidate at D512 on
~100B tokens (2.62e19 FLOPs; ~5000 tok/active-param). Levanter's list-form
``train_weights`` is keyed by training STEP (rescaled internally to data
offsets, see ``rescale_mixture_schedule_for_batch_schedule``); we swap weights
at the 80% boundary, snapped to a step where step * batch is a multiple of
mixture_block_size (Levanter requires it).

mixture_block_size is dropped from datakit_moe_mix's 65535 to 32768 because
data_offset = step * batch_size must be a multiple of block_size for stage
boundaries (``MixtureDataset.__init__`` asserts this). batch is a power of 2
and is coprime with 65535, so no intermediate step is valid; 32768 keeps a
power-of-2 alignment and admits any step that satisfies the gcd condition.
The drop zeroes single-unit weights (those at the 1/65535 lattice quantum),
~4% of nonzero entries totaling ~2e-4 mass per candidate.

Components and the 167/33 mixable/tail split are reused from
``datakit_moe_mix`` (their structure is independent of mixture_block_size).
"""

import csv
from math import gcd

import fsspec
from fray.cluster import ResourceConfig
from levanter.data.text import LmDataConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.processing.tokenize import add_validation_sets_to_mixture

from experiments.defaults import default_validation_sets
from experiments.grug.moe.datakit_moe_mix import COMPONENTS, TARGET_BUDGET
from experiments.grug.moe.grug_moe_mix import run_grug_moe_mix
from experiments.grug.moe.heuristic import build_from_heuristic
from experiments.grug.moe.launch import GrugMoeLaunchConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig
from experiments.marin_models import marin_tokenizer

_MIXTURES_CSV = "gs://marin-us-central2/grug-moe/swarm_data/production_swarm_168p_uscentral2_d_optimal_mixtures.csv"

# D512 trained on ~100B tokens (≈5000 tok/active-param, matching a 2B-active /
# 10T-token regime). 2.62e19 FLOPs = 100e9 tokens via build_from_heuristic.
# target_steps=2**16 picks batch=512 (= largest power-of-2 with tokens_per_batch
# <= total_tokens**0.6, matching the critical-batch heuristic).
_BUDGET, _HIDDEN_DIM = 2.62e19, 512
_TARGET_STEPS: int = 2**16

# Override datakit_moe_mix's _MIXTURE_BLOCK_SIZE (65535) — see module docstring.
_SWARM_BLOCK_SIZE = 32768

_PHASE_PREFIXES: tuple[str, str] = ("phase_0/", "phase_1/")

# Hoisted from _build_step: model/optimizer/batch/steps are pure functions of
# the module constants above, so this is constant across all 840 candidates.
_MODEL, _OPTIMIZER, _BATCH, _STEPS = build_from_heuristic(
    budget=_BUDGET,
    hidden_dim=_HIDDEN_DIM,
    target_steps=_TARGET_STEPS,
)
_EXPERIMENT_BUDGET: int = _BATCH * _STEPS * _MODEL.max_seq_len
assert _EXPERIMENT_BUDGET <= TARGET_BUDGET, f"experiment_budget {_EXPERIMENT_BUDGET} exceeds {TARGET_BUDGET}"

# 80/20 phase split: phase_0 takes the first 80% of steps, phase_1 the final
# 20%. The step boundary must satisfy step * batch % _SWARM_BLOCK_SIZE == 0,
# i.e. step must be a multiple of block_size / gcd(batch, block_size). Snap down.
_PHASE1_STEP_MULTIPLE: int = _SWARM_BLOCK_SIZE // gcd(_BATCH, _SWARM_BLOCK_SIZE)
_PHASE1_START_STEP: int = (_STEPS * 4 // 5 // _PHASE1_STEP_MULTIPLE) * _PHASE1_STEP_MULTIPLE
assert _PHASE1_START_STEP * _BATCH % _SWARM_BLOCK_SIZE == 0


def _load_swarm_mixtures(path: str) -> list[tuple[int, dict[str, float], dict[str, float]]]:
    with fsspec.open(path, "rt") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    p0, p1 = _PHASE_PREFIXES
    bucket_names = [c[len(p0) :] for c in fieldnames if c.startswith(p0)]
    assert len(bucket_names) == 168
    assert [c[len(p1) :] for c in fieldnames if c.startswith(p1)] == bucket_names
    missing = set(bucket_names) - set(COMPONENTS)
    assert not missing, f"swarm buckets missing from COMPONENTS: {sorted(missing)}"
    candidates = []
    for r in rows:
        idx = int(r["experiment_index"])
        w0 = {b: float(r[f"{p0}{b}"]) for b in bucket_names}
        w1 = {b: float(r[f"{p1}{b}"]) for b in bucket_names}
        candidates.append((idx, w0, w1))
    indices = [c[0] for c in candidates]
    assert len(set(indices)) == len(indices), "duplicate experiment_index in swarm CSV"
    return candidates


def _build_step(idx: int, w0: dict[str, float], w1: dict[str, float]) -> ExecutorStep:
    mixture_schedule: list[tuple[int, dict[str, float]]] = [(0, w0), (_PHASE1_START_STEP, w1)]

    base_mixture = LmDataConfig(
        tokenizer=marin_tokenizer,
        cache_dir=None,
        components=COMPONENTS,
        train_weights=mixture_schedule,
        auto_build_caches=False,
        mixture_block_size=_SWARM_BLOCK_SIZE,
        target_budget=TARGET_BUDGET,
        experiment_budget=_EXPERIMENT_BUDGET,
    )
    data = add_validation_sets_to_mixture(base_mixture, default_validation_sets(tokenizer=marin_tokenizer))

    slug = f"d{_HIDDEN_DIM}_{idx:06d}"
    return ExecutorStep(
        name=f"grug/swarm_fisher_dsp_{slug}",
        fn=run_grug_moe_mix,
        config=GrugMoeLaunchConfig(
            model=versioned(_MODEL),
            data=data,
            output_path=this_output_path(),
            run_id=f"swarm_fisher_dsp_{slug}",
            resources=versioned(ResourceConfig.with_tpu("v4-8", zone="us-central2-b", preemptible=False)),
            steps=versioned(_STEPS),
            batch_size=versioned(_BATCH),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(
                project="marin_moe",
                tags=["moe", "swarm", "fisher_dsp", "tau20_lam0p25", "uscentral2", slug],
                group="swarm_fisher_dsp_tau20_lam0p25_uscentral2",
                name=None,
            ),
            optimizer=versioned(_OPTIMIZER),
            grug_trainer=versioned(GrugTrainerConfig(z_loss_weight=1e-4, ema_beta=None, log_every=1)),
            eval=versioned(
                GrugEvalConfig(
                    eval_batch_size=_BATCH,
                    steps_per_eval=1000,
                    eval_current=True,
                    eval_ema=False,
                )
            ),
        ),
    )


_CANDIDATES = _load_swarm_mixtures(_MIXTURES_CSV)

swarm_fisher_dsp_steps: list[ExecutorStep] = [_build_step(idx, w0, w1) for idx, w0, w1 in _CANDIDATES]


if __name__ == "__main__":
    executor_main(
        steps=swarm_fisher_dsp_steps,
        description=(
            "Grug MoE swarm: 840 Fisher DSP-aligned (tau=20, lambda=0.25) two-phase mixtures on "
            "the us-central2 datakit store, D512 trained on ~100B tokens."
        ),
        # Matches v4-reserved size 8 max_slices in marin.yaml; StepRunner's
        # implicit default is 8, which would serialize the swarm.
        max_concurrent=256,
    )
