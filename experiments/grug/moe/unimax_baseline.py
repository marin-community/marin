# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""UniMax baseline at the swarm's training shape.

Same model/optimizer/steps/phase split/resources as `launch_swarm.py` and
`proportional_baseline.py`, but the train_weights are computed via UniMax
(Chung et al., ICLR 2023) with an 8-epoch cap **per phase**.

Token counts come from the datakit store's ``.artifact.json``, which holds the
authoritative per-bucket ``total_tokens`` field. UniMax is applied at the
simulated corpus scale (``TARGET_BUDGET``) the dashboard uses — at the actual
100B-token training budget, even the smallest bucket (210M tokens) clears the
60M cap-activation threshold, so N=8 degenerates to fully uniform there. Using
``TARGET_BUDGET`` (8.3T phase_0 / 2.07T phase_1) is what makes the 8-epoch
ceiling actually bind on the small buckets, which is the whole point of UniMax.
"""

import json
import os

import fsspec
from fray.cluster import ResourceConfig
from levanter.data.text import LmDataConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main
from marin.execution.types import this_output_path, versioned
from marin.processing.tokenize import add_validation_sets_to_mixture

from experiments.defaults import default_validation_sets
from experiments.grug.moe.datakit_moe_mix import COMPONENTS, TARGET_BUDGET
from experiments.grug.moe.grug_moe_mix import GrugEvalConfig, GrugMoeLaunchConfig, GrugTrainerConfig, run_grug_moe_mix
from experiments.grug.moe.launch_swarm import (
    _BATCH,
    _BUDGET,
    _EXPERIMENT_BUDGET,
    _HIDDEN_DIM,
    _MODEL,
    _OPTIMIZER,
    _PHASE1_START_STEP,
    _STEPS,
    _SWARM_BLOCK_SIZE,
)
from experiments.marin_models import marin_tokenizer

_STORE_ARTIFACT = "gs://marin-us-central2/datakit/store_8ac06c74/.artifact.json"
# Override via `UNIMAX_EPOCH_CAP=4 python experiments/grug/moe/unimax_baseline.py`
# at submission time; the cap lands in the step name (`d512_unimax_n{cap}`) so
# distinct caps produce distinct ExecutorSteps that don't collide.
_EPOCH_CAP = int(os.environ.get("UNIMAX_EPOCH_CAP", "8"))
_PHASE0_FRAC = 0.8
_PHASE1_FRAC = 0.2


def _bucket_token_counts() -> dict[str, int]:
    """Read per-bucket ``total_tokens`` from the datakit store's artifact.

    The artifact lists every raw (cluster, quality) cell. The swarm collapses
    a handful of the smallest cells into a single synthetic ``tail`` bucket
    (see ``datakit_moe_mix._TAIL_BUCKETS``); we mirror that here by summing
    the artifact entries that aren't represented individually in
    ``COMPONENTS`` and emitting them under the ``tail`` key.
    """
    fs = fsspec.filesystem("gs")
    with fs.open(_STORE_ARTIFACT, "rt") as f:
        artifact = json.load(f)
    raw = {f"c{b['cluster_id']:02d}q{b['quality_bucket']}": int(b["total_tokens"]) for b in artifact["buckets"]}
    mixable = {k: v for k, v in raw.items() if k in COMPONENTS}
    tail_total = sum(v for k, v in raw.items() if k not in COMPONENTS)
    if tail_total > 0 and "tail" in COMPONENTS:
        mixable["tail"] = tail_total
    return mixable


def _unimax_weights(token_counts: dict[str, int], budget: float, epoch_cap: float) -> dict[str, float]:
    """UniMax allocation: uniform-up-to-epoch-cap.

    Sort buckets smallest-first. For each, if the per-bucket uniform share of
    the remaining budget exceeds ``epoch_cap * tokens``, pin it at the cap and
    redistribute the surplus uniformly across the rest. As soon as one bucket
    can absorb the uniform share, every larger bucket can too, so the
    allocation terminates with a single uniform pass.
    """
    items = sorted(token_counts.items(), key=lambda kv: kv[1])
    allocated: dict[str, float] = {}
    rem = float(budget)
    for i, (name, tokens) in enumerate(items):
        uniform_share = rem / (len(items) - i)
        cap = epoch_cap * tokens
        if uniform_share > cap:
            allocated[name] = cap
            rem -= cap
            continue
        for n2, _ in items[i:]:
            allocated[n2] = uniform_share
        break
    total = sum(allocated.values())
    return {k: v / total for k, v in allocated.items()}


_TOKEN_COUNTS_FROM_ARTIFACT = _bucket_token_counts()
assert len(_TOKEN_COUNTS_FROM_ARTIFACT) == len(
    COMPONENTS
), f"artifact buckets ({len(_TOKEN_COUNTS_FROM_ARTIFACT)}) != COMPONENTS ({len(COMPONENTS)})"

PHASE0_WEIGHTS = _unimax_weights(_TOKEN_COUNTS_FROM_ARTIFACT, TARGET_BUDGET * _PHASE0_FRAC, _EPOCH_CAP)
PHASE1_WEIGHTS = _unimax_weights(_TOKEN_COUNTS_FROM_ARTIFACT, TARGET_BUDGET * _PHASE1_FRAC, _EPOCH_CAP)
print(
    f"unimax_baseline: cap={_EPOCH_CAP} epochs/phase  "
    f"phase_0 budget={TARGET_BUDGET * _PHASE0_FRAC / 1e12:.2f}T "
    f"(min_w={min(PHASE0_WEIGHTS.values()):.2e}, max_w={max(PHASE0_WEIGHTS.values()):.2e})  "
    f"phase_1 budget={TARGET_BUDGET * _PHASE1_FRAC / 1e12:.2f}T "
    f"(min_w={min(PHASE1_WEIGHTS.values()):.2e}, max_w={max(PHASE1_WEIGHTS.values()):.2e})"
)


def _build_step() -> ExecutorStep:
    mixture_schedule: list[tuple[int, dict[str, float]]] = [
        (0, PHASE0_WEIGHTS),
        (_PHASE1_START_STEP, PHASE1_WEIGHTS),
    ]

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

    slug = f"d{_HIDDEN_DIM}_unimax_n{_EPOCH_CAP}"
    return ExecutorStep(
        name=f"grug/unimax_baseline_{slug}",
        fn=run_grug_moe_mix,
        config=GrugMoeLaunchConfig(
            model=versioned(_MODEL),
            data=data,
            output_path=this_output_path(),
            run_id=f"unimax_baseline_{slug}",
            resources=versioned(ResourceConfig.with_tpu("v4-8", zone="us-central2-b", preemptible=False)),
            steps=versioned(_STEPS),
            batch_size=versioned(_BATCH),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(
                project="marin_moe",
                tags=["moe", "unimax_baseline", f"cap{_EPOCH_CAP}", "uscentral2", slug],
                group="unimax_baseline_uscentral2",
                name=None,
            ),
            optimizer=versioned(_OPTIMIZER),
            grug_trainer=versioned(GrugTrainerConfig(z_loss_weight=1e-4, ema_beta=None, log_every=1)),
            eval=versioned(
                GrugEvalConfig(
                    eval_batch_size=512,
                    steps_per_eval=1000,
                    max_eval_batches=8,
                    eval_current=True,
                    eval_ema=False,
                )
            ),
        ),
    )


unimax_baseline_steps: list[ExecutorStep] = [_build_step()]


if __name__ == "__main__":
    executor_main(
        steps=unimax_baseline_steps,
        description=(
            f"UniMax (N={_EPOCH_CAP}) grug-MoE baseline at swarm shape: d={_HIDDEN_DIM}, "
            f"budget={_BUDGET:.2e} FLOPs, steps={_STEPS}, 80/20 phase split at "
            f"{_PHASE1_START_STEP}/{_STEPS}. Weights computed per phase against "
            f"TARGET_BUDGET={TARGET_BUDGET/1e12:.2f}T tokens (simulated corpus scale)."
        ),
    )
