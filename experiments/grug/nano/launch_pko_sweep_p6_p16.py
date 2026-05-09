# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""PKO sweep: adamh-heuristic baseline at p6..p16 with PKO every 4th layer.

Eleven runs, all under one iris job. Each run inherits the corresponding
adamh-heuristic walk's model (p6 -> p16) and flips on
``use_pko_every_4th=True``: every 4th transformer block (i % 4 == 3, the
same long-window layers as the sliding-window pattern) shifts the second
half of its key tensor forward by one position after RoPE.

Source semantics: ``experiments/grug/moe/model.py`` PKO commits
(2b00190a9 + 15684cb39). We only port the key shift; nano's
half-truncated rope stays in place.

Each run uses its own (model, batch, num_steps, optimizer) — the
heuristic LRs / β / ε are recomputed per-step from
``MoeAdamHHeuristic`` at that walk's scale.

Wandb names: ``may7-nano-adamh-heuristic-p{N}-pko``.
"""

import dataclasses
from dataclasses import dataclass

from fray.cluster import ResourceConfig
from jax.sharding import PartitionSpec as P
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.nano.launch import _fineweb_gpt2_data
from experiments.grug.nano.launch_adamh_heuristic import (
    NanoAdamHHeuristicLaunchConfig,
    _resolve_run_id,
    build_heuristic_optimizer,
    run_nano_adamh_heuristic_trial,
)
from experiments.grug.nano.launch_adamh_heuristic_walk_p6 import (
    P6_BATCH_SIZE,
    P6_MODEL,
    P6_TRAIN_STEPS,
)
from experiments.grug.nano.launch_adamh_heuristic_walk_p7 import (
    P7_BATCH_SIZE,
    P7_MODEL,
    P7_TRAIN_STEPS,
)
from experiments.grug.nano.launch_adamh_heuristic_walk_p8 import (
    P8_BATCH_SIZE,
    P8_MODEL,
    P8_TRAIN_STEPS,
)
from experiments.grug.nano.launch_adamh_heuristic_walk_p9 import (
    P9_BATCH_SIZE,
    P9_MODEL,
    P9_TRAIN_STEPS,
)
from experiments.grug.nano.launch_adamh_heuristic_walk_p10 import (
    P10_BATCH_SIZE,
    P10_MODEL,
    P10_TRAIN_STEPS,
)
from experiments.grug.nano.launch_adamh_heuristic_walk_p11 import (
    P11_BATCH_SIZE,
    P11_MODEL,
    P11_TRAIN_STEPS,
)
from experiments.grug.nano.launch_adamh_heuristic_walk_p12 import (
    P12_BATCH_SIZE,
    P12_MODEL,
    P12_TRAIN_STEPS,
)
from experiments.grug.nano.launch_adamh_heuristic_walk_p13 import (
    P13_BATCH_SIZE,
    P13_MODEL,
    P13_TRAIN_STEPS,
)
from experiments.grug.nano.launch_adamh_heuristic_walk_p14 import (
    P14_BATCH_SIZE,
    P14_MODEL,
    P14_TRAIN_STEPS,
)
from experiments.grug.nano.launch_adamh_heuristic_walk_p15 import (
    P15_BATCH_SIZE,
    P15_MODEL,
    P15_TRAIN_STEPS,
)
from experiments.grug.nano.launch_adamh_heuristic_walk_p16 import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    P16_BATCH_SIZE,
    P16_MODEL,
    P16_TRAIN_STEPS,
)
from experiments.grug.nano.train import GrugEvalConfig, GrugTrainerConfig


@dataclass(frozen=True)
class _WalkSpec:
    pkey: str  # "p6", "p7", ...
    model: object
    batch_size: int
    train_steps: int
    use_nemotron: bool


# p6..p15 share the fineweb_gpt2 data path; p16 switches to nemotron + llama3.
WALKS: list[_WalkSpec] = [
    _WalkSpec("p6", P6_MODEL, P6_BATCH_SIZE, P6_TRAIN_STEPS, False),
    _WalkSpec("p7", P7_MODEL, P7_BATCH_SIZE, P7_TRAIN_STEPS, False),
    _WalkSpec("p8", P8_MODEL, P8_BATCH_SIZE, P8_TRAIN_STEPS, False),
    _WalkSpec("p9", P9_MODEL, P9_BATCH_SIZE, P9_TRAIN_STEPS, False),
    _WalkSpec("p10", P10_MODEL, P10_BATCH_SIZE, P10_TRAIN_STEPS, False),
    _WalkSpec("p11", P11_MODEL, P11_BATCH_SIZE, P11_TRAIN_STEPS, False),
    _WalkSpec("p12", P12_MODEL, P12_BATCH_SIZE, P12_TRAIN_STEPS, False),
    _WalkSpec("p13", P13_MODEL, P13_BATCH_SIZE, P13_TRAIN_STEPS, False),
    _WalkSpec("p14", P14_MODEL, P14_BATCH_SIZE, P14_TRAIN_STEPS, False),
    _WalkSpec("p15", P15_MODEL, P15_BATCH_SIZE, P15_TRAIN_STEPS, False),
    _WalkSpec("p16", P16_MODEL, P16_BATCH_SIZE, P16_TRAIN_STEPS, True),
]


def _build_step(spec: _WalkSpec) -> ExecutorStep:
    pko_model = dataclasses.replace(spec.model, use_pko_every_4th=True)
    optimizer = build_heuristic_optimizer(
        batch_size=spec.batch_size,
        num_train_steps=spec.train_steps,
        seq_len=pko_model.max_seq_len,
        hidden_dim=pko_model.hidden_dim,
    )
    data = NEMOTRON_MIX_WITH_DEFAULT_VALIDATION if spec.use_nemotron else _fineweb_gpt2_data()

    # MoE walks (p13+) need the (data, expert) batch sharding; pre-MoE walks
    # use the default P("data") so batch tensors don't pick up an unused
    # expert axis spec.
    is_moe = bool(getattr(pko_model, "use_moe", False))
    train_pspec = P(("data", "expert")) if is_moe else P(("data",))
    eval_pspec = P(("data", "expert")) if is_moe else P(("data",))

    run_id = _resolve_run_id(f"may7-nano-adamh-heuristic-{spec.pkey}-pko")
    # Match each walk's existing eval cadence: p6-p11 had steps_per_eval=125
    # at 3350 steps; p12+ (longer runs) used 250.
    steps_per_eval = 125 if spec.train_steps <= 3500 else 250
    # p6-p11 used eval_batch=BS=128 with max_eval_batches=20 (5.2M tokens).
    # p12+ used 40 (10.5M tokens). Match that.
    max_eval_batches = 20 if spec.train_steps <= 3500 else 40

    return ExecutorStep(
        name=f"grug/nano-adamh-heuristic-{spec.pkey}-pko-trial",
        fn=run_nano_adamh_heuristic_trial,
        config=NanoAdamHHeuristicLaunchConfig(
            model=versioned(pko_model),
            data=data,
            output_path=this_output_path(),
            run_id=run_id,
            resources=versioned(ResourceConfig.with_tpu("v5p-8")),
            steps=versioned(spec.train_steps),
            batch_size=versioned(spec.batch_size),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(
                project="marin",
                tags=[
                    "grug",
                    "nano",
                    "adamh",
                    "heuristic",
                    "pko",
                    spec.pkey,
                    "nemotron" if spec.use_nemotron else "fineweb-gpt2",
                ],
                group="nano-trial",
                name=None,
                replicate_path=this_output_path(),
            ),
            optimizer=versioned(optimizer),
            grug_trainer=versioned(
                GrugTrainerConfig(
                    z_loss_weight=1e-4,
                    ema_beta=None,
                    log_every=1,
                    train_batch_pspec=train_pspec,
                )
            ),
            eval=versioned(
                GrugEvalConfig(
                    eval_batch_size=spec.batch_size,
                    steps_per_eval=steps_per_eval,
                    max_eval_batches=max_eval_batches,
                    eval_current=True,
                    eval_ema=False,
                    eval_batch_pspec=eval_pspec,
                )
            ),
        ),
    )


_STEPS = [_build_step(w) for w in WALKS]


if __name__ == "__main__":
    executor_main(
        steps=_STEPS,
        description="adamh-heuristic + PKO (every 4th layer) sweep across p6..p16 (11 runs).",
    )
