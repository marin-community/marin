# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Two 100-step diagnostic runs to isolate the NaN observed in
`nano_adamh_heuristic_trial` (run_id `may7-nano-adamh-heuristic`):

  Test A — `may7-nano-heur-test-a-lm-adam-nonzero`
      Heuristic optimizer + **lm head on AdamW** (instead of AdamH).
      Init unchanged from the failed run (default truncated-normal,
      `zero_init_proj=False`, so the lm head starts non-zero). If the failed
      run's NaN was caused by AdamH's Frobenius-preserving update on the
      (768, 50304) lm head with `epsilon` near zero, this run trains cleanly.

  Test B — `may7-nano-heur-test-b-lm-adam-zero`
      Heuristic optimizer + **lm head on AdamW** + **zero-init lm head**
      (init_scheme="adamh_ref"). Block weights are still Kaiming-with-multipliers
      so AdamH has non-zero matrices. This is the natural, well-conditioned
      AdamW + AdamH split: AdamW grows the lm head from zero; AdamH preserves
      block weight norms. Step-0 train loss should be exactly ln(50304) =
      10.825 (zero logits → uniform softmax).

Both share the heuristic optimizer config, recomputed for the new (small)
step count: at 100 steps with batch=512, tpb=524k, total_tokens=5.24e7,
the heuristic gives adamh_lr=0.0466, adam_lr=0.01076, beta2=0.996006,
epsilon=9.64e-17. Warmup=0.1 → 10 steps; linear decay over the rest.

The two steps run sequentially under one iris job (~10 min total at ~5
min/run including compile).
"""

from dataclasses import dataclass

import jax
from fray.cluster import ResourceConfig
from levanter.optim import OptimizerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.utils.jax_utils import leaf_key_paths
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.moe.heuristic import MoeAdamHHeuristic
from experiments.grug.nano.launch_adamh_heuristic import (
    NanoAdamHHeuristicLaunchConfig,
    NanoHeuristicAdamHConfig,
    _fineweb_gpt2_data,
    _resolve_run_id,
    run_nano_adamh_heuristic_trial,
)
from experiments.grug.nano.model import NanoModelConfig
from experiments.grug.nano.train import GrugEvalConfig, GrugTrainerConfig

NANO_HEUR_TEST_TRAIN_STEPS = 100


# ---- Optimizer subclass: route lm head to AdamW instead of AdamH ----


@OptimizerConfig.register_subclass("nano_adamh_heuristic_lm_adam")
@dataclass(frozen=True)
class NanoHeuristicAdamHLmAdamConfig(NanoHeuristicAdamHConfig):
    """Diagnostic variant: lm head goes to AdamW, not AdamH.

    Same hyperparameters and base groups as `NanoHeuristicAdamHConfig`. The
    only change is `_create_mask`: any 2-D parameter outside `blocks` (i.e.
    the lm head) is routed to the `adam` group, so AdamH only operates on
    block weights. Embed and 1-D scalars/biases continue to land on `adam`.
    """

    def _create_mask(self, params):
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            path_lower = path_str.lower()

            ndim = getattr(param, "ndim", None)
            if ndim is None or ndim < 2:
                return "adam"

            if "embed" in path_lower:
                return "adam"

            if "blocks" in path_lower:
                return "adamh"

            # Top-level 2-D weights (= the lm head) go to AdamW here, instead
            # of AdamH as in the original heuristic mask.
            return "adam"

        return jax.tree.map(mask_fn, params, paths)


def build_heuristic_optimizer_lm_adam(
    *,
    batch_size: int,
    num_train_steps: int,
    seq_len: int,
    hidden_dim: int,
    heuristic: MoeAdamHHeuristic | None = None,
) -> NanoHeuristicAdamHLmAdamConfig:
    h = heuristic or MoeAdamHHeuristic()
    tpb = batch_size * seq_len
    total_tokens = tpb * num_train_steps
    return NanoHeuristicAdamHLmAdamConfig(
        learning_rate=h._compute_learning_rate(tpb, total_tokens, hidden_dim),
        adam_lr=h._compute_adam_lr(tpb, total_tokens, hidden_dim),
        beta1=h.beta1,
        beta2=h._compute_beta2(tpb),
        epsilon=h._compute_epsilon(tpb, total_tokens),
        max_grad_norm=h.max_grad_norm,
        weight_decay=0.0,
        warmup=h.warmup,
        decay=h.decay,
        lr_schedule=h.lr_schedule,
        min_lr_ratio=h.min_lr_ratio,
    )


# ---- Models ----


# Test A: lm head NON-zero (default truncated-normal init). Same model init
# as the failing `nano_adamh_heuristic_trial`; the only change is the mask.
NANO_TEST_A_MODEL = NanoModelConfig(
    vocab_size=50304,
    hidden_dim=768,
    intermediate_dim=3072,
    num_layers=12,
    num_heads=6,
    head_dim=128,
    max_seq_len=1024,
    zero_init_proj=False,
    init_scheme="default",
)


# Test B: lm head ZERO-init via the adamh_ref init scheme (lm head is
# explicitly zeroed; block weights are Kaiming-with-multipliers).
NANO_TEST_B_MODEL = NanoModelConfig(
    vocab_size=50304,
    hidden_dim=768,
    intermediate_dim=3072,
    num_layers=12,
    num_heads=6,
    head_dim=128,
    max_seq_len=1024,
    zero_init_proj=False,  # required by init_scheme="adamh_ref"
    init_scheme="adamh_ref",
)


# Shared optimizer (same hyperparams for both tests).
NANO_HEUR_TEST_OPTIMIZER = build_heuristic_optimizer_lm_adam(
    batch_size=512,
    num_train_steps=NANO_HEUR_TEST_TRAIN_STEPS,
    seq_len=1024,
    hidden_dim=768,
)


# ---- Launch steps ----


def _make_test_step(name: str, run_id: str, model: NanoModelConfig, extra_tags: list[str]) -> ExecutorStep:
    return ExecutorStep(
        name=name,
        fn=run_nano_adamh_heuristic_trial,
        config=NanoAdamHHeuristicLaunchConfig(
            model=versioned(model),
            data=_fineweb_gpt2_data(),
            output_path=this_output_path(),
            run_id=run_id,
            resources=versioned(ResourceConfig.with_tpu("v5p-8")),
            steps=versioned(NANO_HEUR_TEST_TRAIN_STEPS),
            batch_size=versioned(512),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(
                project="marin",
                tags=["grug", "nano", "adamh", "fineweb-gpt2", "heuristic", "diagnostic", *extra_tags],
                group="nano-trial",
                name=None,
                replicate_path=this_output_path(),
            ),
            optimizer=versioned(NANO_HEUR_TEST_OPTIMIZER),
            grug_trainer=versioned(
                GrugTrainerConfig(
                    z_loss_weight=1e-4,
                    ema_beta=None,
                    log_every=1,
                )
            ),
            eval=versioned(
                GrugEvalConfig(
                    eval_batch_size=512,
                    steps_per_eval=25,  # eval often during the short run
                    max_eval_batches=20,
                    eval_current=True,
                    eval_ema=False,
                )
            ),
        ),
    )


nano_heur_test_a = _make_test_step(
    name="grug/nano-heur-test-a-lm-adam-nonzero",
    run_id=_resolve_run_id("may7-nano-heur-test-a-lm-adam-nonzero"),
    model=NANO_TEST_A_MODEL,
    extra_tags=["test-a", "lm-adam", "nonzero-init"],
)


nano_heur_test_b = _make_test_step(
    name="grug/nano-heur-test-b-lm-adam-zero",
    run_id=_resolve_run_id("may7-nano-heur-test-b-lm-adam-zero"),
    model=NANO_TEST_B_MODEL,
    extra_tags=["test-b", "lm-adam", "zero-init"],
)


if __name__ == "__main__":
    executor_main(
        steps=[nano_heur_test_a, nano_heur_test_b],
        description="Two 100-step diagnostics: lm-head-on-AdamW with non-zero (A) vs zero (B) lm-head init.",
    )
