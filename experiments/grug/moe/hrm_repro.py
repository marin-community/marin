# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""HRM-Text reproduction with the marin MoE recipe.

Reference upstream:
- Model + training: https://github.com/sapientinc/HRM-Text
- Data pipeline:    https://github.com/sapientinc/data_io
- HF model card:    https://huggingface.co/sapientinc/HRM-Text-1B
- Cleaned data:     https://huggingface.co/datasets/sapientinc/HRM-Text-data-io-cleaned-20260515

HRM-Text trains a recurrent text model on a stratified mixture of instruction/
response pairs with PrefixLM packing and target-only loss. The reference XL
1B-parameter model is reported as having seen **40B unique tokens** (HF model
card "Unique tokens trained on: 40B"), corresponds to ``epochs * total_length``
in the upstream config (see ``HRM-Text/pretrain.py::total_steps``).

This module reproduces the same data + token budget with the current marin MoE
recipe (`experiments/grug/moe/`):
- Data: ``sapientinc/HRM-Text-data-io-cleaned-20260515`` tokenized with our
  default Llama-3 tokenizer (vocab 128_256, matches the MoE heuristic). The
  ``instruction``/``response`` columns drive ``SupervisedLmDatasetFormat`` so
  the loss is masked to response tokens — matching HRM-Text's ``target_only``.
- Model: ``MoeAdamHHeuristic.build_model_config(hidden_dim)`` (sized by the
  heuristic, e.g. d768 → 8 layers, d1280 → 13 layers).
- Optimizer + batch + LR: ``MoeAdamHHeuristic.build_optimizer_config`` with the
  token budget passed explicitly so it scales correctly.
- Token budget: ``HRM_REPRO_TARGET_TOKENS`` env var (default 40e9, matching
  HRM XL's 40B unique tokens).
- Steps: ``num_steps = tokens // (batch_size * seq_len)``.

Wall-clock estimates on v5p-8 (from the moe-v16 baseline table in
``experiments/grug/moe/README.md``):

| Dim   | tok/s (v5p-8) | 40B tokens runtime |
|-------|---------------|--------------------|
| 768   | ~273k         | ~40.7 h            |
| 1280  | ~128k         | ~86.8 h (~3.6 d)   |

The d768 entry is the canonical *test run* — same data + heuristic + token
budget as the full d1280 reproduction, just smaller and ~half the wall-clock.

Submit (preemptible, per ``feedback_grug_moe_iris_preemptible``):

    .venv/bin/iris --config lib/iris/config/marin.yaml job run \\
      --no-wait --preemptible --reserve v5p-8 \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.hrm_repro

Override the model size via ``HRM_REPRO_HIDDEN_DIM`` (768 default, 1280 for
the full reproduction). Override the token budget with ``HRM_REPRO_TARGET_TOKENS``.

## In-training evaluation

The training step already runs perplexity eval every 1000 steps against
``default_validation_sets`` (Paloma + uncheatable-eval shards from
``experiments/defaults.py``), which is what is wired into
``GrugEvalConfig`` here. That gives a per-domain perplexity track that's
directly comparable across reproductions.

## Downstream benchmarks (HRM-Text Table)

HRM-Text reports GSM8k, MATH, DROP, MMLU, ARC-C, HellaSwag, Winogrande,
BoolQ. The equivalent task aliases in ``experiments/evals/task_configs.py``
are listed in ``HRM_TEXT_BENCHMARKS`` below.

Running these on the Grug MoE checkpoint requires a Grug → Levanter /
HuggingFace exporter (the in-training ``levanter.eval_harness.lm_eval_harness``
callback expects an ``LmHeadModel`` interface, which Grug's Equinox
``Transformer`` does not implement, and ``marin.evaluation.evaluators`` expects
a vLLM-compatible HF format). That exporter does not exist yet for the MoE
template, so this module only declares the intended benchmark set; the
downstream sweep is left as a follow-up step.
"""

import math
import os

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.processing.tokenize import add_validation_sets_to_mixture

from experiments.defaults import default_validation_sets
from experiments.evals.task_configs import EvalTaskConfig
from experiments.grug.moe.heuristic import (
    DEFAULT_TARGET_STEPS,
    MIN_BATCH_SIZE,
    SEQ_LEN,
    compute_flops_per_token,
    moe_adamh_heuristic,
)
from experiments.grug.moe.hrm_eval import HRM_TEXT_BENCHMARKS_DEFAULT, hrm_eval_step
from experiments.grug.moe.hrm_text_data import (
    hrm_text_clean_steps,
    hrm_text_mixture,
    hrm_text_tokenize_steps,
)
from experiments.grug.moe.launch import GrugMoeLaunchConfig, run_grug_moe_trial
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig
from experiments.llama import llama3_tokenizer

# Downstream benchmarks reported in the HRM-Text README table. Settings follow
# the marin canonical aliases in ``experiments/evals/task_configs.py``.
HRM_TEXT_BENCHMARKS: tuple[EvalTaskConfig, ...] = (
    EvalTaskConfig(name="gsm8k_cot", num_fewshot=8),
    EvalTaskConfig(name="minerva_math", num_fewshot=4, task_alias="math_4shot"),
    EvalTaskConfig(name="drop", num_fewshot=0),
    EvalTaskConfig("mmlu", 0, task_alias="mmlu_0shot"),
    EvalTaskConfig("arc_challenge", 10),
    EvalTaskConfig("hellaswag", 10, task_alias="hellaswag_10shot"),
    EvalTaskConfig("winogrande", 0),
    EvalTaskConfig("boolq", 10),
)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

# The full HRM-Text training mix is ported source-by-source in
# ``hrm_text_data.py``; here we just splice in the standard marin validation
# sets so the training loop reports per-domain perplexity.
HRM_TEXT_MIX = add_validation_sets_to_mixture(
    hrm_text_mixture(),
    default_validation_sets(tokenizer=llama3_tokenizer),
)


# ---------------------------------------------------------------------------
# Model + optimizer sizing
# ---------------------------------------------------------------------------

# HF model card: "Unique tokens trained on: 40B" for the XL (1B-param) reference.
HRM_XL_TARGET_TOKENS: float = 4.0e10

# v5p-8 tok/s from experiments/grug/moe/README.md compute-optimal baselines.
_V5P_8_TOKS_PER_SEC: dict[int, float] = {
    512: 405_630.0,
    768: 273_532.0,
    1024: 175_165.0,
    1280: 128_277.0,
}


def _round_to_power_of_two(x: float) -> int:
    if x <= 1:
        return 1
    return 2 ** math.ceil(math.log2(x))


def _build_hrm_repro_step(
    *,
    hidden_dim: int,
    target_tokens: float,
    run_suffix: str,
) -> tuple[ExecutorStep, ExecutorStep]:
    """Build a single-step training launch sized to ``target_tokens``.

    Uses ``MoeAdamHHeuristic`` for the model and optimizer, but pins the token
    count explicitly (instead of deriving it from a compute budget) so we land
    on the same training-token budget as HRM-Text XL regardless of how the
    isoflop coefficients drift.
    """
    h = moe_adamh_heuristic
    model = h.build_model_config(hidden_dim, seq_len=SEQ_LEN)
    fpt = compute_flops_per_token(model)

    batch_exact = target_tokens / (DEFAULT_TARGET_STEPS * SEQ_LEN)
    batch_size = max(MIN_BATCH_SIZE, _round_to_power_of_two(batch_exact))
    num_steps = max(1, round(target_tokens / (batch_size * SEQ_LEN)))
    achieved_tokens = batch_size * SEQ_LEN * num_steps
    compute_budget = 3.0 * fpt * achieved_tokens

    optimizer = h.build_optimizer_config(batch_size, achieved_tokens, hidden_dim, seq_len=SEQ_LEN)

    eta_hours = (
        achieved_tokens / _V5P_8_TOKS_PER_SEC[hidden_dim] / 3600 if hidden_dim in _V5P_8_TOKS_PER_SEC else float("nan")
    )

    run_id = f"hrm-repro-d{hidden_dim}-{achieved_tokens:.2e}".replace("+", "")
    if run_suffix:
        run_id = f"{run_id}-{run_suffix}"
    step_name = f"grug/hrm_repro/{run_id}"

    print(
        "[hrm_repro] "
        f"hidden_dim={hidden_dim} layers={model.num_layers} "
        f"tokens={achieved_tokens:.3e} batch_size={batch_size} num_steps={num_steps} "
        f"fpt={fpt:.3e} compute_budget={compute_budget:.3e} "
        f"v5p-8 ETA={eta_hours:.1f}h"
    )

    train_step = ExecutorStep(
        name=step_name,
        fn=run_grug_moe_trial,
        config=GrugMoeLaunchConfig(
            model=versioned(model),
            data=HRM_TEXT_MIX,
            output_path=this_output_path(),
            run_id=run_id,
            resources=versioned(ResourceConfig.with_tpu("v5p-8")),
            steps=versioned(num_steps),
            batch_size=versioned(batch_size),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(
                entity="marin-community",
                project="marin_moe",
                tags=["moe", "hrm_repro", f"d{hidden_dim}"],
                group="hrm-repro",
                name=None,
            ),
            optimizer=versioned(optimizer),
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
                    steps_per_eval=1000,
                    max_eval_batches=8,
                    eval_current=True,
                    eval_ema=False,
                )
            ),
        ),
    )

    # Post-training HRM-Text Table eval. Loads the final checkpoint at
    # ``<train_output>/checkpoints/step-<num_steps>``. The dependency is
    # expressed via ``train_step.cd(...)`` so the executor only schedules the
    # eval after the train step succeeds.
    eval_step = hrm_eval_step(
        name=run_id,
        model=model,
        checkpoint_path=train_step.cd(f"checkpoints/step-{num_steps}"),
        tokenizer_name=llama3_tokenizer,
        tasks=HRM_TEXT_BENCHMARKS_DEFAULT,
    )

    return train_step, eval_step


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key, "")
    return float(raw) if raw else default


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key, "")
    return int(raw) if raw else default


HIDDEN_DIM: int = _env_int("HRM_REPRO_HIDDEN_DIM", 768)
TARGET_TOKENS: float = _env_float("HRM_REPRO_TARGET_TOKENS", HRM_XL_TARGET_TOKENS)
RUN_SUFFIX: str = os.environ.get("HRM_REPRO_RUN_SUFFIX", "v1")


hrm_repro_step, hrm_eval_post = _build_hrm_repro_step(
    hidden_dim=HIDDEN_DIM,
    target_tokens=TARGET_TOKENS,
    run_suffix=RUN_SUFFIX,
)


# Standalone entry to materialize just the data pipeline (clean → tokenize) for
# the full HRM-Text mix. Useful for warming the cache before training.
_HRM_DATA_ONLY = os.environ.get("HRM_REPRO_DATA_ONLY", "").strip().lower() in ("1", "true", "yes")


if __name__ == "__main__":
    if _HRM_DATA_ONLY:
        executor_main(
            steps=[*hrm_text_clean_steps(), *hrm_text_tokenize_steps()],
            description="HRM-Text data port: per-source cleaners + tokenize.",
        )
    else:
        executor_main(
            steps=[hrm_repro_step, hrm_eval_post],
            description=(
                "HRM-Text reproduction with marin MoE: "
                f"d{HIDDEN_DIM}, target_tokens={TARGET_TOKENS:.2e}, "
                f"full HRM-Text data mix + post-training HRM-Text Table eval."
            ),
        )
