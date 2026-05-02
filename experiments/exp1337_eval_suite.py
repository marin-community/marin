# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Downstream evals for the Delphi scaling-suite blog post.

Generates the data behind the ``mmlu-emergence`` figure in the Delphi blog
(``content/blog/delphi.md``), which uses a two-step regression to forecast
hard-metric task scores (MMLU accuracy, HumanEval pass@1, GSM8K exact-match):

  1. **Soft-metric scaling law** (left panel) — fit per-choice log-prob (MMLU)
     or bits-per-byte of the reference completion (HumanEval, GSM8K) as a
     power law in compute over the Delphi IsoFLOP optima, then extrapolate to
     the held-out 1e21 / 1e22 / 1e23 runs. The soft metric stays informative
     at small compute budgets where the hard metric is pinned at chance.
  2. **Observational projection** (right panel) — fit a sigmoid from soft
     metric to hard metric on a pool of public open-weights models. The fit
     is a property of the task, not of any one training recipe, so it
     transfers between the external pool and Delphi.

For both halves to plot on the same axes, every model needs the soft metric
(logprob / bpb) AND the hard metric (accuracy / pass@1 / exact-match) for
every task. Two model groups, one set of tasks:

  * Delphi / Completed AdamH ladder — 7 IsoFLOP sweep winners (3e18 to 3e20)
    plus the 3 compute-optimal target-budget runs (1e21 / 1e22 / 1e23).
    Referenced by MARIN_PREFIX-relative paths via ``InputName.hardcoded``;
    ``discover_latest_checkpoint=True`` picks the latest step-N under /hf.
    These supply the IsoFLOP optima and held-out points in the left panel.
  * External open-weights pool — Qwen3-Base (0.6B-14B), Llama-2 (7B/13B),
    Llama-3.1-8B, Llama-3.2 (1B/3B), OLMo-2 (7B/13B), Marin-8B-Base. Pulled
    via ``experiments/models.py``; ``discover_latest_checkpoint=False``
    (single HF snapshot). These supply the points the sigmoid is fit on in
    the right panel — they need MMLU logprobs too, so the soft→hard mapping
    has paired observations across the whole task list.

Three task families, routed by output type:

  * Generation (vLLM harness, ``evaluate_lm_evaluation_harness``):
      ``gsm8k`` 5-shot, ``humaneval`` 10-shot — the hard-metric side
      (exact-match, pass@1) of GSM8K and HumanEval.
  * Logprob (TPU-native Levanter harness, ``evaluate_levanter_lm_evaluation_harness``):
      ``mmlu_sl_verb`` 0/5-shot — supplies both the soft (per-choice log-prob)
      and hard (accuracy) metrics for MMLU. ``logprob_gsm8k`` 5-shot and
      ``logprob_humaneval`` 10-shot — the soft (bpb / nll on the reference
      completion) side of the generative tasks.

The logprob GSM8K / HumanEval tasks are inlined under fresh names so lm-eval
builds the Entry straight from the dict instead of applying registered-task
override semantics, which currently drop ``dataset_path`` and crash with
``TypeError: expected str, bytes or os.PathLike object, not NoneType`` in
``datasets.load_dataset`` when ``num_fewshot`` is overridden on the
registered ``logprob_gsm8k`` / ``logprob_humaneval``.

Launch via iris:

    uv run iris --cluster=marin job run --no-wait \\
        --zone us-east5-a \\
        --memory 8GB --enable-extra-resources \\
        -e WANDB_API_KEY "$WANDB_API_KEY" \\
        -e HF_TOKEN "$(cat ~/.cache/huggingface/token)" \\
        -- python experiments/exp1337_eval_suite.py
"""

from fray.cluster import ResourceConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import ExecutorStep, InputName, executor_main

from experiments.evals.evals import (
    evaluate_levanter_lm_evaluation_harness,
    evaluate_lm_evaluation_harness,
    extract_model_name_and_path,
)
from experiments.models import (
    llama_2_7b,
    llama_2_13b,
    llama_3_1_8b,
    llama_3_2_1b,
    llama_3_2_3b,
    marin_8b_base,
    olmo_2_base_8b,
    olmo_2_base_13b,
    qwen3_0_6b_base,
    qwen3_1_7b_base,
    qwen3_4b_base,
    qwen3_8b_base,
    qwen3_14b_base,
)

# Env vars plumbed through to each iris child worker running ``vllm serve``.
# The coordinator's ``os.environ`` does NOT propagate to iris-spawned children;
# they must be threaded through ``remote(env_vars=...)`` at step-construction time.
#
#   * MARIN_VLLM_MODE=native           — skip docker-mode (no /var/run/docker.sock on iris).
#   * VLLM_ENABLE_V1_MULTIPROCESSING=0 — keep APIServer + EngineCore in one process so the
#     v5p TPU stays claimed by a single process; otherwise the spawned EngineCore child can't
#     re-open libtpu and JAX falls back to CPU, surfacing as ``AttributeError`` on
#     ``device.coords`` inside ``tpu_inference.utils.make_optimized_mesh``. (RL's rollout
#     worker sets the same flag at module import — see
#     ``lib/marin/src/marin/rl/environments/inference_ctx/vllm.py:40``.)
#   * VLLM_ALLOW_LONG_MAX_MODEL_LEN, VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION,
#     VLLM_TPU_SKIP_PRECOMPILE — match Harbor's working recipe
#     (lib/marin/src/marin/evaluation/evaluators/harbor_evaluator.py:236-238).
#   * HF_ALLOW_CODE_EVAL=1 — HF ``evaluate``'s code_eval refuses to run generated code without
#     it; HumanEval pass@1 needs it. Inert for the GSM8K exact_match task.
VLLM_GENERATION_ENV_VARS: dict[str, str] = {
    "MARIN_VLLM_MODE": "native",
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
    "VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION": "1",
    "VLLM_TPU_SKIP_PRECOMPILE": "1",
    "HF_ALLOW_CODE_EVAL": "1",
}

RESOURCE_CONFIG = ResourceConfig.with_tpu("v5p-8")

# Halve the attention-kernel scratch so the larger models (e.g. 1e+23 with 42 heads,
# Qwen3-14B with 40 heads) fit VMEM. XLA:TPU attention scratch =
# 2 * chunk_len * num_heads * 2 * head_dim * 2 bytes; at chunk_len=2048 a 42-head
# model needs 88 MB but VMEM is 64 MB — dropping chunk_len to 1024 gives 44 MB.
GENERATION_ENGINE_KWARGS: dict[str, int] = {"max_num_batched_tokens": 1024}

GENERATION_TASKS: tuple[EvalTaskConfig, ...] = (
    EvalTaskConfig("gsm8k", 5, task_alias="gsm8k_5shot"),
    EvalTaskConfig("humaneval", 10, task_alias="humaneval_10shot"),
)

LOGPROB_TASKS: tuple[EvalTaskConfig, ...] = (
    EvalTaskConfig("mmlu_sl_verb", 0, task_alias="mmlu_sl_verb_0shot"),
    EvalTaskConfig("mmlu_sl_verb", 5, task_alias="mmlu_sl_verb_5shot"),
    EvalTaskConfig(
        name="logprob_gsm8k_5shot",
        num_fewshot=5,
        task_alias="logprob_gsm8k_5shot",
        task_kwargs={
            "tag": ["logprob_generative", "math_word_problems"],
            "dataset_path": "openai/gsm8k",
            "dataset_name": "main",
            "output_type": "loglikelihood",
            "training_split": "train",
            "fewshot_split": "train",
            "test_split": "test",
            "doc_to_text": "Question: {{question}}\nAnswer:",
            "doc_to_target": " {{answer}}",
            "metric_list": [
                {"metric": "bpb", "aggregation": "mean", "higher_is_better": False},
                {"metric": "nll", "aggregation": "mean", "higher_is_better": False},
            ],
            "metadata": {"version": 1.0},
        },
    ),
    EvalTaskConfig(
        name="logprob_humaneval_10shot",
        num_fewshot=10,
        task_alias="logprob_humaneval_10shot",
        task_kwargs={
            "tag": ["logprob_generative", "code"],
            "dataset_path": "openai/openai_humaneval",
            "output_type": "loglikelihood",
            "test_split": "test",
            "doc_to_text": "{{prompt}}",
            "doc_to_target": "{{canonical_solution}}",
            "metric_list": [
                {"metric": "bpb", "aggregation": "mean", "higher_is_better": False},
                {"metric": "nll", "aggregation": "mean", "higher_is_better": False},
            ],
            "metadata": {"version": 1.0},
        },
    ),
)

# MARIN_PREFIX-relative paths to each Delphi model's /hf export. The executor
# resolves ``InputName.hardcoded`` to ``{MARIN_PREFIX}/{path}`` at runtime.
DELPHI_SWEEP_WINNERS: dict[str, str] = {
    "3e+18": "checkpoints/isoflop/isoflop-3e+18-d1024-L11-B8-adamh_scaling_v6/hf",
    "9e+18": "checkpoints/isoflop/isoflop-9e+18-d1152-L12-B16-adamh_scaling_v6/hf",
    "2e+19": "checkpoints/isoflop/isoflop-2e+19-d1408-L15-B16-adamh_scaling_v6/hf",
    "3e+19": "checkpoints/isoflop/isoflop-3e+19-d1536-L16-B32-adamh_scaling_v6/hf",
    "9e+19": "checkpoints/isoflop/isoflop-9e+19-d1792-L18-B64-adamh_scaling_v6/hf",
    "2e+20": "checkpoints/isoflop/isoflop-2e+20-d2048-L21-B64-adamh_scaling_v6/hf",
    "3e+20": "checkpoints/isoflop/isoflop-3e+20-d2304-L23-B128-adamh_scaling_v6/hf",
}

DELPHI_OPTIMAL_RUNS: dict[str, str] = {
    "1e+21": "adamh-scaling-ladder-nemotron-optimal-1e+21-v5-019021/hf",
    "1e+22": "adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e/hf",
    "1e+23": "adamh-scaling-ladder-nemotron-optimal-1e+23-v5-27f2fb/hf",
}

# Qwen3-32B-Base does not exist on HF — Qwen only shipped 32B Instruct.
HF_MODEL_STEPS: tuple[ExecutorStep, ...] = (
    qwen3_0_6b_base,
    qwen3_1_7b_base,
    qwen3_4b_base,
    qwen3_8b_base,
    qwen3_14b_base,
    llama_2_7b,
    llama_2_13b,
    llama_3_1_8b,
    llama_3_2_1b,
    llama_3_2_3b,
    olmo_2_base_8b,
    olmo_2_base_13b,
    marin_8b_base,
)


def _eval_steps_for_model(
    model_slug: str,
    model_path,
    *,
    discover_latest_checkpoint: bool,
) -> list[ExecutorStep]:
    steps: list[ExecutorStep] = []
    for task in GENERATION_TASKS:
        steps.append(
            evaluate_lm_evaluation_harness(
                model_name=f"{model_slug}_{task.task_alias or task.name}",
                model_path=model_path,
                evals=[task],
                resource_config=RESOURCE_CONFIG,
                discover_latest_checkpoint=discover_latest_checkpoint,
                env_vars=VLLM_GENERATION_ENV_VARS,
                engine_kwargs=GENERATION_ENGINE_KWARGS,
            )
        )
    for task in LOGPROB_TASKS:
        steps.append(
            evaluate_levanter_lm_evaluation_harness(
                model_name=f"{model_slug}_{task.task_alias or task.name}",
                model_path=model_path,
                evals=[task],
                resource_config=RESOURCE_CONFIG,
                discover_latest_checkpoint=discover_latest_checkpoint,
            )
        )
    return steps


def build_eval_steps() -> list[ExecutorStep]:
    eval_steps: list[ExecutorStep] = []

    for rel_path in (*DELPHI_SWEEP_WINNERS.values(), *DELPHI_OPTIMAL_RUNS.values()):
        # "checkpoints/isoflop/isoflop-3e+18-.../hf" -> "isoflop-3e+18-..."
        # "adamh-scaling-ladder-nemotron-optimal-1e+21-v5-019021/hf" -> same (sans /hf)
        slug = rel_path.removesuffix("/hf").rsplit("/", 1)[-1]
        eval_steps.extend(
            _eval_steps_for_model(
                model_slug=slug,
                model_path=InputName.hardcoded(rel_path),
                discover_latest_checkpoint=True,
            )
        )

    for model_step in HF_MODEL_STEPS:
        model_name, model_path = extract_model_name_and_path(model_step)
        # ExecutorStep.name is "models/<repo-slashes-as-dashes>--<rev>";
        # keep the repo slug, drop the trailing revision.
        slug = model_name.split("/")[-1].rsplit("--", 1)[0]
        eval_steps.extend(
            _eval_steps_for_model(
                model_slug=slug,
                model_path=model_path,
                discover_latest_checkpoint=False,
            )
        )

    return eval_steps


if __name__ == "__main__":
    executor_main(steps=build_eval_steps())
