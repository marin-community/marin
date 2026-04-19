# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Re-judge Bloom-format inference artifacts with GPT-oss-120B in ONE session.

Replaces the expensive GPT-4.1 API judge with GPT-oss-120B served locally on
TPU via vLLM. Uses the ``run_batch_eval_judge`` path, which loads the judge
model **once** and then sequentially judges every selected target within the
same ``BatchedVllmServeSession`` — avoiding ~10 min of model-load + XLA
recompilation per target.

Resumable: targets whose ``{output}/{label}/summary.json`` already exists are
skipped on restart, so preemption only costs rejudging the in-flight target
plus any unstarted ones.

Caching note: this script produces a SINGLE ExecutorStep. Its hash depends on
the set of selected targets, so running with a different ``--target`` subset
produces a different step and re-judges everything. For incremental sweeps,
rerun with the same ``--target`` set — the resume logic will skip completed
targets inside that step.

Usage:

    # v6e-8 smoke in us-east5 (same-region inference + model):
    uv run iris --config lib/iris/examples/marin.yaml job run \
        --no-wait \
        --job-name judge-goss120b-v6e8-smoke \
        --cpu 4 --memory 16GB --disk 10GB \
        --region us-east5 \
        -e MARIN_PREFIX gs://marin-us-east5 \
        -- python experiments/posttrain/judge_all_goss120b.py \
            --tpu-type v6e-8 \
            --target beta001_lr5e7_seed0

    # All 8 targets on v6e-8, one session, resumable:
    uv run iris --config lib/iris/examples/marin.yaml job run \
        --no-wait \
        --job-name judge-goss120b-all \
        --cpu 4 --memory 16GB --disk 10GB \
        -- python experiments/posttrain/judge_all_goss120b.py \
            --tpu-type v6e-8
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from dataclasses import dataclass
from pathlib import Path

from gpt_oss_tpu import gpt_oss_120b_tpu_vllm_config

from marin.alignment.evaluate import BatchEvalJudgeConfig, EvalJudgeTarget, run_batch_eval_judge
from marin.alignment.inference_config import VLLMConfig
from marin.execution.executor import ExecutorStep, executor_main, mirrored, this_output_path
from marin.execution.remote import remote

SPEC_PATH = str(Path(__file__).parent / "specs" / "openai_model_spec.jsonl")

# Match GPT-4.1 judge runs for fair comparison.
JUDGE_MAX_TOKENS = 4000
JUDGE_BATCH_SIZE = 256

# TPU presets: (tpu_type, tensor_parallel_size, ram)
# v6e-8: 8 chips x 32 GB HBM = 256 GB. 120B BF16 ~240 GB weights, tight fit.
# v5p-8: 4 chips x 95 GB HBM = 380 GB. Comfortable but fewer zones.
TPU_PRESETS: dict[str, tuple[int, str]] = {
    "v6e-8": (8, "256g"),
    "v5p-8": (4, "400g"),
}
DEFAULT_TPU_TYPE = "v6e-8"


@dataclass(frozen=True)
class JudgeTarget:
    """An existing inference artifact to re-judge."""

    label: str
    inference_path: str


# ---------------------------------------------------------------------------
# Group A: Seed-0 full-DPO sweep (batch=128)
# ---------------------------------------------------------------------------
GROUP_A_TARGETS = [
    JudgeTarget(
        label="sft",
        inference_path="gs://marin-us-east1/eval/marin_8b_instruct_bloom_speceval/inference-89612d",
    ),
    JudgeTarget(
        label="beta001_lr5e7_seed0",
        inference_path="gs://marin-us-east5/eval/marin_dpo_beta001_lr5e7_seed0_bloom_speceval/inference-aaf42f",
    ),
    JudgeTarget(
        label="beta001_lr75e7_seed0",
        inference_path="gs://marin-us-east5/eval/marin_dpo_beta001_lr75e7_seed0_bloom_speceval/inference-d2c220",
    ),
    JudgeTarget(
        label="beta01_lr5e7_seed0",
        inference_path="gs://marin-us-east5/eval/marin_dpo_beta01_lr5e7_seed0_bloom_speceval/inference-a8afc8",
    ),
    JudgeTarget(
        label="beta01_lr75e7_seed0",
        inference_path="gs://marin-us-central1/eval/marin_dpo_beta01_lr75e7_seed0_bloom_speceval/inference-190643",
    ),
]

# ---------------------------------------------------------------------------
# Group B: Matched batch-64 LoRA vs full-DPO comparison
# ---------------------------------------------------------------------------
GROUP_B_TARGETS = [
    JudgeTarget(
        label="full_dpo_beta01_b64_step1699",
        inference_path=(
            "gs://marin-eu-west4/eval/"
            "marin_dpo_compare_lora_beta01_seed0_b64_step1699_bloom_speceval_seed0fullb64euw4r1/"
            "inference-1179e2"
        ),
    ),
    JudgeTarget(
        label="lora_lr5e6_b64_step1699",
        inference_path=(
            "gs://marin-eu-west4/eval/"
            "marin_dpo_tune_lora_lr5e6_seed0_step1699_bloom_speceval_seed0paireuw4r2/"
            "inference-abdde9"
        ),
    ),
    JudgeTarget(
        label="lora_lr1e5_b64_step1699",
        inference_path=(
            "gs://marin-us-central1/eval/"
            "marin_dpo_tune_lora_lr1e5_seed0_step1699_bloom_speceval_cleanreexportr2/"
            "inference-ee9768"
        ),
    ),
]

ALL_TARGETS = {t.label: t for t in GROUP_A_TARGETS + GROUP_B_TARGETS}


def _to_mirror_relative(gs_path: str) -> str:
    """Strip the marin regional bucket prefix so the path can be looked up across regions.

    MirrorFileSystem treats paths as namespace-relative across all marin regional
    buckets, so we drop the ``gs://marin-<region>/`` portion and let the mirror
    layer resolve which bucket actually has the data.

    Example: ``gs://marin-us-east5/eval/foo/bar`` -> ``eval/foo/bar``
    """
    if not gs_path.startswith("gs://marin-"):
        raise ValueError(f"Expected a gs://marin-* URL for mirroring, got {gs_path!r}")
    without_scheme = gs_path[len("gs://") :]
    parts = without_scheme.split("/", 1)
    if len(parts) < 2 or not parts[1]:
        raise ValueError(f"Path {gs_path!r} has no component after the bucket name")
    return parts[1]


def _build_judge_config(tpu_type: str) -> VLLMConfig:
    """GPT-oss-120B judge serving config for the given TPU type."""
    if tpu_type not in TPU_PRESETS:
        raise ValueError(f"Unknown tpu_type {tpu_type!r}. Supported: {sorted(TPU_PRESETS)}")
    tp, ram = TPU_PRESETS[tpu_type]
    base = gpt_oss_120b_tpu_vllm_config(
        tpu_type=tpu_type,
        tensor_parallel_size=tp,
        max_model_len=8192,
        ram=ram,
        model_impl_type="vllm",
        prefer_jax_for_bootstrap=False,
    )
    # Tee vLLM native stderr into Iris logs so we can see model loading
    # progress and XLA compilation status.
    return dataclasses.replace(base, native_stderr_mode="tee")


def build_batch_judge_step(targets: list[JudgeTarget], tpu_type: str) -> ExecutorStep:
    """Build a single ExecutorStep that judges every target in one vLLM session.

    Each target's results land under ``{step_output}/{label}/``. The batched
    runner loads the judge model once, then sequentially judges each target,
    skipping any whose ``summary.json`` already exists (resume after preempt).

    Inference artifacts are wrapped with :func:`mirrored` so the executor can
    pull them from whichever regional bucket has them into the job's local
    bucket at run time. This lets us submit the job in any region that has
    the judge model staged, regardless of where the inference artifacts live.
    """
    judge_vllm = _build_judge_config(tpu_type)
    eval_targets = [
        EvalJudgeTarget(
            label=t.label,
            eval_responses_path=mirrored(_to_mirror_relative(t.inference_path), budget_gb=1),
            output_path=this_output_path(t.label),
        )
        for t in targets
    ]
    return ExecutorStep(
        name="eval/judge_goss120b_batch",
        description=f"GPT-oss-120B batch judge on {tpu_type} ({len(targets)} targets)",
        fn=remote(
            run_batch_eval_judge,
            resources=judge_vllm.resources,
            pip_dependency_groups=judge_vllm.pip_dependency_groups,
            pip_packages=judge_vllm.pip_packages,
        ),
        config=BatchEvalJudgeConfig(
            targets=eval_targets,
            spec_path=SPEC_PATH,
            judge_model=judge_vllm,
            batch_size=JUDGE_BATCH_SIZE,
            judge_max_tokens=JUDGE_MAX_TOKENS,
        ),
    )


def parse_args() -> tuple[list[JudgeTarget], str, list[str]]:
    parser = argparse.ArgumentParser(description="Re-judge inference artifacts with GPT-oss-120B.")
    parser.add_argument(
        "--target",
        dest="targets",
        action="append",
        choices=sorted(ALL_TARGETS),
        help="Target label to judge. Repeat to select a subset. Defaults to all.",
    )
    parser.add_argument(
        "--tpu-type",
        default=DEFAULT_TPU_TYPE,
        choices=sorted(TPU_PRESETS),
        help=f"TPU type for judge workers. Defaults to {DEFAULT_TPU_TYPE}.",
    )
    args, remaining = parser.parse_known_args()
    selected = [ALL_TARGETS[k] for k in args.targets] if args.targets else list(ALL_TARGETS.values())
    return selected, args.tpu_type, remaining


if __name__ == "__main__":
    selected, tpu_type, remaining = parse_args()
    sys.argv = [sys.argv[0], *remaining]
    labels = ", ".join(t.label for t in selected)
    executor_main(
        steps=[build_batch_judge_step(selected, tpu_type)],
        description=f"GPT-oss-120B batch judge on {tpu_type} for: {labels}",
    )
