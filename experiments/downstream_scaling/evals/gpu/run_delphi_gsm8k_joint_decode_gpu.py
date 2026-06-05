"""Run joint-decode-gpu GSM8K evals against the Delphi ladder.

This runner reuses the existing ``GSM8KTask`` for prompts and grading and plugs
joint-decode-gpu into the ``CompletionAlgorithm`` slot. The completions step is
an in-process ``ExecutorStep`` (no remote dispatch) because it runs on the same
GPU box that drives the executor. Outputs land under the prefix from ``--prefix``
if given, otherwise from the ``MARIN_PREFIX`` env var, otherwise the GCE region
bucket (``gs://marin-{region}``) when running on GCP, otherwise ``/tmp/marin``.

Install the required packages into the project venv before running::

    uv pip install --python .venv/bin/python 'joint-decode-gpu @ git+https://github.com/RohithKuditipudi/joint-decode-gpu'
    uv pip install --python .venv/bin/python 'lm-eval[math,api]@git+https://github.com/stanford-crfm/lm-evaluation-harness@d5e3391f22cde186c827674d5c3ec7c5f4fe0cab'
"""

from __future__ import annotations

import functools
import json
import random
from dataclasses import dataclass
from typing import Any

import fsspec

from marin.execution.executor import ExecutorStep, InputName, MirroredValue, executor_main
from marin.execution.types import this_output_path, versioned

from experiments.downstream_scaling.evals.framework.core import make_eval_step
from experiments.downstream_scaling.evals.framework.schema import completions_file, read_prompt_rows
from experiments.downstream_scaling.evals.tasks.gsm8k import GSM8KTask, GSM8KTaskConfig
from experiments.downstream_scaling.evals.utils import version_path
from experiments.downstream_scaling.models.delphi import DELPHI_HF_REPOS

from joint_decode_gpu.aggregation import select_avg_logits
from joint_decode_gpu.config import JointDecodeConfig, JointDecodeModelConfig, JointDecodeSamplingConfig
from joint_decode_gpu.coordinator import run_joint_decode

N_SAMPLES = 32
N_PROBLEMS = 256
NUM_FEWSHOT = 5
FEWSHOT_SEED = 1234
MAX_TOKENS = 512
SEED = 42
STOP_TOKENS: tuple[str, ...] = ("Question:", "</s>", "<|im_end|>")
TEMPERATURE = 0.4
TOP_K_A = 16
TOP_K_B = 16
MICROBATCH_SIZE = 8
BARRIER_TIMEOUT_S = 600.0

DELPHI_SLUGS = ["1e22"]
ADVISOR_MODEL = "meta-llama/Llama-3.1-8B"
ADVISOR_WEIGHTS = [float(i/10.0) for i in range(11)]


def select_token_proto(
    a_topk: list[dict[str, Any]],
    b_topk: list[dict[str, Any]],
    *,
    advisor_weight: float,
    temperature: float,
    rng: random.Random,
) -> int:
    return select_avg_logits(a_topk, b_topk, advisor_weight=advisor_weight, temperature=temperature, rng=rng)


@dataclass(frozen=True)
class JointDecodeGpuCompletionStepConfig:
    output_path: str
    prompts_path: str
    decoder_model_path: str
    advisor_model_path: str
    sampling: JointDecodeSamplingConfig
    n_samples: int
    advisor_weight: float
    temperature: float
    max_model_len: int
    gpu_memory_utilization: float
    enable_prefix_caching: bool
    enforce_eager: bool


def run_joint_decode_gpu_completions(config: JointDecodeGpuCompletionStepConfig) -> None:
    rows = list(read_prompt_rows(config.prompts_path))
    flat = [(row, sample_index) for row in rows for sample_index in range(config.n_samples)]
    prompts = [row["prompt"] for row, _ in flat]

    jd_config = JointDecodeConfig(
        model_a=JointDecodeModelConfig(
            model_path=config.decoder_model_path,
            gpu_index=0,
            max_model_len=config.max_model_len,
            gpu_memory_utilization=config.gpu_memory_utilization,
            enable_prefix_caching=config.enable_prefix_caching,
            enforce_eager=config.enforce_eager,
        ),
        model_b=JointDecodeModelConfig(
            model_path=config.advisor_model_path,
            gpu_index=1,
            max_model_len=config.max_model_len,
            gpu_memory_utilization=config.gpu_memory_utilization,
            enable_prefix_caching=config.enable_prefix_caching,
            enforce_eager=config.enforce_eager,
        ),
        sampling=config.sampling,
    )

    select_token = functools.partial(select_token_proto, advisor_weight=config.advisor_weight, temperature=config.temperature)
    outputs = run_joint_decode(jd_config, prompts, prompts, select_token=select_token)

    by_id: dict[str, list[dict[str, Any]]] = {}
    for (row, sample_index), output in zip(flat, outputs, strict=True):
        by_id.setdefault(row["id"], []).append(
            {
                "text": output.text,
                "metadata": {"sample_index": sample_index, "finish_reason": output.finish_reason},
            }
        )

    with fsspec.open(completions_file(config.output_path), "wt", compression="gzip") as f:
        for row in rows:
            f.write(json.dumps({"id": row["id"], "completions": by_id[row["id"]]}) + "\n")


@dataclass(frozen=True)
class JointDecodeGpuCompletionAlgorithm:
    advisor_model_path: str | InputName | MirroredValue
    sampling: JointDecodeSamplingConfig
    n_samples: int
    advisor_weight: float
    temperature: float
    max_model_len: int = 2048
    gpu_memory_utilization: float = 0.9
    enable_prefix_caching: bool = False
    enforce_eager: bool = True

    def make_completions_step(
        self,
        *,
        name: str,
        model_path: str | InputName | MirroredValue,
        prompts_path: str | InputName | MirroredValue,
    ) -> ExecutorStep:
        return ExecutorStep(
            name=name,
            fn=run_joint_decode_gpu_completions,
            config=JointDecodeGpuCompletionStepConfig(
                output_path=this_output_path(),
                prompts_path=version_path(prompts_path),  # type: ignore[arg-type]
                decoder_model_path=version_path(model_path),  # type: ignore[arg-type]
                advisor_model_path=version_path(self.advisor_model_path),  # type: ignore[arg-type]
                sampling=versioned(self.sampling),  # type: ignore[arg-type]
                n_samples=versioned(self.n_samples),  # type: ignore[arg-type]
                advisor_weight=versioned(self.advisor_weight),  # type: ignore[arg-type]
                temperature=versioned(self.temperature),  # type: ignore[arg-type]
                max_model_len=versioned(self.max_model_len),  # type: ignore[arg-type]
                gpu_memory_utilization=versioned(self.gpu_memory_utilization),  # type: ignore[arg-type]
                enable_prefix_caching=versioned(self.enable_prefix_caching),  # type: ignore[arg-type]
                enforce_eager=versioned(self.enforce_eager),  # type: ignore[arg-type]
            ),
        )


def main() -> None:
    task = GSM8KTask(
        config=GSM8KTaskConfig(
            num_fewshot=NUM_FEWSHOT,
            fewshot_seed=FEWSHOT_SEED,
            n_problems=N_PROBLEMS,
        )
    )

    sampling = JointDecodeSamplingConfig(
        max_tokens=MAX_TOKENS,
        top_k_a=TOP_K_A,
        top_k_b=TOP_K_B,
        microbatch_size=MICROBATCH_SIZE,
        barrier_timeout_s=BARRIER_TIMEOUT_S,
        seed=SEED,
        stop=STOP_TOKENS,
    )

    steps = [
        make_eval_step(
            name=(
                f"downstream_scaling/evals/gpu/delphi/gsm8k/joint_decode/"
                f"advisor_weight{round(advisor_weight * 100):03d}/{slug}"
            ),
            model_path=DELPHI_HF_REPOS[slug],
            task=task,
            alg=JointDecodeGpuCompletionAlgorithm(
                advisor_model_path=ADVISOR_MODEL,
                sampling=sampling,
                n_samples=N_SAMPLES,
                advisor_weight=advisor_weight,
                temperature=TEMPERATURE,
            ),
        )
        for slug in DELPHI_SLUGS
        for advisor_weight in ADVISOR_WEIGHTS
    ]

    executor_main(
        steps=steps,
        max_concurrent=1,
        description="Delphi joint-decode (GPU) evals on GSM8K.",
    )


if __name__ == "__main__":
    main()
