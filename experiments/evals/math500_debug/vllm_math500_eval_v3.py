# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Evaluate a model on MATH-500 using vLLM in native mode.

Uses the same two-level dispatch as the Harbor evaluator:
- Outer ExecutorStep runs on CPU (no TPU conflicts)
- Inner Fray job runs on TPU with a clean environment for vLLM
"""

import logging
import os
from dataclasses import dataclass
from typing import Callable

from fray.cluster import Entrypoint, EnvironmentConfig, JobRequest, ResourceConfig, current_cluster
from marin.evaluation.evaluators.evaluator import ModelConfig
from marin.execution.executor import (
    ExecutorStep,
    InputName,
    executor_main,
    output_path_of,
    this_output_path,
)
from marin.inference.vllm_server import VLLM_NATIVE_PIP_PACKAGES

from zephyr import Backend, Dataset, load_jsonl
from datasets import load_dataset

from experiments.models import olmo_2_base_32b

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DownloadMath500Config:
    output_path: str


def download_math500(config: DownloadMath500Config):
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

    rows = [dict(row) for row in dataset]

    pipeline = (
        Dataset.from_list(rows)
        .reshard(1)
        .write_jsonl(f"{config.output_path}/math500-{{shard:05d}}.jsonl.gz")
    )

    Backend.execute(pipeline)


download_math500_step = ExecutorStep(
    name="raw/math500",
    fn=download_math500,
    config=DownloadMath500Config(
        output_path=this_output_path(),
    ),
)


PromptFormatter = Callable[[str], str]
PROMPT_FORMAT_REGISTRY: dict[str, PromptFormatter] = {}


def register_prompt_format(name: str):
    def decorator(fn: PromptFormatter) -> PromptFormatter:
        PROMPT_FORMAT_REGISTRY[name] = fn
        return fn
    return decorator


@register_prompt_format("question_only")
def format_question_only(problem: str) -> str:
    return problem


@register_prompt_format("question_with_boxed_suffix")
def format_question_with_boxed_suffix(problem: str) -> str:
    from marin.rl.environments.tinker_environments.math_env import MathEnv
    return problem + MathEnv.question_suffix()


@register_prompt_format("standard_fewshot")
def format_standard_fewshot(problem: str) -> str:
    from marin.rl.environments.tinker_environments.math_env import MathEnv
    fewshot = MathEnv.standard_fewshot_prefix()
    fewshot_prefix = fewshot[0]["content"] + "\n\n" + fewshot[1]["content"] + "\n\n"

    return fewshot_prefix + problem + MathEnv.question_suffix()


@dataclass(frozen=True)
class Math500EvalConfig:
    model_path: str
    output_path: str

    prompt_format: str = "question_only"

    max_tokens: int = 2048
    temperature: float = 0.7
    n_samples: int = 10

    math500_path: str | InputName = output_path_of(download_math500_step)


def _collect_env_vars() -> dict[str, str]:
    env_vars: dict[str, str] = {}
    for key in (
        "HF_TOKEN",
        "WANDB_API_KEY",
        "MARIN_PREFIX",
        "MARIN_VLLM_MODE",
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN",
        "VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION",
        "VLLM_TPU_SKIP_PRECOMPILE",
    ):
        value = os.environ.get(key)
        if value:
            env_vars[key] = value
    env_vars.setdefault("MARIN_VLLM_MODE", "native")
    env_vars.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
    env_vars.setdefault("VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION", "1")
    env_vars.setdefault("VLLM_TPU_SKIP_PRECOMPILE", "1")
    return env_vars


def run_math500_eval(config: Math500EvalConfig):
    """Runs on CPU. Loads data, formats prompts, dispatches vLLM eval to TPU via Fray."""

    dataset = Dataset.from_files(os.path.join(config.math500_path, "*.jsonl.gz")).flat_map(load_jsonl)
    rows = Backend.execute(dataset)

    problems = [row["problem"] for row in rows]
    answers = [row["answer"] for row in rows]

    prompt_formatter = PROMPT_FORMAT_REGISTRY[config.prompt_format]
    prompts = [prompt_formatter(problem) for problem in problems]

    model_path = config.model_path
    output_path = config.output_path
    max_tokens = config.max_tokens
    temperature = config.temperature
    n_samples = config.n_samples

    def _run_on_tpu():
        import json
        import logging
        import os

        import fsspec
        import openai
        from marin.evaluation.evaluators.evaluator import ModelConfig
        from marin.inference.vllm_server import VllmEnvironment
        from marin.rl.environments.tinker_environments.math_grading import extract_boxed, grade_answer
        from marin.utils import remove_tpu_lockfile_on_exit

        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)

        os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
        os.environ.setdefault("VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION", "1")
        os.environ.setdefault("VLLM_TPU_SKIP_PRECOMPILE", "1")

        with remove_tpu_lockfile_on_exit():
            model_config = ModelConfig(
                name=model_path,
                path=model_path,
                engine_kwargs={"load_format": "runai_streamer"},
            )

            with VllmEnvironment(model_config) as env:
                client = openai.OpenAI(base_url=env.server_url, api_key="unused")

                results = []
                pass_at_1, pass_at_k = 0, 0

                for i, prompt in enumerate(prompts):
                    ground_truth = answers[i]

                    response = client.completions.create(
                        model=env.model_id,
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        n=n_samples,
                    )

                    samples = []
                    total_correct = 0
                    for choice in response.choices:
                        generated = choice.text
                        try:
                            extracted = extract_boxed(generated)
                        except ValueError:
                            extracted = None

                        is_correct = False
                        if extracted is not None:
                            is_correct = grade_answer(extracted, ground_truth)

                        if is_correct:
                            total_correct += 1

                        samples.append({
                            "generated": generated,
                            "extracted_answer": extracted,
                            "correct": is_correct,
                        })

                    pass_at_1 += total_correct / n_samples
                    pass_at_k += (total_correct > 0)

                    results.append({
                        "problem": problems[i],
                        "ground_truth": ground_truth,
                        "samples": samples,
                    })

            avg_pass_at_1 = pass_at_1 / len(results)
            avg_pass_at_k = pass_at_k / len(results)
            print(f"Pass@1: {avg_pass_at_1:.4f}")
            print(f"Pass@{n_samples}: {avg_pass_at_k:.4f} ({pass_at_k}/{len(results)})")

            output = {
                "pass_at_1": avg_pass_at_1,
                "pass_at_k": avg_pass_at_k,
                "k": n_samples,
                "results": results,
            }

            with fsspec.open(os.path.join(output_path, "results.json.gz"), "wt", compression="gzip") as f:
                json.dump(output, f, indent=2)

    env_vars = _collect_env_vars()

    job_request = JobRequest(
        name="math500-vllm-eval",
        entrypoint=Entrypoint.from_callable(_run_on_tpu),
        resources=ResourceConfig.with_tpu("v5p-8"),
        environment=EnvironmentConfig.create(
            extras=["vllm", "math"],
            pip_packages=list(VLLM_NATIVE_PIP_PACKAGES),
            env_vars=env_vars,
        ),
    )

    cluster = current_cluster()
    job_id = cluster.launch(job_request)
    cluster.wait(job_id, raise_on_failure=True)


@dataclass(frozen=True)
class Math500ProcessConfig:
    eval_path: str
    output_path: str

    filter: str = "all"  # "correct", "incorrect", "all"


def process_math500_data(config: Math500ProcessConfig):
    with fsspec.open(os.path.join(config.eval_path, "results.json.gz"), "rt", compression="gzip") as f:
        output = json.load(f)

    with fsspec.open(os.path.join(config.output_path, "data.jsonl.gz"), "wt", compression="gzip") as f:
        i = 0
        for result in output["results"]:
            problem = result["problem"]
            samples = result["samples"]

            for sample in samples:
                if config.filter == "correct" and not sample["correct"]:
                    continue
                if config.filter == "incorrect" and sample["correct"]:
                    continue

                response = sample["generated"]

                messages = [
                    {"role": "user", "content": problem},
                    {"role": "assistant", "content": response}
                ]

                processed_row = {
                    "messages": messages,
                    "id": i,
                }
                i += 1

                f.write(json.dumps(processed_row) + "\n")


name = "olmo-stuff5-v3-hmm"
model_step = olmo_2_base_32b

eval_step = ExecutorStep(
    name=f"rohith_math500_eval/analysis/{name}",
    fn=run_math500_eval,
    config=Math500EvalConfig(
        model_path=output_path_of(model_step),
        output_path=this_output_path(),
    ),
    resources=ResourceConfig.with_cpu(),
    pip_dependency_groups=["math"],
)
process_step = ExecutorStep(
    name=f"rohith_math500_eval/documents/{name}",
    fn=process_math500_data,
    config=Math500ProcessConfig(
        eval_path=output_path_of(eval_step),
        output_path=this_output_path(),
    ),
)

if __name__ == "__main__":
    executor_main(steps=[eval_step, process_step], description="MATH-500 evaluation with vLLM (native mode).")
