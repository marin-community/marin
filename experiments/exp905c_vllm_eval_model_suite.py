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

"""Run vLLM evals for a suite of base models."""

import argparse
import logging
from typing import Iterable, Sequence

from experiments.evals.evals import evaluate_lm_evaluation_harness
from experiments.evals.resource_configs import ResourceConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import ExecutorStep, OutputName, executor_main

DEFAULT_MAX_MODEL_LEN = 4096
BASE_MODEL_DEFAULTS = {
    "apply_chat_template": False,
    "tensor_parallel_size": 8,
    "max_model_len": DEFAULT_MAX_MODEL_LEN,
}

resource_config = ResourceConfig(num_tpu=8, tpu_type="TPU-v6e-8", strategy="STRICT_PACK")

EVAL_TASKS = [
    EvalTaskConfig("eq_bench", num_fewshot=0),
]

DEFAULT_GENERATION_PARAMS = {
    "temperature": 0.7,
    "top_p": 1.0,
    "do_sample": True,
    "n": 1,
    "seed": 42,
}

DEFAULT_ENGINE_KWARGS = {
    "max_model_len": 8192,
    "max_gen_toks": 2048,
}

MODEL_REGION = "marin-us-east1"
GCSFUSE_MODEL_ROOT = f"gs://{MODEL_REGION}/gcsfuse_mount/models"
MODEL_ROOT = GCSFUSE_MODEL_ROOT


def compile_results(eval_steps: Sequence[ExecutorStep], *, step_name: str) -> ExecutorStep:
    import json
    import pandas as pd
    import fsspec

    logger = logging.getLogger(__name__)

    def _compile_results_fn(config) -> None:
        all_results = []
        input_paths = config["input_paths"]
        output_path = config["output_path"]

        logger.info(f"Input paths: {input_paths}")

        if not input_paths:
            raise Exception("No input paths found!")

        for i, input_path in enumerate(input_paths):
            base_dir = input_path
            if base_dir.endswith("results.json"):
                base_dir = base_dir.rsplit("/", 1)[0]

            logger.info(f"Loading lm-eval samples from root {base_dir}")

            if base_dir.startswith("gs://"):
                gcs_root = base_dir
            else:
                gcs_root = "gs://" + base_dir.lstrip("/")

            pattern = gcs_root.rstrip("/") + "/*/__tmp__*/samples_*.jsonl"
            fs = fsspec.filesystem("gcs")
            sample_files: list[str] = fs.glob(pattern)

            if not sample_files:
                logger.warning(f"No samples_*.jsonl files found for input root {base_dir}")
                continue

            for sample_file in sample_files:
                logger.info(f"Reading samples from {sample_file}")
                path_parts = sample_file.split("/")

                if len(path_parts) >= 3:
                    task_dir = path_parts[-3]
                    if "_" in task_dir:
                        dataset_name = task_dir.rsplit("_", 1)[0]
                    else:
                        dataset_name = task_dir
                else:
                    dataset_name = "unknown_dataset"

                if len(path_parts) >= 4:
                    model_dir = path_parts[-4]
                elif len(path_parts) >= 2:
                    model_dir = path_parts[-2]
                else:
                    model_dir = "unknown_model"

                if "-" in model_dir:
                    model_name = model_dir.rsplit("-", 1)[0]
                else:
                    model_name = model_dir

                with fs.open(sample_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except Exception:
                            logger.warning(f"Failed to parse JSON line in {sample_file}: {line[:200]}")
                            continue

                        record["dataset_name"] = dataset_name
                        record["model_name"] = model_name
                        all_results.append(record)

        if not all_results:
            raise Exception("No results found in any of the provided steps")

        df = pd.DataFrame(all_results)

        results_file = f"{output_path}/compiled_results.json"
        with fsspec.open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)

        csv_file = f"{output_path}/compiled_results.csv"
        with fsspec.open(csv_file, "w") as f:
            df.to_csv(f, index=False)

        logger.info(f"Compiled results saved to: {results_file}")

    input_paths = [step.cd("results.json") for step in eval_steps]
    output_path = OutputName("compiled_results")

    return ExecutorStep(
        name=step_name,
        fn=_compile_results_fn,
        config={"input_paths": input_paths, "output_path": output_path},
        description="Compile results from multiple lm-eval steps into a single DataFrame",
    )


def build_eval_steps(
    model_config: dict,
    *,
    eval_tasks: Iterable[EvalTaskConfig] | None = None,
    download_steps: Sequence[ExecutorStep] | None = None,
    generation_params: dict | None = None,
    engine_kwargs_override: dict | None = None,
) -> list[ExecutorStep]:
    tasks = list(eval_tasks) if eval_tasks is not None else EVAL_TASKS

    generation = dict(DEFAULT_GENERATION_PARAMS)
    if generation_params:
        generation.update(generation_params)

    steps: list[ExecutorStep] = []
    if download_steps:
        steps.extend(download_steps)

    eval_steps: list[ExecutorStep] = []
    for task in tasks:
        engine_kwargs = {
            "tensor_parallel_size": model_config.get("tensor_parallel_size", 4),
            "max_model_len": model_config.get("max_model_len", DEFAULT_ENGINE_KWARGS["max_model_len"]),
            "max_gen_toks": model_config.get("max_gen_toks", DEFAULT_ENGINE_KWARGS["max_gen_toks"]),
        }
        if engine_kwargs_override:
            engine_kwargs.update(engine_kwargs_override)

        eval_steps.append(
            evaluate_lm_evaluation_harness(
                model_config["name"],
                model_config["path"],
                evals=[task],
                max_eval_instances=None,
                engine_kwargs=engine_kwargs,
                resource_config=resource_config,
                apply_chat_template=model_config.get("apply_chat_template", False),
                generation_params=generation,
            )
        )

    steps.extend(eval_steps)
    steps.append(compile_results(eval_steps, step_name=f"{model_config['name']}_compile_results"))
    return steps


def run_model_eval(
    model_config: dict,
    *,
    eval_tasks: Iterable[EvalTaskConfig] | None = None,
    download_steps: Sequence[ExecutorStep] | None = None,
    generation_params: dict | None = None,
    engine_kwargs_override: dict | None = None,
) -> None:
    logging.getLogger("ray").setLevel(logging.WARNING)

    steps = build_eval_steps(
        model_config,
        eval_tasks=eval_tasks,
        download_steps=download_steps,
        generation_params=generation_params,
        engine_kwargs_override=engine_kwargs_override,
    )
    executor_main(steps=steps)

MODEL_CONFIGS: dict[str, dict] = {
    "marin-32b-base": {
        "name": "marin-32b-base",
        "path": f"{GCSFUSE_MODEL_ROOT}/marin-community--marin-32b-base--main",
    },
    "OLMo-2-32B": {
        "name": "olmo-2-32b",
        "path": f"{GCSFUSE_MODEL_ROOT}/allenai--OLMo-2-0325-32B--stage2-ingredient2-step9000-tokens76B",
    },
    "Qwen2.5-32B": {
        "name": "qwen2.5-32b",
        "path": f"{GCSFUSE_MODEL_ROOT}/Qwen--Qwen2.5-32B--1818d35",
    },
    "Meta-Llama-3-70B": {
        "name": "meta-llama-3-70b",
        "path": f"{GCSFUSE_MODEL_ROOT}/meta-llama--Meta-Llama-3-70B--main",
    },
    "marin-8b-base": {
        "name": "marin-8b-base",
        "path": f"{GCSFUSE_MODEL_ROOT}/marin-community--marin-8b-base--0f1f658",
    },
    "Llama-3.1-8B": {
        "name": "llama-3.1-8b",
        "path": f"{GCSFUSE_MODEL_ROOT}/meta-llama--Llama-3-1-8B--d04e592",
    },
    "OLMo-2-1124-7B": {
        "name": "olmo-2-1124-7b",
        "path": f"{GCSFUSE_MODEL_ROOT}/allenai--OLMo-2-1124-7B--7df9a82",
    },
    "Qwen3-8B-Base": {
        "name": "qwen3-8b-base",
        "path": f"{GCSFUSE_MODEL_ROOT}/Qwen--Qwen3-8B-Base--49e3418",
    },
    "Olmo-3-1125-32B": {
        "name": "olmo-3-1125-32b",
        "path": f"{GCSFUSE_MODEL_ROOT}/allenai--Olmo-3-1125-32B--main",
    },
    "Olmo-3-1025-7B": {
        "name": "olmo-3-1025-7b",
        "path": f"{GCSFUSE_MODEL_ROOT}/allenai--Olmo-3-1025-7B--main",
    },
    # "gemma-3-27b-pt": { # not working for now
    #     "name": "gemma-3-27b-pt",
    #     "path": f"{GCSFUSE_MODEL_ROOT}/google--gemma-3-27b-pt--main",
    # },
}   


def _select_configs(requested: Iterable[str] | None) -> list[dict]:
    if not requested:
        return [{**BASE_MODEL_DEFAULTS, **config} for config in MODEL_CONFIGS.values()]

    configs: list[dict] = []
    for name in requested:
        config = MODEL_CONFIGS.get(name)
        if config is None:
            available = ", ".join(sorted(MODEL_CONFIGS))
            raise SystemExit(f"Unknown model '{name}'. Available: {available}")
        configs.append({**BASE_MODEL_DEFAULTS, **config})
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run vLLM evals for one or more base models.")
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Model name to run (can repeat). Defaults to running the full suite.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List configured models and exit.",
    )
    args = parser.parse_args()

    if args.list:
        for name in sorted(MODEL_CONFIGS):
            print(name)
        return

    for config in _select_configs(args.models):
        run_model_eval(config)


if __name__ == "__main__":
    main()
