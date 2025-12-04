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

import logging

from experiments.evals.evals import evaluate_lm_evaluation_harness
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import ExecutorStep, executor_main
from experiments.evals.resource_configs import ResourceConfig

resource_config = ResourceConfig(num_tpu=4, tpu_type="TPU-v6e-8", strategy="STRICT_PACK")

"""
Note for people trying to do evals:
- This script uses evaluate_lm_evaluation_harness which uses vLLM and lm-evaluation-harness
- The lm-evaluation-harness library automatically loads evaluation datasets
- Similar to exp905c_levanter_eval_model.py but uses vLLM instead of Levanter engine
- The structure follows exp905c_levanter_eval_model.py with EVAL_TASKS, MODELS, and compile_results

"""
EVAL_TASKS = [
    EvalTaskConfig("aime24", num_fewshot=0, task_alias="aime24_0shot"),
    # EvalTaskConfig("aime25", num_fewshot=0, task_alias="aime25_0shot"),
]

MODELS = [
    {
        "name": "qwen2.5-7b-instruct",
        "path": "gs://marin-us-central2/models/qwen2.5-7b-instruct",
        "apply_chat_template": True,
        "tensor_parallel_size": 4,
    }
]


def compile_results(steps: list[ExecutorStep]) -> ExecutorStep:
    """
    Takes in a list of ExecutorSteps for lm-eval tasks and compiles the results into a single DataFrame.

    Newer lm-eval runs (with EvaluationTracker + log_samples=True) write per-task outputs under
    paths like:

        gs://.../evaluation/lm_evaluation_harness/{model_name}-{hash}/{dataset}_0shot/__tmp__{model}/samples_{dataset}_TIMESTAMP.jsonl

    This helper scans each step's eval root for those `samples_*.jsonl` files, and aggregates the
    JSONL records into a flat DataFrame, annotating each row with `dataset_name` and `model_name`.
    """
    import json
    import logging
    import pandas as pd
    import fsspec
    from marin.execution.executor import InputName, OutputName

    logger = logging.getLogger(__name__)

    def _compile_results_fn(config) -> None:
        """Function that will be executed by the ExecutorStep to compile results."""
        all_results = []

        # Extract parameters from config
        input_paths = config["input_paths"]
        output_path = config["output_path"]

        logger.info(f"Input paths: {input_paths}")

        if not input_paths:
            raise Exception("No input paths found!")

        # Read results from each step's output path
        for i, input_path in enumerate(input_paths):
            # Normalise input_path to the evaluation root directory.
            # Older code passed .../{hash}/results.json; newer runs conceptually
            # treat .../{hash}/ as the root. We strip any trailing "results.json"
            # and then look for per-task sample files under that root.
            base_dir = input_path
            if base_dir.endswith("results.json"):
                base_dir = base_dir.rsplit("/", 1)[0]

            logger.info(f"Loading lm-eval samples from root {base_dir}")

            # Normalize to a GCS URL if the scheme was stripped by the executor packaging.
            # We assume eval outputs live in the "marin-us-east1" bucket when no scheme is present.
            if base_dir.startswith("gs://"):
                gcs_root = base_dir
            else:
                # Avoid accidental relative local paths like "marin-us-east1/..."
                gcs_root = "gs://" + base_dir.lstrip("/")

            # Pattern for per-task sample files, e.g.:
            # {root}/{dataset}_0shot/__tmp__{model}/samples_{dataset}_TIMESTAMP.jsonl
            pattern = gcs_root.rstrip("/") + "/*/__tmp__*/samples_*.jsonl"
            fs = fsspec.filesystem("gcs")
            sample_files: list[str] = fs.glob(pattern)

            if not sample_files:
                logger.warning(f"No samples_*.jsonl files found for input root {base_dir}")
                continue

            for sample_file in sample_files:
                logger.info(f"Reading samples from {sample_file}")
                path_parts = sample_file.split("/")

                # Infer dataset_name from the task directory: {dataset}_0shot
                # e.g. .../qwen-.../aime24_0shot/__tmp__.../samples_aime24_*.jsonl
                if len(path_parts) >= 3:
                    task_dir = path_parts[-3]
                    # Strip trailing "_{shot}" suffix (e.g. "_0shot", "_5shot")
                    if "_" in task_dir:
                        dataset_name = task_dir.rsplit("_", 1)[0]
                    else:
                        dataset_name = task_dir
                else:
                    dataset_name = "unknown_dataset"

                # Infer model_name from the model/hash directory
                # e.g. .../evaluation/lm_evaluation_harness/qwen2.5-7b-instruct-47227c/aime24_0shot/...
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

                # Read JSONL samples from GCS using the same filesystem
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

                        # Annotate with dataset and model metadata
                        record["dataset_name"] = dataset_name
                        record["model_name"] = model_name
                        all_results.append(record)

        if not all_results:
            raise Exception("No results found in any of the provided steps")

        # Convert to DataFrame
        df = pd.DataFrame(all_results)

        # Save compiled results
        results_file = f"{output_path}/compiled_results.json"
        with fsspec.open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)

        csv_file = f"{output_path}/compiled_results.csv"
        with fsspec.open(csv_file, "w") as f:
            df.to_csv(f, index=False)

        logger.info(f"Compiled results saved to: {results_file}")

    # Create input paths and output path
    input_paths = [step.cd("results.json") for step in steps]
    output_path = OutputName("compiled_results")

    return ExecutorStep(
        name="scratch/chiheem/compile_results",
        fn=_compile_results_fn,
        config={"input_paths": input_paths, "output_path": output_path},
        description="Compile results from multiple lm-eval steps into a single DataFrame",
    )


if __name__ == "__main__":
    # Quiet ray logs for this experiment
    logging.getLogger("ray").setLevel(logging.WARNING)

    all_steps = []

    for model_config in MODELS:
        for task in EVAL_TASKS:
            # Use evaluate_lm_evaluation_harness which handles dataset loading from lm-evaluation-harness
            engine_kwargs = {
                "tensor_parallel_size": model_config.get("tensor_parallel_size", 4),
            }
            # Ensure that max_model_len > max_gen_toks + prompt len.
            # Note that max_gen_toks is controlled by lm-eval 
            engine_kwargs["max_model_len"] = int(32768+2048)
            engine_kwargs["max_gen_toks"] = int(32768) # No point. It will be overwritten by lm-eval task's yaml config
            
            lm_eval_task_step = evaluate_lm_evaluation_harness(
                model_config["name"],
                model_config["path"],
                evals=[task],
                max_eval_instances=None,
                engine_kwargs=engine_kwargs,
                resource_config=resource_config,
                apply_chat_template=model_config.get("apply_chat_template", False),
                generation_params={
                    "temperature": 0.7,
                    "top_p": 1.0,
                    "do_sample": True,
                    "n": 1,  # Generate 1 sample per prompt
                    "seed": 42,
                },
            )
            all_steps.append(lm_eval_task_step)

    # Add compile results step
    compile_step = compile_results(all_steps)
    all_steps.append(compile_step)

    executor_main(steps=all_steps)
