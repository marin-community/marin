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
    EvalTaskConfig("aime25", num_fewshot=0, task_alias="aime25_0shot"),
]

MODELS = [
    {
        "name": "qwen2.5-7b-instruct",
        "path": "gs://marin-us-central2/models/qwen2.5-7b-instruct",
        "apply_chat_template": True,
        "max_length": int(8094*4), # Do not limit
        "tensor_parallel_size": 4,
    }
]


def compile_results(steps: list[ExecutorStep]) -> ExecutorStep:
    """
    Takes in a list of ExecutorSteps for lm-eval tasks and compiles the results into a single DataFrame.
    Reads the results.json files from each step's output path, extracts the 'results' field,
    combines them into a DataFrame, and logs the results.
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
            logger.info(f"Loading results from {input_path}")

            # Read the JSON file from the step's output path
            with fsspec.open(input_path, "r") as f:
                data = json.load(f)

            # Extract the 'results' field
            if "results" in data:
                results = data["results"]
                # Add step identifier and model name
                for task_name, task_results in results.items():
                    task_results["task_name"] = task_name

                    # Extract model name from the input path
                    # The path format should be: gs://bucket/evaluation/lm_evaluation_harness/{model_name}-{hash}/results.json
                    path_parts = input_path.split("/")
                    if len(path_parts) > 1:
                        # Get the directory name (second to last part)
                        dir_name = path_parts[-2]
                        if "-" in dir_name:
                            # Remove the hash part
                            model_name = dir_name.rsplit("-", 1)[0]
                            # Remove the task prefix to get just the model name
                            if "_" in model_name:
                                # Split on underscores to remove 'lm_evaluation_harness_' prefix
                                parts = model_name.split("_")
                                if len(parts) >= 3:
                                    model_name = "_".join(parts[2:])  # Get everything after 'lm_evaluation_harness_'
                        else:
                            model_name = dir_name
                    else:
                        model_name = f"unknown_model_{i}"

                    task_results["model_name"] = model_name
                    all_results.append(task_results)
            else:
                logger.warning(f"No 'results' field found in {input_path}")

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
        for eval_task in EVAL_TASKS:
            # Use evaluate_lm_evaluation_harness which handles dataset loading from lm-evaluation-harness
            lm_eval_task_step = evaluate_lm_evaluation_harness(
                model_config["name"],
                model_config["path"],
                evals=[eval_task],
                max_eval_instances=None,
                engine_kwargs={
                    "tensor_parallel_size": model_config.get("tensor_parallel_size", 4),
                    # "max_model_len": model_config.get("max_length", 8192),
                },
                resource_config=resource_config,
                apply_chat_template=model_config.get("apply_chat_template", False),
                generation_params={
                    # "max_gen_toks": max(1, model_config.get("max_length", 8192) - 2048)
                    # if model_config.get("max_length")
                    # else 8192,
                    "temperature": 0.7,
                    "top_p": 1.0,
                    "do_sample": True,
                    "n": 10,  # Generate 1 sample per prompt
                    "seed": 42,
                },
            )
            all_steps.append(lm_eval_task_step)

    # Add compile results step
    compile_step = compile_results(all_steps)
    all_steps.append(compile_step)

    executor_main(steps=all_steps)
