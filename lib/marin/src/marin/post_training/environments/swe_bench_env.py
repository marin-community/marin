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

import json
import os
import random
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import datasets
import jax
import numpy as np
from swebench.inference.make_datasets.utils import extract_diff
from tqdm.auto import tqdm

from .marin_env import EnvStep, InferenceContext, MarinEnv


@dataclass(frozen=True)
class EvaluationTask:
    """Data class to hold evaluation task information."""

    step_id: str
    swe_split: str
    predictions_path: str
    instance_id: str
    output_path: str
    response_index: tuple[int, int]


class SWEBenchEnv(MarinEnv):
    """
    Environment for the SWE-Bench benchmark, which evaluates code generation models on software engineering tasks.
    https://github.com/SWE-bench/SWE-bench
    """

    def __init__(self, tokenizer, output_dir_path: str, **kwargs):
        self.tokenizer = tokenizer
        self._environment_id = uuid.uuid4().hex
        self._output_dir_path: str = os.path.join(output_dir_path, self._environment_id)
        os.makedirs(self._output_dir_path, exist_ok=True)

        self.dataset_name = kwargs.get("dataset_name", "princeton-nlp/SWE-bench_oracle")
        self.max_workers = kwargs.get("max_workers", 64)
        self.eval_max_workers = kwargs.get("eval_max_workers", 64)

        dataset = datasets.load_dataset(self.dataset_name, trust_remote_code=True)
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]

        # Convert to the format expected by the training code
        self.train_examples = []
        for item in tqdm(train_dataset, desc="Processing train set"):
            self.train_examples.append(
                {
                    "prompt": item["text"],
                    "instance_id": item["instance_id"],
                    # Store other fields that might be needed for evaluation
                    "repo": item.get("repo", ""),
                    "base_commit": item.get("base_commit", ""),
                    "patch": item.get("patch", ""),
                    "test_patch": item.get("test_patch", ""),
                }
            )

        self.eval_examples = []
        for item in tqdm(test_dataset, desc="Processing test set"):
            self.eval_examples.append(
                {
                    "prompt": item["text"],
                    "instance_id": item["instance_id"],
                    # Store other fields that might be needed for evaluation
                    "repo": item.get("repo", ""),
                    "base_commit": item.get("base_commit", ""),
                    "patch": item.get("patch", ""),
                    "test_patch": item.get("test_patch", ""),
                }
            )

        print(
            f"Initialized SWEBenchEnv with {len(self.train_examples)} train examples "
            "and {len(self.eval_examples)} eval examples"
        )

    def step(
        self,
        inference_ctx: InferenceContext,
        n_examples: int,
        prng_key,
        mode: str = "train",
        n_generations: int = 1,
        temperature: float = 1.0,
        **kwargs,
    ) -> EnvStep:
        """
        Sample problems and generate responses using the model.

        Args:
            inference_ctx: Context for generating responses (hides model params)
            n_examples: Number of examples to sample
            prng_key: Random key for sampling
            mode: "train" or "eval" (maps to "test" split for SWE-Bench)
            n_generations: Number of generations per example
            temperature: Generation temperature
        """
        if mode == "train":
            available_examples = self.train_examples
            swe_split = "train"
        else:
            available_examples = self.eval_examples
            swe_split = "test"

        with jax.default_device(jax.devices("cpu")[0]):
            n_to_sample = min(n_examples, len(available_examples))
            indices = jax.random.choice(prng_key, len(available_examples), shape=(n_to_sample,), replace=False)
            examples = [available_examples[int(idx)] for idx in indices]

        prompts = [example["prompt"] for example in examples]
        responses = inference_ctx.generate(
            prompts,
            temperature=temperature,
            n_generations=n_generations,
        )

        rewards, metrics = self._compute_rewards(examples, responses, swe_split, inference_ctx.tokenizer)

        return EnvStep(examples=examples, responses=responses, rewards=rewards, metrics=metrics)

    def _compute_rewards(
        self, examples: list[dict[str, Any]], responses: list[list[dict[str, np.ndarray]]], swe_split: str, tokenizer
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Compute rewards for generated responses by running SWE-Bench evaluation in parallel."""

        evaluation_tasks = []
        all_lens = []

        print("Preparing evaluation tasks...")
        for i, response_group in enumerate(responses):
            for j, inner_response in enumerate(response_group):
                all_lens.append(len(inner_response["tokens"]))
                decoded_response = tokenizer.decode(inner_response["tokens"], skip_special_tokens=True)

                # Extract patch from the response
                try:
                    patch = extract_diff(decoded_response)
                except Exception as e:
                    print(f"Failed to extract patch: {e}")
                    patch = ""

                # Create unique step ID for this evaluation
                step_id = f"{uuid.uuid4().hex}_{i}_{j}"
                output_path = os.path.join(self._output_dir_path, step_id)
                os.makedirs(output_path, exist_ok=True)

                record = {
                    "model_name_or_path": "marin_model",
                    "instance_id": examples[i]["instance_id"],
                    "full_output": decoded_response,
                    "model_patch": patch,
                }

                predictions_path = os.path.join(output_path, "predictions.jsonl")
                with open(predictions_path, "w") as f:
                    f.write(json.dumps(record) + "\n")

                task = EvaluationTask(
                    step_id=step_id,
                    swe_split=swe_split,
                    predictions_path=predictions_path,
                    instance_id=examples[i]["instance_id"],
                    output_path=output_path,
                    response_index=(i, j),
                )
                evaluation_tasks.append(task)

                # Occasionally print debug information
                if random.random() < 1 / 32:
                    print("=" * 25)
                    print("Instance ID:", examples[i]["instance_id"])
                    print(
                        "Response:", decoded_response[:300] + "..." if len(decoded_response) > 300 else decoded_response
                    )
                    print("Extracted patch:", patch[:200] + "..." if len(patch) > 200 else patch)
                    print("=" * 25)

        # Run all evaluations in parallel
        print(f"Running {len(evaluation_tasks)} evaluations in parallel with {self.eval_max_workers} workers...")
        reward_results = {}
        successful_evaluations = 0
        total_evaluations = len(evaluation_tasks)

        with ThreadPoolExecutor(max_workers=self.eval_max_workers) as executor:
            # Submit all tasks
            future_to_task = {executor.submit(self._evaluate_patch_task, task): task for task in evaluation_tasks}

            # Collect results with progress bar
            for future in tqdm(as_completed(future_to_task), total=len(evaluation_tasks), desc="Evaluating patches"):
                task = future_to_task[future]
                try:
                    reward = future.result()
                    reward_results[task.response_index] = float(reward)
                    if reward > 0:
                        successful_evaluations += 1
                except Exception as e:
                    print(f"Error evaluating task {task.step_id}: {e}")
                    reward_results[task.response_index] = 0.0

        # Organize rewards back into the original structure
        all_rewards = []
        for i in range(len(responses)):
            group_rewards = []
            for j in range(len(responses[i])):
                reward = reward_results.get((i, j), 0.0)
                group_rewards.append(reward)
            all_rewards.append(group_rewards)

        all_rewards = np.asarray(all_rewards)
        all_lens = np.asarray(all_lens)

        metrics = {
            "train/rewards": np.mean(all_rewards),
            "train/output_len": np.mean(all_lens),
            "train/solve_rate": np.mean(all_rewards),  # For SWE-Bench, reward is binary so this is solve rate
            "train/successful_evals": successful_evaluations,
            "train/total_evals": total_evaluations,
        }

        return all_rewards, metrics

    def _evaluate_patch_task(self, task: EvaluationTask) -> float:
        """Run SWE-Bench evaluation for a single patch (designed for parallel execution)."""
        return self._evaluate_patch(
            step_id=task.step_id,
            swe_split=task.swe_split,
            predictions_path=task.predictions_path,
            instance_id=task.instance_id,
            output_path=task.output_path,
        )

    def _evaluate_patch(
        self, step_id: str, swe_split: str, predictions_path: str, instance_id: str, output_path: str
    ) -> float:
        """Run SWE-Bench evaluation for a single patch."""
        commands = [
            "python",
            "-m",
            "swebench.harness.run_evaluation",
            "--run_id",
            step_id,
            "--dataset_name",
            self.dataset_name,
            "--split",
            swe_split,
            "--predictions_path",
            predictions_path,
            "--instance_ids",
            instance_id,
            "--max_workers",
            str(self.max_workers),
            "--report_dir",
            output_path,
        ]

        try:
            process = subprocess.Popen(commands, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, _stderr = process.communicate(timeout=300)  # 5 minute timeout

            # Award a reward of 1 when GitHub issue is resolved
            if "Instances resolved: 1" in stdout:
                return 1.0
            else:
                return 0.0

        except subprocess.TimeoutExpired:
            print(f"Evaluation timed out for instance {instance_id}")
            process.kill()
            return 0.0
        except Exception as e:
            print(f"Error evaluating instance {instance_id}: {e}")
            return 0.0

    def get_eval_examples(self, n_examples: int) -> list[dict[str, Any]]:
        """Get evaluation examples for evaluation."""
        # Use a fixed seed for reproducible evaluation
        eval_key = jax.random.PRNGKey(42)
        with jax.default_device(jax.devices("cpu")[0]):
            n_to_sample = min(n_examples, len(self.eval_examples))
            indices = jax.random.choice(eval_key, len(self.eval_examples), shape=(n_to_sample,), replace=False)
            return [self.eval_examples[int(idx)] for idx in indices]
