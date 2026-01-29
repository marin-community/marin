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
Harbor framework evaluator - runs ANY Harbor dataset from the registry.

Supports 45+ benchmarks including:
- AIME (60 math problems)
- Terminal-Bench (89 terminal tasks)
- SWE-bench Verified (500 tasks)
- And all others in https://harborframework.com/registry

No custom adapters needed - Harbor's registry handles all datasets generically.
"""

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path

from marin.evaluation.evaluators.evaluator import Evaluator, ModelConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.utils import is_remote_path, upload_to_gcs
from fray.cluster import ResourceConfig
from fray.cluster.ray.deps import build_runtime_env_for_packages

logger = logging.getLogger(__name__)


class HarborEvaluator(Evaluator):
    """
    Generic evaluator for any Harbor dataset from the registry.

    Can run any Harbor benchmark without custom adapters:
    - Uses Harbor's CLI for execution
    - Loads datasets from Harbor registry
    - Supports all Harbor agents and environments
    """

    def get_runtime_env(self) -> dict:
        """Returns Ray runtime environment."""
        env_vars = {
            "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", ""),
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
        }

        # Pass through any API keys that might be needed
        for key in ["DAYTONA_API_KEY", "E2B_API_KEY", "MODAL_API_KEY"]:
            if key in os.environ:
                env_vars[key] = os.environ[key]

        return build_runtime_env_for_packages(
            extra=["harbor"],
            pip_packages=[],
            env_vars=env_vars,
        )

    def evaluate(
        self,
        model: ModelConfig,
        evals: list[EvalTaskConfig],
        output_path: str,
        max_eval_instances: int | None = None,
        wandb_tags: list[str] | None = None,
        generation_params: dict | None = None,
    ) -> None:
        """
        Run Harbor evaluation on any dataset.

        Note: Harbor uses its own config system, so most parameters come from
        model.engine_kwargs['harbor_config'] which should contain:
        {
            "dataset": str,        # e.g., "aime", "terminal-bench"
            "version": str,        # e.g., "1.0", "2.0"
            "agent": str,          # e.g., "claude-code", "custom-vllm"
            "n_concurrent": int,   # Parallel trials
            "env": str,            # "local", "daytona", "e2b"
        }
        """
        # Extract Harbor config from model config
        harbor_config = (model.engine_kwargs or {}).get("harbor_config", {})

        dataset = harbor_config.get("dataset", "aime")
        version = harbor_config.get("version", "1.0")
        agent = harbor_config.get("agent", "claude-code")
        n_concurrent = harbor_config.get("n_concurrent", 4)
        env_type = harbor_config.get("env", "local")

        logger.info(f"Running Harbor evaluation: {dataset}@{version}")
        logger.info(f"Agent: {agent}, Model: {model.name}, Concurrent: {n_concurrent}, Env: {env_type}")

        if max_eval_instances:
            logger.info(f"Limiting to first {max_eval_instances} tasks")

        # Run Harbor trials
        results = self._run_harbor_trials(
            dataset=dataset,
            version=version,
            model_name=model.name,
            agent=agent,
            n_concurrent=n_concurrent,
            env_type=env_type,
            task_limit=max_eval_instances,
        )

        # Parse and save results
        parsed_results = self._parse_results(results)
        self._save_results(parsed_results, output_path, wandb_tags, model.name, dataset)

        logger.info("Harbor evaluation completed successfully")

    def _run_harbor_trials(
        self,
        dataset: str,
        version: str,
        model_name: str,
        agent: str,
        n_concurrent: int,
        env_type: str,
        task_limit: int | None = None,
    ) -> dict:
        """Run Harbor trials using CLI."""

        # Build Harbor run command
        cmd = [
            "harbor", "run",
            "--dataset", f"{dataset}@{version}",
            "--agent", agent,
            "--model", model_name,
            "--n-concurrent", str(n_concurrent),
        ]

        if env_type != "local":
            cmd.extend(["--env", env_type])

        # Harbor supports task slicing
        if task_limit:
            cmd.extend(["--tasks", f"0:{task_limit}"])

        # Create temporary output directory
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "harbor_results"
            cmd.extend(["--output", str(output_dir)])

            logger.info(f"Running Harbor command: {' '.join(cmd)}")

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=7200,  # 2 hour timeout for safety
                )
                logger.info(f"Harbor execution completed")
                logger.debug(f"Harbor stdout: {result.stdout}")

                # Read results
                results_file = output_dir / "results.json"
                if results_file.exists():
                    with open(results_file) as f:
                        return json.load(f)
                else:
                    logger.warning("No results.json found, returning empty results")
                    return {"trials": {}}

            except subprocess.CalledProcessError as e:
                logger.error(f"Harbor command failed with exit code {e.returncode}")
                logger.error(f"Stdout: {e.stdout}")
                logger.error(f"Stderr: {e.stderr}")
                raise RuntimeError(f"Harbor execution failed: {e.stderr}") from e
            except subprocess.TimeoutExpired as e:
                logger.error("Harbor command timed out")
                raise RuntimeError("Harbor execution timed out after 2 hours") from e

    def _parse_results(self, results: dict) -> dict:
        """Parse Harbor results into Marin format."""

        trials = results.get("trials", {})

        parsed = {
            "trials": {},
            "aggregate": {
                "total_trials": 0,
                "successful_trials": 0,
                "mean_reward": 0.0,
                "accuracy": 0.0,
            }
        }

        total_reward = 0.0
        successful = 0

        for trial_id, trial_data in trials.items():
            reward = trial_data.get("reward", 0.0)
            total_reward += reward

            # Consider reward >= 0.99 as success (allows for small floating point errors)
            if reward >= 0.99:
                successful += 1

            parsed["trials"][trial_id] = {
                "task_id": trial_id,
                "reward": reward,
                "correct": reward >= 0.99,
                "status": trial_data.get("status", "unknown"),
                "trajectory_length": len(trial_data.get("trajectory", [])),
                "error": trial_data.get("error"),
            }

        total = len(trials)
        if total > 0:
            parsed["aggregate"]["total_trials"] = total
            parsed["aggregate"]["successful_trials"] = successful
            parsed["aggregate"]["mean_reward"] = total_reward / total
            parsed["aggregate"]["accuracy"] = successful / total

        logger.info(
            f"Results summary: {successful}/{total} successful "
            f"(accuracy: {parsed['aggregate']['accuracy']:.2%}, "
            f"mean reward: {parsed['aggregate']['mean_reward']:.3f})"
        )

        return parsed

    def _save_results(
        self,
        results: dict,
        output_path: str,
        wandb_tags: list[str] | None,
        model_name: str,
        dataset: str,
    ) -> None:
        """Save results to GCS and log to W&B."""

        # Save to local first
        local_results_dir = "/tmp/harbor_results"
        os.makedirs(local_results_dir, exist_ok=True)
        local_results_file = os.path.join(local_results_dir, "results.json")

        with open(local_results_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved results locally to {local_results_file}")

        # Upload to GCS if remote path
        if is_remote_path(output_path):
            logger.info(f"Uploading results to {output_path}")
            try:
                upload_to_gcs(local_results_dir, output_path)
                logger.info(f"Successfully uploaded results to {output_path}")
            except Exception as e:
                logger.error(f"Failed to upload to GCS: {e}")
                # Don't fail the whole eval if upload fails

        # Log to W&B
        try:
            import wandb

            wandb.init(
                project="marin",
                name=f"{model_name}-{dataset}",
                tags=(wandb_tags or []) + ["harbor", dataset],
                config={
                    "model": model_name,
                    "dataset": dataset,
                    "evaluator": "harbor",
                }
            )

            # Log aggregate metrics
            wandb.log(results["aggregate"])

            # Log table of per-trial results
            import pandas as pd
            trials_df = pd.DataFrame.from_dict(results["trials"], orient="index")
            wandb.log({"trials": wandb.Table(dataframe=trials_df)})

            wandb.finish()
            logger.info("Results logged to W&B")

        except Exception as e:
            logger.warning(f"Failed to log to W&B: {e}")
            # Don't fail the whole eval if W&B logging fails

    def launch_evaluate_with_ray(
        self,
        model: ModelConfig,
        evals: list[EvalTaskConfig],
        output_path: str,
        resource_config: ResourceConfig,
        max_eval_instances: int | None = None,
        wandb_tags: list[str] | None = None,
        generation_params: dict | None = None,
    ) -> None:
        """Launch evaluation with Ray (for distributed execution)."""

        # For now, just call evaluate directly
        # Harbor manages its own parallelism via n_concurrent
        # TODO: Could implement proper Ray job submission for truly distributed execution
        self.evaluate(
            model=model,
            evals=evals,
            output_path=output_path,
            max_eval_instances=max_eval_instances,
            wandb_tags=wandb_tags,
            generation_params=generation_params,
        )
