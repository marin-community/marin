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
import shutil
import subprocess
import tempfile
import time
import asyncio
from pathlib import Path

import fsspec
from marin.evaluation.evaluators.evaluator import Evaluator, ModelConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.utils import is_remote_path
from fray.cluster import ResourceConfig
from fray.cluster.ray.deps import build_runtime_env_for_packages

logger = logging.getLogger(__name__)


class HarborEvaluator(Evaluator):
    """
    Generic evaluator for any Harbor dataset from the registry.

    Can run any Harbor benchmark without custom adapters:
    - Uses Harbor's programmatic API for execution
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
        agent_kwargs = dict(harbor_config.get("agent_kwargs", {}))

        logger.info(f"Running Harbor evaluation: {dataset}@{version}")
        logger.info(f"Agent: {agent}, Model: {model.name}, Concurrent: {n_concurrent}, Env: {env_type}")

        if max_eval_instances:
            logger.info(f"Limiting to first {max_eval_instances} tasks")

        # If model has a path, serve it via vLLM and point Harbor at the local server
        if model.path:
            server_url, served_model_name = self._start_vllm_server(model)
            vllm_model_name = f"hosted_vllm/{served_model_name}"
            api_base = server_url
            logger.info(f"vLLM server ready: model={served_model_name}, api_base={api_base}")
            agent_kwargs.setdefault("api_base", api_base)
            agent_kwargs.setdefault(
                "model_info",
                {
                    "max_input_tokens": 32768,
                    "max_output_tokens": 8192,
                    "input_cost_per_token": 0.0,
                    "output_cost_per_token": 0.0,
                },
            )
            try:
                self._run_eval_inner(
                    model_name=vllm_model_name,
                    harbor_config=harbor_config,
                    agent=agent,
                    agent_kwargs=agent_kwargs,
                    n_concurrent=n_concurrent,
                    env_type=env_type,
                    output_path=output_path,
                    max_eval_instances=max_eval_instances,
                    wandb_tags=wandb_tags,
                    dataset=dataset,
                    version=version,
                )
            finally:
                from marin.evaluation.utils import kill_process_on_port

                try:
                    kill_process_on_port(8000)
                except Exception as e:
                    logger.warning(f"Failed to kill vLLM server: {e}")
        else:
            self._run_eval_inner(
                model_name=model.name,
                harbor_config=harbor_config,
                agent=agent,
                agent_kwargs=agent_kwargs,
                n_concurrent=n_concurrent,
                env_type=env_type,
                output_path=output_path,
                max_eval_instances=max_eval_instances,
                wandb_tags=wandb_tags,
                dataset=dataset,
                version=version,
            )

    @staticmethod
    def _start_vllm_server(
        model: ModelConfig,
        host: str = "127.0.0.1",
        port: int = 8000,
        timeout_seconds: int = 3600,
    ) -> tuple[str, str]:
        """Start a vLLM server as a native subprocess and wait for it to be ready.

        Returns (server_url, served_model_name).
        """
        import dataclasses
        import requests
        from urllib.parse import urlparse

        # Auto-enable streaming for GCS paths
        parsed = urlparse(model.path or "")
        if parsed.scheme in ("gs", "s3") and "load_format" not in model.engine_kwargs:
            engine_kwargs = dict(model.engine_kwargs)
            engine_kwargs["load_format"] = "runai_streamer"
            model = dataclasses.replace(model, engine_kwargs=engine_kwargs)

        model_name_or_path = model.path if model.path else model.name

        # Use model.name as the served name (simple, no slashes from GCS paths)
        served_model_name = model.name

        # Build CLI args from engine_kwargs
        extra_args: list[str] = []
        load_format = model.engine_kwargs.get("load_format")
        if isinstance(load_format, str):
            extra_args.extend(["--load-format", load_format])

        command: list[str] = [
            "vllm",
            "serve",
            model_name_or_path,
            "--served-model-name",
            served_model_name,
            "--trust-remote-code",
            "--host",
            host,
            "--port",
            str(port),
            "--distributed-executor-backend",
            "ray",
            *extra_args,
        ]

        env = dict(os.environ)
        env.setdefault("MODEL_IMPL_TYPE", "vllm")
        env.setdefault("VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION", "1")
        env.setdefault("VLLM_TPU_SKIP_PRECOMPILE", "1")

        logger.info(f"Starting vLLM server: {' '.join(command)}")
        subprocess.Popen(command, env=env)

        server_url = f"http://{host}:{port}/v1"
        start_time = time.time()
        while True:
            try:
                response = requests.get(f"{server_url}/models")
                if response.status_code == 200:
                    raw = response.json()
                    loaded = [m["id"] for m in raw["data"]]
                    logger.info(f"vLLM server up at {server_url}: {loaded}")
                    if served_model_name in loaded:
                        logger.info(f"Model {served_model_name} loaded.")
                        break
                    logger.info(f"Model not loaded yet. Loaded: {loaded}")
            except requests.ConnectionError:
                elapsed = time.time() - start_time
                logger.info(f"vLLM not ready yet ({elapsed:.0f}s)")

            if time.time() - start_time > timeout_seconds:
                raise TimeoutError("vLLM server did not start within timeout.")
            time.sleep(5)

        return server_url, served_model_name

    def _run_eval_inner(
        self,
        model_name: str,
        harbor_config: dict,
        agent: str,
        agent_kwargs: dict,
        n_concurrent: int,
        env_type: str,
        output_path: str,
        max_eval_instances: int | None,
        wandb_tags: list[str] | None,
        dataset: str,
        version: str,
    ) -> None:
        """Run Harbor trials and save results."""
        results, job_dir = self._run_harbor_trials(
            dataset=dataset,
            version=version,
            model_name=model_name,
            agent=agent,
            agent_kwargs=agent_kwargs,
            n_concurrent=n_concurrent,
            env_type=env_type,
            task_limit=max_eval_instances,
        )

        parsed_results = self._parse_results(results)
        parsed_results["meta"] = {
            "dataset": dataset,
            "version": version,
            "agent": agent,
            "n_concurrent": n_concurrent,
            "env": env_type,
            "max_eval_instances": max_eval_instances,
            "model": model_name,
        }
        self._save_results(parsed_results, output_path, wandb_tags, model_name, dataset, version, job_dir)

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
        agent_kwargs: dict | None = None,
    ) -> tuple[dict, Path | None]:
        """Run Harbor trials using programmatic API."""
        from harbor.job import Job
        from harbor.models.job.config import JobConfig, RegistryDatasetConfig
        from harbor.models.trial.config import AgentConfig, EnvironmentConfig
        from harbor.models.orchestrator_type import OrchestratorType
        from harbor.models.registry import RemoteRegistryInfo
        from harbor.models.environment_type import EnvironmentType

        # Create temporary output directory
        tmpdir = tempfile.mkdtemp(prefix="harbor_eval_")
        try:
            output_dir = Path(tmpdir) / "harbor_results"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Map environment type
            # "local" in Marin means Docker in Harbor
            harbor_env_type = EnvironmentType.DOCKER
            if env_type != "local":
                try:
                    harbor_env_type = EnvironmentType(env_type)
                except ValueError:
                    logger.warning(f"Unknown environment type: {env_type}, falling back to docker")

            # Create Harbor JobConfig
            config = JobConfig(
                job_name=f"eval_{dataset}_{int(time.time())}",
                jobs_dir=output_dir,
                datasets=[
                    RegistryDatasetConfig(
                        registry=RemoteRegistryInfo(),
                        name=dataset,
                        version=version,
                        n_tasks=task_limit,
                    )
                ],
                agents=[
                    AgentConfig(
                        name=agent,
                        model_name=model_name,
                        kwargs=agent_kwargs or {},
                    )
                ],
                orchestrator=dict(
                    type=OrchestratorType.LOCAL,
                    n_concurrent_trials=n_concurrent,
                ),
                environment=EnvironmentConfig(
                    type=harbor_env_type,
                ),
            )

            job = Job(config)
            logger.info(f"Starting Harbor job {job.config.job_name} via programmatic API")

            # Run the job
            asyncio.run(job.run())

            logger.info("Harbor execution completed")
            job_dir = job.job_dir

            # Fix permissions on Docker-created files so we can read them
            try:
                subprocess.run(
                    ["sudo", "chmod", "-R", "755", str(job_dir)],
                    check=False,
                    capture_output=True,
                )
            except Exception:
                pass  # Continue even if chmod fails

            # Read trial results from Harbor's result.json files
            results = {"trials": {}}

            trial_dirs = [d for d in job_dir.iterdir() if d.is_dir()]
            for trial_dir in trial_dirs:
                result_file = trial_dir / "result.json"
                if not result_file.exists():
                    continue

                trial_result = json.loads(result_file.read_text())
                trial_id = trial_result.get("task_name", trial_dir.name)

                # Extract reward from verifier_result.rewards.reward
                verifier_result = trial_result.get("verifier_result") or {}
                rewards = verifier_result.get("rewards") or {}
                reward = rewards.get("reward", 0.0)

                # Extract error info if present
                error = None
                exception_info = trial_result.get("exception_info")
                if exception_info:
                    error = {
                        "type": exception_info.get("exception_type"),
                        "message": exception_info.get("exception_message"),
                    }

                # Read trajectory from agent/trajectory.json if it exists
                trajectory_file = trial_dir / "agent" / "trajectory.json"
                trajectory_content = None
                trajectory_length = 0
                if trajectory_file.exists():
                    try:
                        trajectory_content = trajectory_file.read_text()
                        trajectory_data = json.loads(trajectory_content)
                        trajectory_length = len(trajectory_data.get("steps", []))
                    except Exception as e:
                        logger.warning(f"Failed to read trajectory for {trial_id}: {e}")

                results["trials"][trial_id] = {
                    "reward": reward,
                    "status": "completed" if not exception_info else "failed",
                    "task_name": trial_id,
                    "started_at": trial_result.get("started_at"),
                    "finished_at": trial_result.get("finished_at"),
                    "error": error,
                    "trajectory_content": trajectory_content,
                    "trajectory_length": trajectory_length,
                }

            return results, job_dir
        finally:
            # Clean up temp directory, ignoring permission errors from Docker
            try:
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception as e:
                logger.warning(f"Could not fully clean up temp directory {tmpdir}: {e}")
                # Try using sudo to clean up Docker files if available
                try:
                    subprocess.run(["sudo", "rm", "-rf", tmpdir], check=False, capture_output=True)
                except Exception:
                    pass

    def _parse_results(self, results: dict) -> dict:
        """Parse Harbor results into Marin format."""

        trials = results.get("trials", {})

        parsed: dict = {
            "trials": {},
            "aggregate": {
                "total_trials": 0,
                "successful_trials": 0,
                "mean_reward": 0.0,
                "accuracy": 0.0,
            },
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
                "trajectory_length": trial_data.get("trajectory_length", 0),  # Already extracted
                "trajectory_content": trial_data.get("trajectory_content"),  # Already extracted
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
        version: str,
        job_dir: Path | None = None,
    ) -> None:
        """Save results to GCS and log to W&B."""

        timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        results_file_name = f"results_{timestamp}.json"
        samples_file_name = f"samples_{timestamp}.jsonl"

        # Ensure local output path exists for local filesystem runs.
        if not is_remote_path(output_path):
            os.makedirs(output_path, exist_ok=True)
            os.makedirs(os.path.join(output_path, "trajectories"), exist_ok=True)

        samples_path = os.path.join(output_path, samples_file_name)
        results_path = os.path.join(output_path, results_file_name)

        # Save trajectory files that were already extracted (before temp dir cleanup)
        for trial_id, trial_data in results["trials"].items():
            trajectory_content = trial_data.get("trajectory_content")
            if trajectory_content:
                try:
                    # Determine file extension based on content format
                    # If it looks like JSON (starts with {), save as .json, otherwise as .jsonl
                    ext = ".json" if trajectory_content.strip().startswith("{") else ".jsonl"
                    trajectory_path = os.path.join(output_path, "trajectories", f"{trial_id}{ext}")

                    with fsspec.open(trajectory_path, "w") as dst:
                        dst.write(trajectory_content)

                    results["trials"][trial_id]["trajectory_path"] = trajectory_path
                    logger.info(
                        f"Saved trajectory for {trial_id} to {trajectory_path} "
                        f"(length: {trial_data.get('trajectory_length', 0)})"
                    )

                except Exception as e:
                    logger.warning(f"Failed to save trajectory for {trial_id}: {e}")

        trials_for_output = {}
        for trial_id, trial_data in results["trials"].items():
            trial_output = dict(trial_data)
            # Remove internal fields that shouldn't be in the output
            trial_output.pop("trial_dir", None)
            trial_output.pop("trajectory_content", None)  # Already saved to file
            trials_for_output[trial_id] = trial_output

        with fsspec.open(samples_path, "w") as f:
            for trial_id, trial_data in sorted(trials_for_output.items()):
                sample = {
                    "task_id": trial_id,
                    "dataset": dataset,
                    "version": version,
                    "model": model_name,
                    "timestamp": timestamp,
                    **trial_data,
                }
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        aggregated_results = {
            "meta": results.get("meta", {}) | {"timestamp": timestamp},
            "aggregate": results["aggregate"],
            "samples_path": samples_path,
        }
        with fsspec.open(results_path, "w") as f:
            json.dump(aggregated_results, f, indent=2, ensure_ascii=False)

        logger.info(f"Wrote samples to {samples_path}")
        logger.info(f"Wrote aggregated results to {results_path}")

        # Log to W&B
        try:
            import wandb

            wandb.init(
                project=os.environ.get("WANDB_PROJECT") or "harbor",
                name=f"{model_name}-{dataset}",
                tags=(wandb_tags or []) + ["harbor", dataset],
                config={
                    "model": model_name,
                    "dataset": dataset,
                    "version": version,
                    "evaluator": "harbor",
                },
            )

            # Log aggregate metrics
            wandb.log(results["aggregate"])

            # Log table of per-trial results
            import pandas as pd

            trials_df = pd.DataFrame.from_dict(trials_for_output, orient="index")
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
        """Launch evaluation with Ray (for distributed execution).

        When model.path is set (vLLM serving), launches a fray sub-job on TPU/GPU
        with the correct vllm extras. Otherwise calls evaluate directly.
        """
        if model.path:
            from fray.cluster import Entrypoint, EnvironmentConfig, JobRequest, current_cluster

            evaluator = self

            def _run():
                import logging

                logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
                evaluator.evaluate(
                    model=model,
                    evals=evals,
                    output_path=output_path,
                    max_eval_instances=max_eval_instances,
                    wandb_tags=wandb_tags,
                    generation_params=generation_params,
                )

            env_vars = {}
            for key in [
                "WANDB_API_KEY",
                "HF_TOKEN",
                "ANTHROPIC_API_KEY",
                "DAYTONA_API_KEY",
                "OPENAI_API_KEY",
                "E2B_API_KEY",
                "MODAL_API_KEY",
            ]:
                val = os.environ.get(key)
                if val:
                    env_vars[key] = val
            env_vars["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

            job_request = JobRequest(
                name="harbor-vllm-evaluation",
                entrypoint=Entrypoint.from_callable(_run),
                resources=resource_config,
                environment=EnvironmentConfig.create(
                    extras=["harbor", "vllm"],
                    env_vars=env_vars,
                ),
            )
            cluster = current_cluster()
            job_id = cluster.launch(job_request)
            result = cluster.monitor(job_id)
            if result.status == "failed":
                raise RuntimeError(f"Harbor vLLM evaluation job failed: {result.error_message}")
            return

        self.evaluate(
            model=model,
            evals=evals,
            output_path=output_path,
            max_eval_instances=max_eval_instances,
            wandb_tags=wandb_tags,
            generation_params=generation_params,
        )
