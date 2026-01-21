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

from collections.abc import Sequence

from fray.cluster import Entrypoint, EnvironmentConfig, JobRequest, ResourceConfig, current_cluster

from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.evaluators.evaluator import Evaluator, ModelConfig
from marin.utils import remove_tpu_lockfile_on_exit


def launch_evaluate_with_ray(
    *,
    evaluator: Evaluator,
    job_name: str,
    model: ModelConfig,
    evals: list[EvalTaskConfig],
    output_path: str,
    resource_config: ResourceConfig,
    max_eval_instances: int | None = None,
    wandb_tags: list[str] | None = None,
    extras: Sequence[str] = (),
    pip_packages: Sequence[str] = (),
    env_vars: dict[str, str] | None = None,
    configure_logging: bool = True,
) -> None:
    """Launch an evaluator on the Ray/Fray cluster."""

    def launch(
        model: ModelConfig,
        evals: list[EvalTaskConfig],
        output_path: str,
        max_eval_instances: int | None = None,
        wandb_tags: list[str] | None = None,
    ) -> None:
        if configure_logging:
            import logging

            logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
        evaluator.evaluate(model, evals, output_path, max_eval_instances, wandb_tags)

    def _run() -> None:
        with remove_tpu_lockfile_on_exit():
            launch(model, evals, output_path, max_eval_instances, wandb_tags)

    if resource_config is None:
        resource_config = ResourceConfig()

    job_request = JobRequest(
        name=job_name,
        entrypoint=Entrypoint.from_callable(_run),
        resources=resource_config,
        environment=EnvironmentConfig.create(
            extras=list(extras),
            pip_packages=list(pip_packages),
            env_vars=env_vars,
        ),
    )

    cluster = current_cluster()
    job_id = cluster.launch(job_request)
    cluster.wait(job_id, raise_on_failure=True)
