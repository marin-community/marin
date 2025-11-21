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

"""Common types for inference requests and responses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fray import ResourceConfig


@dataclass
class InferenceRequest:
    """Request for inference."""

    prompt: str
    sampling_params: dict[str, Any] | None = None
    request_id: str | None = None


@dataclass
class InferenceResult:
    """Result of inference."""

    text: list[str]
    request_id: str | None = None
    error: str | None = None


@dataclass
class ModelConfig:
    name: str
    """The name of the model e.g., allenai/olmo-7b"""

    path: str | None
    """
    The path to the model checkpoint. Can be a local path or GCS path.
    Both vLLM and Levanter can load directly from GCS.
    """

    engine_kwargs: dict[str, Any]
    """
    Additional keyword arguments to pass to the vLLM engine.
    """

    generation_params: dict | None = None
    """
    Additional keyword arguments passed to the SamplingParams for the vLLM engine
    """

    apply_chat_template: bool = False
    """
    Whether or not this model was trained with a Chat Template in the tokenizer
    """


@dataclass(frozen=True)
class EvalTaskConfig:
    name: str
    """Name of the evaluation task."""

    num_fewshot: int
    """Number of few-shot examples to evaluate on."""

    task_alias: str | None = None
    """Alias for the task name."""


@dataclass(frozen=True)
class EvaluationConfig:
    evaluator: str
    """Name of the evaluator to run."""

    model_config: ModelConfig
    worker_resources: ResourceConfig

    evaluation_path: str = "tmp/output"
    """
    Where to write results to. Can be a local path (e.g., /path/to/output) or
    a path on GCS (e.g., gs://bucket/path/to/output).
    """

    evals: list[EvalTaskConfig] = field(default_factory=list)
    """
    List of specific evals within an evaluation harness to run. This would be a list of
    tasks in for EleutherAI's lm-evaluation-harness or a list of evals from HELM (e.g., mmlu, lite, etc.).
    See https://github.com/stanford-crfm/helm/tree/main/src/helm/benchmark/presentation, or
    https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks
    for the full list.
    """

    discover_latest_checkpoint: bool = False
    """
    Whether to discover the latest HF checkpoint in the model path.
    """

    launch_with_ray: bool = True
    """
    Whether to launch the evaluation run with Ray.
    """

    max_eval_instances: int | None = None
    """
    Maximum number of evaluation instances to run.
    """

    engine_kwargs: dict | None = None
    """
    Additional keyword arguments to pass to the vLLM engine.
    """

    generation_params: dict | None = None
    """
    Additional keyword arguments passed to the vLLM sampling params engine
    """

    apply_chat_template: bool = False
    """
    Whether or not this model was trained with a Chat Template in the tokenizer
    """

    wandb_tags: list[str] | None = None
    """
    Tags to add to the wandb run.
    """
