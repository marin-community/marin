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

"""Configuration for Kelp tree diffusion evaluation."""

from dataclasses import dataclass, field

from fray.cluster import ResourceConfig


@dataclass(frozen=True)
class KelpEvalTaskConfig:
    """Configuration for a single Kelp evaluation task."""

    name: str
    """Name of the evaluation task (e.g., 'mbpp', 'humaneval', 'validity')."""

    num_samples: int = 1
    """Number of samples to generate per problem for pass@k computation."""

    max_iterations: int = 100
    """Maximum diffusion iterations for generation."""

    temperature: float = 1.0
    """Sampling temperature for generation."""


# Pre-defined evaluation task configurations
VALIDITY_EVAL = KelpEvalTaskConfig(name="validity", num_samples=1)
MBPP_EVAL = KelpEvalTaskConfig(name="mbpp", num_samples=10)
HUMANEVAL_EVAL = KelpEvalTaskConfig(name="humaneval", num_samples=10)


@dataclass(frozen=True)
class KelpEvaluationConfig:
    """Configuration for Kelp tree diffusion model evaluation."""

    model_path: str
    """Path to the model checkpoint (local or GCS)."""

    output_path: str
    """Where to write evaluation results."""

    evals: list[KelpEvalTaskConfig] = field(default_factory=lambda: [VALIDITY_EVAL, MBPP_EVAL])
    """List of evaluation tasks to run."""

    resource_config: ResourceConfig = field(default_factory=lambda: ResourceConfig.with_tpu("v4-8"))
    """Resource configuration for running evaluation."""

    max_eval_instances: int | None = None
    """Maximum number of evaluation instances per task."""

    batch_size: int = 8
    """Batch size for evaluation."""

    use_grammar_constraints: bool = True
    """Whether to use grammar-based logit masking during generation."""

    seed: int = 42
    """Random seed for reproducibility."""
