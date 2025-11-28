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

from dataclasses import dataclass, field
from marin.core.runtime import TaskConfig
from marin.processing.classification.autoscaler import DEFAULT_AUTOSCALING_ACTOR_POOL_CONFIG, AutoscalingActorPoolConfig


@dataclass(frozen=True)
class DatasetSchemaConfig:
    input_columns: list[str] = field(default_factory=lambda: ["text", "id"])
    output_columns: list[str] = field(default_factory=lambda: ["id", "attributes", "generated_text", "text"])

    id_column: tuple[str, ...] = field(default_factory=lambda: ("id",))
    """Path (tuple of keys) to the unique identifier in a row, e.g. ("metadata", "id")."""

    prompt_column: str = "text"


@dataclass
class RuntimeConfig:
    memory_limit_gb: int
    resources: dict = field(default_factory=dict)


@dataclass
class InferenceConfig:
    input_path: str

    # A path to a model or the name of a model. If it doesn't have the classifier type in its name, you need to
    # specify the model_type.
    model_name: str
    attribute_name: str

    # The type of the model. Check AutoClasssifier for the available models.
    model_type: str | None = None
    output_path: str | None = None

    # Ray runtime config.
    runtime: RuntimeConfig = field(
        default_factory=lambda: RuntimeConfig(
            memory_limit_gb=0.1,
        )
    )

    # Ray task config.
    task: TaskConfig = field(default_factory=TaskConfig)

    # The filetype of the input data.
    filetype: str = "jsonl.gz"

    # Batch size for processing documents
    batch_size: int = 512

    # Whether to resume from existing progress
    resume: bool = True

    classifier_kwargs: dict = field(default_factory=dict)

    # Whether to use the autoscaling actor pool
    autoscaling_actor_pool_config: AutoscalingActorPoolConfig = field(
        default_factory=lambda: DEFAULT_AUTOSCALING_ACTOR_POOL_CONFIG
    )

    # Dataset schema configuration (input/output columns)
    dataset_schema: DatasetSchemaConfig = field(default_factory=DatasetSchemaConfig)

    # Number of batches to wait for before uploading the results to the output file.
    num_batches_per_upload: int = 10
