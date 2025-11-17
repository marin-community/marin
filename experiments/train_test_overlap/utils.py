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
from dataclasses import dataclass

import ray

# Import evaluation dataset conversion steps so executor can resolve paths
from experiments.train_test_overlap.eval_datasets_overlap import (
    ai2_arc_convert_dolma,
    bbh_convert_dolma,
    boolq_convert_dolma,
    commonsense_qa_convert_dolma,
    gpqa_convert_dolma,
    gsm8k_convert_dolma,
    hellaswag_convert_dolma,
    humaneval_convert_dolma,
    instruction_following_convert_dolma,
    lambada_openai_convert_dolma,
    math_convert_dolma,
    mmlu_convert_dolma,
    mmlu_pro_convert_dolma,
    musr_convert_dolma,
    openbookqa_convert_dolma,
    piqa_convert_dolma,
    truthful_qa_convert_dolma,
    winograd_wsc_convert_dolma,
)
from marin.execution.executor import ExecutorStep
from marin.processing.classification.dedupe import DedupeConfig, DedupMode, NGramConfig, dedupe

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# List of evaluation dataset steps - same as in aggregate_total.py
EVAL_DATASET_STEPS: list[ExecutorStep] = [
    gsm8k_convert_dolma,
    math_convert_dolma,
    truthful_qa_convert_dolma,
    bbh_convert_dolma,
    mmlu_convert_dolma,
    humaneval_convert_dolma,
    instruction_following_convert_dolma,
    gpqa_convert_dolma,
    musr_convert_dolma,
    mmlu_pro_convert_dolma,
    hellaswag_convert_dolma,
    ai2_arc_convert_dolma,
    boolq_convert_dolma,
    commonsense_qa_convert_dolma,
    lambada_openai_convert_dolma,
    openbookqa_convert_dolma,
    piqa_convert_dolma,
    winograd_wsc_convert_dolma,
]


ALLOWED_TPU_TYPES: tuple[str, ...] = ("v4-8", "v5p-8", "v6e-4")


@dataclass(frozen=True)
class UnifiedResources:
    """Unified resource configuration for Ray tasks.

    This provides a single place to describe CPU/memory and TPU-related resource tags
    in a consistent way across tasks.

    Fields:
        tpu_type: TPU type string (e.g., "v4-8", "v6e-4"). If provided and
            `tpu_head_fraction` is set, we add a custom resource key
            f"TPU-{tpu_type}-head": tpu_head_fraction.
        tpu_head_fraction: Fractional amount to reserve against the TPU head resource
            (used as a global concurrency limiter for tasks on a particular TPU fleet).
        tpus: Number of TPU chips to require on a worker (adds {"TPU": tpus}).
            Most non-TPU tasks can leave this as None.
        num_cpus: Number of CPUs for the Ray task options.
        memory: Memory in bytes for the Ray task options.
        extra: Additional custom resource requirements to merge into the resources dict.
    """

    tpu_type: str | None = None
    tpu_head_fraction: float | None = None
    tpus: int | None = None
    num_cpus: int | None = None
    memory: int | None = None
    extra: dict[str, float] | None = None

    def to_ray_overrides(self) -> dict:
        overrides: dict = {}
        resources: dict[str, float] = {}

        if self.num_cpus is not None:
            overrides["num_cpus"] = self.num_cpus
        if self.memory is not None:
            overrides["memory"] = self.memory

        if self.tpu_type and self.tpu_head_fraction is not None:
            if self.tpu_type not in ALLOWED_TPU_TYPES:
                raise ValueError(f"Unsupported tpu_type '{self.tpu_type}'. Allowed: {', '.join(ALLOWED_TPU_TYPES)}")
            resources[f"TPU-{self.tpu_type}-head"] = self.tpu_head_fraction
        if self.tpus is not None:
            resources["TPU"] = float(self.tpus)
        if self.extra:
            resources.update(self.extra)

        if resources:
            overrides["resources"] = resources
        return overrides


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for a single dataset to process for train-test overlap detection."""

    name: str
    """Human-readable name for the dataset (used in output paths)."""

    path: str
    """Path to the dataset directory (local, GCS, or S3)."""

    max_in_flight: int
    """Maximum number of parallel tasks to run for this dataset."""

    text_field: str = "text"
    """Name of the text field in the parquet file."""


@dataclass(frozen=True)
class ShardedDedupeConfig:
    """Configuration for running train-test overlap detection for an entire dataset.

    Zephyr handles file discovery and parallelism internally, so this just configures
    dataset-level parameters.
    """

    dataset_dir: str
    """Path to the training dataset directory (Zephyr will discover all files)."""

    output_path: str
    """Base output directory for results."""

    eval_dataset_steps: list[ExecutorStep] = None
    """Evaluation dataset steps for path resolution."""

    text_field: str = "text"
    """Name of the text field in the data files."""

    processes: int = 15
    """Number of parallel processes for Zephyr backend."""

    unified_resources: UnifiedResources | None = None
    """Ray resource configuration for the dataset-level task."""


# N-gram configuration for train-test overlap detection
DEFAULT_NGRAM_CONFIG = NGramConfig(
    ngram_length=[15],  # Multiple n-gram sizes - modify this to change n-grams
    overlap_threshold=1e-6,
    stride=0,
)


@ray.remote
def run_train_test_overlap(config: ShardedDedupeConfig) -> str:
    """
    Run train-test overlap detection for an entire dataset.

    Zephyr handles file discovery and parallelism internally - this just calls
    dedupe once with the dataset directory. Ray parallelism is only at the
    dataset level (multiple datasets can run in parallel as separate Ray tasks).

    Args:
        config: Configuration including dataset directory, output path, and evaluation dataset steps

    Returns:
        Output path where results were written
    """
    logger.info(f"Running train-test overlap for dataset at {config.dataset_dir}")

    # Run dedupe once for the entire dataset (Zephyr discovers files internally)
    dedupe_config = DedupeConfig(
        input_path=config.eval_dataset_steps,
        output_path=config.output_path,
        decontaminate_source=config.dataset_dir,  # Entire dataset directory!
        attribute_name="ngram_overlap",
        false_positive_rate=1e-20,
        ngram=DEFAULT_NGRAM_CONFIG,
        processes=config.processes,
        mode=DedupMode.TRAIN_TEST_OVERLAP,
        text_field=config.text_field,
    )

    logger.info(f"Calling dedupe with {config.processes} processes for Zephyr backend")
    dedupe(dedupe_config)

    logger.info(f"Train-test overlap completed! Results written to {config.output_path}")
    return config.output_path
