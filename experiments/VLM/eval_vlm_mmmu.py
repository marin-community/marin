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
VLM Evaluation on MMMU using Marin/Ray distribution.

This script creates 30 ExecutorSteps (one per MMMU subject) and distributes
them across multiple TPU VMs via Ray for parallel evaluation.

Usage:
    # Run all 30 MMMU subjects in parallel (requires Ray cluster)
    python experiments/VLM/eval_vlm_mmmu.py

    # Run a specific subject only
    python experiments/VLM/eval_vlm_mmmu.py --executor.run_only="evaluation/vlm_evaluation_harness/Art"

    # Custom checkpoint
    VLM_CHECKPOINT="gs://..." python experiments/VLM/eval_vlm_mmmu.py

    # Custom TPU type
    TPU_TYPE="v5p-8" python experiments/VLM/eval_vlm_mmmu.py
"""

import os

from fray.cluster import ResourceConfig

from experiments.evals.evals import default_eval_vlm
from experiments.evals.task_configs import EvalTaskConfig
from marin.execution.executor import executor_main

# ============================================================================
# MMMU Subject Configuration (30 subjects)
# ============================================================================
MMMU_SUBJECTS = [
    "Accounting",
    "Agriculture",
    "Architecture_and_Engineering",
    "Art",
    "Art_Theory",
    "Basic_Medical_Science",
    "Biology",
    "Chemistry",
    "Clinical_Medicine",
    "Computer_Science",
    "Design",
    "Diagnostics_and_Laboratory_Medicine",
    "Economics",
    "Electronics",
    "Energy_and_Power",
    "Finance",
    "Geography",
    "History",
    "Literature",
    "Manage",
    "Marketing",
    "Materials",
    "Math",
    "Mechanical_Engineering",
    "Music",
    "Pharmacy",
    "Physics",
    "Psychology",
    "Public_Health",
    "Sociology",
]

# ============================================================================
# Configuration
# ============================================================================
# Default checkpoint path (can be overridden via environment variable)
DEFAULT_CHECKPOINT = (
    "gs://marin-us-east1/checkpoints/vlm-official-qwen3-1.7b-stage3_new-0-4e97cb/"
    "hf/vlm-official-qwen3-1.7b-stage3_new-0-4e97cb/step-39061/"
)

CHECKPOINT = os.environ.get("VLM_CHECKPOINT", DEFAULT_CHECKPOINT)
TPU_TYPE = os.environ.get("TPU_TYPE", "v5p-8")
MAX_EXAMPLES = os.environ.get("MAX_EXAMPLES", None)

if MAX_EXAMPLES is not None:
    MAX_EXAMPLES = int(MAX_EXAMPLES)


def create_mmmu_eval_steps():
    """Create one ExecutorStep per MMMU subject.

    Returns:
        List of ExecutorSteps, one for each MMMU subject
    """
    eval_steps = []

    for subject in MMMU_SUBJECTS:
        task_name = f"mmmu_val_{subject}"

        step = default_eval_vlm(
            step=CHECKPOINT,
            resource_config=ResourceConfig.with_tpu(TPU_TYPE),
            evals=[EvalTaskConfig(task_name, 0)],  # 0-shot
            max_eval_instances=MAX_EXAMPLES,
            discover_latest_checkpoint=False,
        )
        eval_steps.append(step)

    return eval_steps


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    eval_steps = create_mmmu_eval_steps()

    print("=" * 60)
    print("VLM Evaluation: MMMU (Marin/Ray Distribution)")
    print("=" * 60)
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"TPU Type: {TPU_TYPE}")
    print(f"Max Examples: {MAX_EXAMPLES or 'All'}")
    print(f"Subjects: {len(MMMU_SUBJECTS)}")
    print("=" * 60)

    executor_main(steps=eval_steps)
