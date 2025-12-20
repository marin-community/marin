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

"""Evaluate a (HF-exported) checkpoint on a TPU slice via the executor.

This is intentionally lightweight: provide a `--model-path` pointing at an HF export directory
(e.g. `gs://.../hf/step-60000/`) and it will run a default evaluation suite.
"""

import argparse
import sys

from experiments.evals.evals import default_eval
from experiments.evals.task_configs import CORE_TASKS_PLUS_MMLU
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main


def _parse_args():
    parser = argparse.ArgumentParser(description="Run a small evaluation suite on an HF-exported checkpoint.")
    parser.add_argument(
        "--model-path",
        default="gs://marin-us-central1/checkpoints/example/hf/step-0/",
        help='HF export directory, e.g. "gs://.../hf/step-60000/".',
    )
    parser.add_argument(
        "--tpu-type",
        default="v5p-64",
        help='TPU type for `ResourceConfig.with_tpu` (default: "v5p-64").',
    )
    parser.add_argument(
        "--slice-count",
        type=int,
        default=1,
        help="Number of TPU slices (default: 1).",
    )
    return parser.parse_known_args()


if __name__ == "__main__":
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]

    eval_step = default_eval(
        args.model_path,
        ResourceConfig.with_tpu(args.tpu_type, slice_count=args.slice_count),
        evals=CORE_TASKS_PLUS_MMLU,
    )
    executor_main([eval_step])
