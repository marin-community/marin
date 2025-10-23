#!/usr/bin/env python3
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
Analyze vLLM RL pipeline benchmark results from wandb.

Usage:
    python analyze_vllm_benchmark.py <wandb-run-path>
    
Example:
    python analyze_vllm_benchmark.py username/project/run_id
"""

import sys
import wandb
from marin.rl.benchmark_utils import compute_benchmark_summary, print_benchmark_report


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    run_path = sys.argv[1]
    
    print(f"Loading wandb run: {run_path}")
    api = wandb.Api()
    run = api.run(run_path)
    
    print("Computing benchmark metrics...")
    metrics = compute_benchmark_summary(run)
    
    print_benchmark_report(metrics)
    
    # Print key benchmark results in machine-readable format
    print("\nKey Metrics (copy-paste friendly):")
    print(f"Weight transfer time per step: {metrics.avg_weight_transfer_duration_sec:.3f} seconds")
    print(f"vLLM inference throughput: {metrics.avg_inference_tokens_per_sec:.1f} tokens/second")
    print(f"Levanter training throughput: {metrics.avg_training_tokens_per_sec:.1f} tokens/second")


if __name__ == "__main__":
    main()
