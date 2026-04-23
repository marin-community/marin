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

"""Launch SLURM jobs for a rerank-decode sweep on MATH-500.

Submits one job per (chunk_size, num_samples) pair. Each job runs
sweep_math500.py as an executor step. The executor framework handles
output paths via this_output_path().

Usage:
    python launch_sweep_math500.py --prefix gs://my-bucket/path --dry_run
    python launch_sweep_math500.py --prefix gs://my-bucket/path
"""

import argparse
import itertools
import os
import subprocess

CHUNK_SIZES = [1, 2, 5, 10]
NUM_SAMPLES = [16, 32, 64]

WORKER_SCRIPT = "./experiments/rerank_decode/sweep_math500.py"


def build_cmd(args: dict[str, object], slurm_prefix: str = "nlprun -g 2 -d a6000", log_path: str | None = None) -> str:
    args_str = " ".join(f"--{k}={v}" for k, v in args.items())
    log_str = f"-o {log_path} " if log_path else ""
    return f"{slurm_prefix} {log_str}'uv run {WORKER_SCRIPT} {args_str}'"


def main():
    parser = argparse.ArgumentParser(description="Launch MATH-500 rerank-decode sweep as SLURM jobs.")
    parser.add_argument("--prefix", type=str, required=True, help="Executor prefix (e.g. gs://my-bucket/path)")
    parser.add_argument("--log_dir", type=str, default="./slurm_logs/rerank_decode_sweep")
    parser.add_argument("--slurm_prefix", type=str, default="nlprun -g 2 -d a6000")
    parser.add_argument("--scorer", type=str, default="kv_cache", choices=["vllm", "kv_cache"])
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if not args.dry_run:
        os.makedirs(args.log_dir, exist_ok=True)

    combos = list(itertools.product(CHUNK_SIZES, NUM_SAMPLES))

    for i, (chunk_size, num_samples) in enumerate(combos):
        key = f"chunk{chunk_size}_n{num_samples}"
        job_args = {
            "chunk_size": chunk_size,
            "num_samples": num_samples,
            "scorer": args.scorer,
            "prefix": args.prefix,
        }
        log_path = os.path.join(args.log_dir, f"{key}.out")

        cmd = build_cmd(job_args, slurm_prefix=args.slurm_prefix, log_path=log_path)

        print(f"[{i + 1}/{len(combos)}] {key}")
        print(f"  {cmd}")
        if not args.dry_run:
            subprocess.run(cmd, shell=True, check=True)

    print(f"\nSubmitted {len(combos)} jobs")


if __name__ == "__main__":
    main()
