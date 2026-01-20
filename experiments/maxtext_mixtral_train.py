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
Run a MaxText Mixtral training smoke test on a TPU slice via Fray's Ray TPU launcher.

This script exists because MaxText is not wired into Marin/Levanter's training entrypoints; we instead
launch `python -m MaxText.train ...` inside the TPU pod environment.

Expected setup:
- `submodules/maxtext` is cloned (see https://github.com/AI-Hypercomputer/maxtext).
- Run this via `uv run python -m marin.run.ray_run ... --extra tpu -- python -m experiments.maxtext_mixtral_train`.
"""

import argparse
import datetime as dt
import logging
import os
import subprocess
import sys
from pathlib import Path
import shutil

import ray

from fray.cluster.ray.tpu.execution import run_on_pod

logger = logging.getLogger("ray")

MAXTEXT_SRC = "submodules/maxtext/src"
MAXTEXT_BASE_CONFIG = os.path.join(MAXTEXT_SRC, "MaxText", "configs", "base.yml")
MIXTRAL_TOKENIZER = os.path.join(MAXTEXT_SRC, "MaxText", "assets", "tokenizer.mistral-v1")


def _tpu_chip_count(tpu_type: str) -> int:
    try:
        return int(tpu_type.rsplit("-", 1)[-1])
    except Exception as e:
        raise ValueError(f"Could not parse chip count from tpu_type={tpu_type!r} (expected suffix like '-64').") from e


def _default_run_name() -> str:
    ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"mixtral_8x7b_maxtext_bs256_seq4096_{ts}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch MaxText Mixtral training on a TPU slice.")
    parser.add_argument("--tpu-type", default="v5p-64")
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--dataset-type", default="synthetic", choices=["synthetic", "tfds", "hf", "grain"])
    parser.add_argument(
        "--ici-expert-parallelism",
        type=int,
        default=1,
        help="MaxText ICI expert parallelism. 1 disables expert-parallel sharding; >1 enables it.",
    )
    parser.add_argument("--base-output-directory", default="gs://marin-us-central1/maxtext")
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    chips = _tpu_chip_count(args.tpu_type)
    if args.global_batch_size % chips != 0:
        raise ValueError(
            f"--global-batch-size={args.global_batch_size} must be divisible by TPU chip count={chips} "
            f"for tpu_type={args.tpu_type}."
        )
    per_device_batch_size = args.global_batch_size // chips
    run_name = args.run_name or _default_run_name()

    logger.info(
        "Launching MaxText Mixtral: tpu=%s global_bs=%s per_device_bs=%s seq=%s steps=%s dataset_type=%s run_name=%s",
        args.tpu_type,
        args.global_batch_size,
        per_device_batch_size,
        args.seq_len,
        args.steps,
        args.dataset_type,
        run_name,
    )

    ray.init(address="auto", ignore_reinit_error=True, log_to_driver=True)

    @ray.remote(max_calls=1)
    def _run_maxtext_train() -> int:
        # MaxText strongly recommends Python 3.12 + a dedicated venv. TPU runtime images for Marin
        # may differ; we follow the MaxText docs when possible and fall back to the current Python.
        venv_dir = Path("/tmp/maxtext_venv")
        python312 = shutil.which("python3.12")
        python_cmd = python312 or sys.executable

        bash = [
            "bash",
            "-lc",
            "set -euo pipefail\n"
            f"PY='{python_cmd}'\n"
            f"VENV='{venv_dir}'\n"
            'if [ ! -d "$VENV" ]; then\n'
            '  "$PY" -m venv "$VENV"\n'
            "fi\n"
            'source "$VENV/bin/activate"\n'
            "python -m pip install -U pip uv\n"
            "# If we have Python 3.12, follow MaxText's recommended install path.\n"
            "if python -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 12) else 1)'; then\n"
            "  cd submodules/maxtext\n"
            "  uv pip install -e '.[tpu]' --resolution=lowest\n"
            "  install_maxtext_github_deps || true\n"
            "  cd - >/dev/null\n"
            "fi\n"
            "# Install TensorFlow stack (MaxText imports TF unconditionally for some input pipeline modules).\n"
            "python -m pip install -U 'tensorflow>=2.19.1' 'tensorflow-text>=2.19.0' 'tensorflow-datasets>=4.9.9'\n"
            f"export PYTHONPATH='{MAXTEXT_SRC}:'\"${{PYTHONPATH:-}}\"\n"
            "python -m MaxText.train "
            f"'{MAXTEXT_BASE_CONFIG}' "
            f"base_output_directory='{args.base_output_directory}' "
            f"run_name='{run_name}' "
            "model_name=mixtral-8x7b "
            f"per_device_batch_size={per_device_batch_size} "
            f"steps={args.steps} "
            f"max_target_length={args.seq_len} "
            f"dataset_type={args.dataset_type} "
            f"tokenizer_path='{MIXTRAL_TOKENIZER}' "
            "enable_checkpointing=false async_checkpointing=false "
            "attention=flash dtype=bfloat16 weight_dtype=bfloat16 "
            "megablox=true sparse_matmul=true "
            "ici_fsdp_parallelism=-1 "
            f"ici_expert_parallelism={args.ici_expert_parallelism}\n",
        ]
        logger.info("MaxText (bash) command: %s", " ".join(bash))
        subprocess.run(bash, check=True)
        return 0

    run_on_pod(_run_maxtext_train, args.tpu_type)


if __name__ == "__main__":
    main()
