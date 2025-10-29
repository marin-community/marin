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

"""Utility to launch all exp1602 LM eval configs via ray-based runner."""

from __future__ import annotations

import argparse
import subprocess
import sys
import os
from pathlib import Path


TASK_SUFFIXES = {
    "reasoning",
    "emotional_ethics",
    "language",
    "code",
    # "medical",
    "knowledge",
    # "bias_safety",
    "action",
    "truthfulness",
    "specialized",
}

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent / "exp1602_lm_eval_configs"
REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run exp1602 LM eval configs via src/marin/run/ray_run.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models",
        nargs="*",
        # default=["olmo_2_base_32b"],
        default=["tootsie_32b_cooldown_mantis_adamc_v2"],
        # default=["qwen2_5_32b"],
        # default=["marin_32b_base"],
        # default=["gemma2_27b", "qwen2_5_32b", "olmo_2_base_32b"],
        # default=["qwen3_32b", "qwen2_5_32b", "olmo_2_base_32b", "gemma2_27b"],
        help="Limit runs to specific model identifiers (e.g. marin_32b_base).",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=["knowledge"],
        # default=["emotional_ethics"],
        # default=["language"],
        # default=["medical"],
        # default=["truthfulness"],
        # default=["reasoning", "truthfulness"],
        # default=list(TASK_SUFFIXES),
        # default=["language", "bias_safety", "code"],
        # default=["reasoning", "language", "code", "emotional_ethics", "action", "specialized", "truthfulness"],
        help="Limit runs to specific task suffixes "
        f"({', '.join(sorted(TASK_SUFFIXES))}).",
    )
    parser.add_argument(
        "--force-run-failed",
        dest="force_run_failed",
        choices=("true", "false"),
        default="true",
        help="Value passed to --force_run_failed for each invocation.",
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Keep launching remaining configs even if one fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        help="Additional arguments appended to each evaluation command.",
    )
    parser.add_argument(
        "--config-dir",
        default=DEFAULT_CONFIG_DIR,
        help="Directory containing exp1602 config scripts to run.",
    )
    return parser.parse_args()


def iter_configs(config_dir: Path):
    for path in sorted(config_dir.glob("exp1602_lm_eval_*.py")):
        name = path.stem
        prefix = "exp1602_lm_eval_"
        if not name.startswith(prefix):
            continue
        suffix = None
        for candidate in TASK_SUFFIXES:
            if name.endswith(f"_{candidate}"):
                suffix = candidate
                break
        if not suffix:
            continue
        model_part = name[len(prefix) : -len(suffix) - 1]
        yield path, model_part, suffix


def build_command(path: Path, force_run_failed: str, extra_args: list[str] | None) -> list[str]:
    relative_path = path.relative_to(REPO_ROOT).as_posix()
    cmd = [
        "python",
        "src/marin/run/ray_run.py",
        "--no_wait",
        "--env_vars",
        "WANDB_API_KEY",
        "4b8c591686efb17a11070e6f4fbe8ef1b559dd80",
        "--env_vars",
        "HF_TOKEN",
        os.environ["HF_TOKEN"],
        "--env_vars",
        "WANDB_ENTITY",
        "marin-community",
        "--env_vars",
        "WANDB_PROJECT",
        "marin",
        "--cluster",
        "us-central1",
        "--",
        "python",
        relative_path,
        "--force_run_failed",
        force_run_failed,
    ]
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def main() -> int:
    args = parse_args()
    config_dir = Path(args.config_dir).expanduser().resolve()

    if not config_dir.exists():
        print(f"Config directory not found: {config_dir}", file=sys.stderr)
        return 2

    requested_models = set(args.models or [])
    requested_tasks = set(args.tasks or [])

    if args.tasks and not set(args.tasks).issubset(TASK_SUFFIXES):
        missing = set(args.tasks) - TASK_SUFFIXES
        print(f"Unknown task suffixes: {', '.join(sorted(missing))}", file=sys.stderr)
        return 2

    extra_args = args.extra_args or []

    for config_path, model, task in iter_configs(config_dir):
        if requested_models and model not in requested_models:
            continue
        if requested_tasks and task not in requested_tasks:
            continue

        cmd = build_command(config_path, args.force_run_failed, extra_args)
        print(" ".join(cmd), flush=True)

        if args.dry_run:
            continue

        result = subprocess.run(cmd, check=False)
        if result.returncode != 0 and not args.continue_on_failure:
            return result.returncode

    return 0


if __name__ == "__main__":
    sys.exit(main())
