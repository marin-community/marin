# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate or submit the canonical SWE-bench context-compaction grid."""

import argparse
import subprocess
from pathlib import Path

from generate_configs import ARMS, MODELS

ROOT = Path(__file__).resolve().parent


def _command(model: str, arm: str, *, submit: bool, limit: int | None) -> list[str]:
    command = [
        "uv",
        "run",
        "python",
        "-m",
        "experiments.evaluation.cli",
        "launch",
        "--model-config",
        str(ROOT / "configs" / "models" / model / f"{arm}.yaml"),
        "--harbor-config",
        str(ROOT / "configs" / "harbor" / f"{arm}.yaml"),
        "--federated_cluster",
        "cw-rno2a",
        "--priority",
        "interactive",
        "--description",
        "SWE-bench Verified context and mini-swe-agent compaction ablation",
    ]
    if limit is not None:
        command.extend(("--limit", str(limit)))
    command.append("--no-wait" if submit else "--dry-run")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=[model.key for model in MODELS], action="append")
    parser.add_argument("--arm", choices=[arm.key for arm in ARMS], action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    models = args.model or [model.key for model in MODELS]
    arms = args.arm or [arm.key for arm in ARMS]
    for model in models:
        for arm in arms:
            command = _command(model, arm, submit=args.submit, limit=args.limit)
            print(" ".join(command), flush=True)
            subprocess.run(command, check=True, cwd=ROOT.parents[5])


if __name__ == "__main__":
    main()
