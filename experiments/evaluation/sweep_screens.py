# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the frozen agentic LR-sweep screens."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

CONFIGS = Path(__file__).parent / "task208-screens"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    command = [
        sys.executable,
        "-m",
        "experiments.evaluation.cli",
        "launch",
        "--model-config",
        str(args.model_config),
        "--harbor-config",
        str(CONFIGS / "swebench-verified-100.yaml"),
        "--harbor-config",
        str(CONFIGS / "openthoughts-tblite-100.yaml"),
        "--harbor-config",
        str(CONFIGS / "bfclparity-pi.yaml"),
        "--evalchemy-config",
        str(CONFIGS / "gsm8k-0shot.yaml"),
        "--platform",
        "gpu",
        "--accelerator",
        "H100x8",
        "--federated_cluster",
        "cw-rno2a",
        "--priority",
        "interactive",
        "--version",
        args.version,
        "--no-wait",
    ]
    if not args.run:
        print(subprocess.list2cmdline(command))
        return
    if not os.environ.get("MARIN_PREFIX"):
        raise ValueError("Set MARIN_PREFIX explicitly before launching")
    environment = dict(os.environ, NCCL_DEBUG="WARN", NCCL_DEBUG_SUBSYS="")
    subprocess.run(command, env=environment, check=True)


if __name__ == "__main__":
    main()
