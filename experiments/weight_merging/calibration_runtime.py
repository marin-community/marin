# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Install pinned standalone Grug dependencies and execute calibration."""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skyrl-revision", required=True)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--code-revision", required=True)
    args = parser.parse_args()
    subprocess.run(
        ["uv", "pip", "install", "--python", sys.executable, "transformers==5.16.1", "loguru==0.7.3"], check=True
    )
    directory = Path(tempfile.mkdtemp(prefix="merge-calibration-skyrl-"))
    subprocess.run(["git", "init", str(directory)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(directory),
            "fetch",
            "--depth=1",
            "https://github.com/marin-community/MarinSkyRL",
            args.skyrl_revision,
        ],
        check=True,
    )
    subprocess.run(["git", "-C", str(directory), "checkout", "--detach", "FETCH_HEAD"], check=True)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join([str(directory / "skyrl-train"), str(Path.cwd())])
    command = [
        sys.executable,
        "-m",
        "experiments.weight_merging.calibrate",
        "--recipe",
        args.recipe,
        "--code-revision",
        args.code_revision,
    ]
    os.execve(sys.executable, command, environment)


if __name__ == "__main__":
    main()
