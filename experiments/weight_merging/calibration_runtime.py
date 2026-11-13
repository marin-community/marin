# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Install pinned standalone Grug dependencies and execute calibration."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

CALIBRATION_MODULES = (
    "experiments.weight_merging.calibrate",
    "experiments.weight_merging.campaign_calibrate",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skyrl-revision", required=True)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--module", choices=CALIBRATION_MODULES, required=True)
    args = parser.parse_args()
    recipe = json.loads(Path(args.recipe).read_text())
    subprocess.run(
        ["uv", "pip", "install", "--python", sys.executable, "transformers==5.16.1", "loguru==0.7.3"], check=True
    )
    with tempfile.TemporaryDirectory(prefix="merge-calibration-skyrl-") as temporary:
        directory = Path(temporary)
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
        if "skyrl_patch" in recipe:
            patch = Path(recipe["skyrl_patch"])
            if hashlib.sha256(patch.read_bytes()).hexdigest() != recipe["skyrl_patch_sha256"]:
                raise ValueError("SkyRL patch hash mismatch")
            subprocess.run(["git", "-C", str(directory), "apply", str(patch.resolve())], check=True)
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join([str(directory / "skyrl-train"), str(Path.cwd())])
        command = [
            sys.executable,
            "-m",
            args.module,
            "--recipe",
            args.recipe,
            "--code-revision",
            args.code_revision,
        ]
        subprocess.run(command, env=environment, check=True)


if __name__ == "__main__":
    main()
