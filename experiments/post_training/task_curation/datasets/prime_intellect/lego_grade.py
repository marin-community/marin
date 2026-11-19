# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Keep subprocess options supported by the source images' Python 3.6.
# ruff: noqa: UP021, UP022

"""Preserve SWE-Lego's source patch restoration and pytest-summary grading."""

import json
import os
import subprocess
from pathlib import Path

from native_scripts import ENV, REVERT_TEST_FILES, build_eval_script, parse_pytest_outcomes


def main() -> None:
    config = json.loads(Path("/tests/config.json").read_text(encoding="utf-8"))
    os.chdir("/testbed")
    env = {**os.environ, **ENV}
    reward = Path("/logs/verifier/reward.txt")
    if config["test_patch"].strip():
        patch = Path("/tmp/test_patch.patch")
        patch.write_text(config["test_patch"], encoding="utf-8")
        if config["base_commit"]:
            subprocess.run(
                ["sh", "-c", REVERT_TEST_FILES],
                env={**env, "patch": str(patch), "base": config["base_commit"]},
                check=False,
            )
        for command in (
            ["git", "apply", "--whitespace=fix", str(patch)],
            ["patch", "--batch", "--fuzz=5", "-p1", "-i", str(patch)],
        ):
            if subprocess.run(command, env=env, check=False).returncode == 0:
                break
        else:
            reward.write_text("0.0")
            return
    result = subprocess.run(
        ["bash", "-c", build_eval_script(config["test_cmd"])],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    print(result.stdout)
    outcomes = parse_pytest_outcomes(result.stdout)
    required = config["test_ids"]
    reward.write_text(str(float(bool(required) and all(outcomes.get(test) == "PASSED" for test in required))))


if __name__ == "__main__":
    main()
