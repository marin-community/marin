# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Keep subprocess options supported by the source images' Python 3.6.
# ruff: noqa: UP021, UP022

"""Run Scale-SWE's original test restoration and scorer in its source image."""

import json
import os
import re
import subprocess
from pathlib import Path

from native_scripts import ENV, RESTORE

CONFIG = Path("/tests/config.json")


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    os.chdir(config["workdir"])
    env = {**os.environ, **ENV, "base": config["base_commit"]}
    subprocess.run(["sh", "-c", RESTORE], env=env, check=False)
    if config["f2p_patch"].strip():
        patch = Path("/tmp/scaleswe_f2p.patch")
        patch.write_text(config["f2p_patch"], encoding="utf-8")
        for command in (
            ["git", "apply", "--verbose", str(patch)],
            ["git", "apply", "--verbose", "--ignore-space-change", "--ignore-whitespace", str(patch)],
            ["patch", "--batch", "--fuzz=5", "-p1", "-i", str(patch)],
            ["git", "apply", "--verbose", "--reject", "--ignore-whitespace", str(patch)],
        ):
            if subprocess.run(command, env=env, check=False).returncode == 0:
                break
    if config["f2p_script"]:
        Path("test_fail_to_pass.py").write_text(config["f2p_script"], encoding="utf-8")
    ids = Path("/tmp/scaleswe_test_ids.json")
    ids.write_text(json.dumps(config["test_ids"]))
    result = subprocess.run(
        ["python", "/tests/scale_score.py", str(ids)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    print(result.stdout)
    matches = re.findall(r"<score>([0-9.]+)</score>", result.stdout)
    score = float(matches[-1]) if matches else 0.0
    Path("/logs/verifier/reward.txt").write_text(str(score))


if __name__ == "__main__":
    main()
