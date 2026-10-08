# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score an ARC submission with the vendored NVARC scorer and print the reward.

``/tests/config.json`` holds ``{"mode", "contract"}``: the NVARC record (``expected_output``, and
``test_input`` for inductive tasks). The submission is ``/app/solution.py`` when present, else
``/app/answer.txt``: a reply, or a file the agent wrote. A transductive submission is parsed as a
grid. An inductive submission's ``transform`` runs on the test input in a fresh interpreter under uid
65534 in the temporary directory; the script first makes the config readable by root alone, so the
program cannot read the expected output.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "/tests")

from local_sandbox import LocalSandbox
from skyrl_gym.envs.nemotron_ultra.nvarc import grade_inductive_arc, grade_transductive_arc

CONFIG = Path("/tests/config.json")
SUBMISSIONS = (Path("/app/solution.py"), Path("/app/answer.txt"))
UNPRIVILEGED_USER = 65534


def submission() -> str:
    for path in SUBMISSIONS:
        if path.is_file():
            return path.read_text(errors="replace")
    return ""


def main() -> None:
    config = json.loads(CONFIG.read_text())
    mode, record = config["mode"], config["contract"]
    if mode == "transductive":
        reward, detail = grade_transductive_arc(submission(), record)
    elif mode == "inductive":
        CONFIG.chmod(0o600)
        sandbox = LocalSandbox(user=UNPRIVILEGED_USER)
        reward, detail = grade_inductive_arc(submission(), record, sandbox=sandbox)
    else:
        raise ValueError(f"Unknown ARC mode: {mode}")
    print(json.dumps(detail, default=str), file=sys.stderr)
    print(reward)


if __name__ == "__main__":
    main()
