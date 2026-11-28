# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The OpenSWE grading scripts: the upstream eval_script is the sole authority.

Ruling 2026-11-26 6:03 AM (option a): the upstream eval_script is kept verbatim
as the selection and grading authority; the recorded ``OPENSWE_EXIT_CODE`` is
the verdict; grading fails closed on a missing or ambiguous exit. The reward
starts at 0 and only an unambiguous verdict scores 1:

- exactly one ``^OPENSWE_EXIT_CODE=`` line in the eval output, and that line is
  exactly ``OPENSWE_EXIT_CODE=0`` — the upstream fork's resolution contract
  (``re.search(r'^OPENSWE_EXIT_CODE=0$', log, re.MULTILINE)``) narrowed by the
  ambiguity rule: more than one marker line means the recorded verdict is not
  trustworthy (the eval script echoes it once, so a second line is injected or
  broken output), and the row scores 0.
- anything else — a missing marker, a nonzero exit, a malformed value — scores 0.
"""

VERIFIER_SCRIPT = """#!/bin/bash
set -uo pipefail
# OpenSWE grading: the upstream eval_script runs verbatim and its recorded
# OPENSWE_EXIT_CODE is the verdict. Fail closed: the reward starts at 0 and
# only an unambiguous single OPENSWE_EXIT_CODE=0 line scores 1. The script
# always exits 0 — the reward file arbitrates the score.
mkdir -p /logs/verifier
echo 0 > /logs/verifier/reward.txt
if [ ! -f /tests/eval_script.sh ]; then
  echo "GRADING_FAILURE: the upstream eval_script is missing from the verifier material"
  exit 0
fi
bash /tests/eval_script.sh > /logs/verifier/eval-output.txt 2>&1
eval_status=$?
printf '%s\\n' "$eval_status" > /logs/verifier/eval-exit-status.txt
cat /logs/verifier/eval-output.txt
total=$(grep -c '^OPENSWE_EXIT_CODE=' /logs/verifier/eval-output.txt || true)
zero=$(grep -c '^OPENSWE_EXIT_CODE=0$' /logs/verifier/eval-output.txt || true)
if [ "$total" -eq 1 ] && [ "$zero" -eq 1 ]; then
  echo 1 > /logs/verifier/reward.txt
fi
exit 0
"""

SOLVE_SCRIPT = """#!/bin/bash
set -euo pipefail
cd {workdir}
git apply --whitespace=nowarn /solution/gold.patch
"""


def verifier_script() -> bytes:
    """The tests/test.sh bytes: run the upstream eval_script verbatim, fail closed."""
    return VERIFIER_SCRIPT.encode()


def solve_script(workdir: str) -> bytes:
    """The solution/solve.sh bytes: the oracle applies the gold patch in the workspace."""
    if not workdir.startswith("/"):
        raise ValueError(f"An oracle workspace must be absolute: {workdir!r}")
    return SOLVE_SCRIPT.format(workdir=workdir).encode()


def reward(output: str) -> int:
    """The grader's verdict for one captured eval output; the same rule the script applies.

    Exposed for tests and the fidelity gate's evidence: exactly one
    ``OPENSWE_EXIT_CODE=`` line and it is exactly ``OPENSWE_EXIT_CODE=0``.
    """
    lines = [line for line in output.splitlines() if line.startswith("OPENSWE_EXIT_CODE=")]
    return 1 if len(lines) == 1 and lines[0] == "OPENSWE_EXIT_CODE=0" else 0
