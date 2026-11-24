# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score a reply with verifyit's IFEval checks: the fraction of its instructions the reply satisfies.

``constraints`` in ``/tests/config.json`` are IFEval instruction ids with their arguments. A blank reply
scores 0.
"""

import json
from pathlib import Path

from verifyit.modes.grade_ifeval import grade_ifeval_candidate
from verifyit.spec import DEFAULT_OUTPUT, Constraint, IfevalSpec

CONFIG = Path("/tests/config.json")
ANSWER = Path(DEFAULT_OUTPUT)


def main() -> None:
    constraints = json.loads(CONFIG.read_text())["constraints"]
    spec = IfevalSpec(tuple(Constraint(constraint["name"], constraint["params"]) for constraint in constraints))
    verdict = grade_ifeval_candidate(spec, ANSWER.read_text())
    # A blank reply is scored before any check runs, so its verdict lists no constraints.
    passed = sum(result["passed"] for result in verdict.detail.get("constraints", []))
    print(passed / len(spec.constraints))


if __name__ == "__main__":
    main()
