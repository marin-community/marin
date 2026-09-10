# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mode ifeval: the answer file must satisfy every IFEval constraint the spec lists.

The reward is all-or-nothing, and the detail records each constraint's verdict so a failing run
says which instruction was missed. A constraint the registry does not know is a task defect, not a
candidate failure, so it raises ``InvalidTask`` before the candidate is read.
"""

from pathlib import Path

from tasktrove_verify.modes.ifeval_constraints import CONSTRAINTS, Check
from tasktrove_verify.output import read_output
from tasktrove_verify.reward import InvalidTask, Reward, scored
from tasktrove_verify.spec import Constraint, IfevalSpec


def resolve_checks(constraints: tuple[Constraint, ...]) -> list[tuple[Constraint, Check]]:
    if not constraints:
        raise InvalidTask("ifeval spec lists no constraints")
    unknown = sorted({c.name for c in constraints} - set(CONSTRAINTS))
    if unknown:
        raise InvalidTask(f"unknown ifeval constraints: {unknown}")
    return [(c, CONSTRAINTS[c.name]) for c in constraints]


def grade(spec: IfevalSpec, tests_dir: Path, workspace: Path) -> Reward:
    checks = resolve_checks(spec.constraints)
    text = read_output(spec, workspace)
    if text is None:
        return scored(0.0, reason="no_output")

    results = []
    for constraint, check in checks:
        try:
            passed, detail = check(text, constraint.params)
        except Exception as error:
            passed, detail = False, f"{type(error).__name__}: {error}"
        results.append({"name": constraint.name, "passed": passed, "detail": detail})
    failed = [result["name"] for result in results if not result["passed"]]
    return scored(0.0 if failed else 1.0, constraints=results, failed=failed)
