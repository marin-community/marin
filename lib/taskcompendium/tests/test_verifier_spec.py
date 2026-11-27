# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The staged verifier.toml carries the task's declared contract, not newer runtime defaults."""

import tomllib

from verifyit.spec import parse_spec

from taskcompendium.models import CommandSemantics, EnvironmentRequirements, VerifyitGrader, verifyit_spec
from taskcompendium.runtime.verifier_spec import declared_verifier_toml

LOCK_ENVIRONMENT = EnvironmentRequirements(
    command_semantics=CommandSemantics.LINUX_PROCESS, packages_lock="s3://bucket/grader/requirements.lock"
)


def math_grader(**extra) -> VerifyitGrader:
    # The retained OpenMathReasoning row's declared parameters (LOCAL proof c).
    parameters = {
        "expected": "54",
        "math_type": "scalar",
        "output": "/app/answer.txt",
        "empty_output": "zero",
        "profile": "anchored",
        "allow_additive_constant": False,
        **extra,
    }
    return VerifyitGrader(mode="math", parameters=parameters, environment=LOCK_ENVIRONMENT)


def test_undeclared_default_fields_are_not_staged():
    """A frozen grader runtime that predates positive_symbols must still read the declared spec."""
    grader = math_grader()
    staged = tomllib.loads(declared_verifier_toml(grader).decode())
    assert set(staged) == {"mode", *grader.parameters}
    assert "positive_symbols" not in staged
    assert parse_spec(declared_verifier_toml(grader).decode()) == verifyit_spec(grader)


def test_declared_new_fields_are_staged_for_the_runtime_to_judge():
    grader = math_grader(positive_symbols=["x"])
    staged = tomllib.loads(declared_verifier_toml(grader).decode())
    assert staged["positive_symbols"] == ["x"]
    assert parse_spec(declared_verifier_toml(grader).decode()) == verifyit_spec(grader)
