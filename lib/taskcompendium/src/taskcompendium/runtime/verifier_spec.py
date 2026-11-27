# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage a verifyit grader's declared contract as ``verifier.toml``.

A grading environment may carry its own frozen verifyit runtime (the one its
release was oracle-qualified with), which rejects fields newer verifyit
versions add with defaults. Rendering only the declared fields keeps the staged
contract identical to what the task declares: an undeclared field takes the
environment runtime's own default, and a declared field the runtime does not
know still fails closed inside the sandbox.
"""

import tomllib

import tomli_w
from verifyit.spec import render_spec

from taskcompendium.models import VerifyitGrader, verifyit_spec


def declared_verifier_toml(grader: VerifyitGrader) -> bytes:
    """The rendered spec restricted to ``mode`` and the grader's declared parameters."""
    table = tomllib.loads(render_spec(verifyit_spec(grader)))
    declared = {"mode", *grader.parameters}
    return tomli_w.dumps({key: value for key, value in table.items() if key in declared}).encode()
