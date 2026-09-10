# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A synthesized oracle for output-file modes: ``solution/solve.sh`` writes the known-good answer."""

import shlex

from tasktrove_verify.probe import positive_candidate
from tasktrove_verify.spec import Spec

SOLVE_SH = "solution/solve.sh"


def answer_solution(spec: Spec) -> dict[str, bytes]:
    """Solution files that write the spec's positive candidate to its output file.

    Empty when the mode has no positive candidate, so execution modes keep whatever the source shipped.
    """
    candidate = positive_candidate(spec)
    if candidate is None:
        return {}
    output = spec.output
    script = f"#!/bin/bash\nset -e\nmkdir -p {shlex.quote(str(output.rsplit('/', 1)[0]))}\n"
    script += f"printf '%s\\n' {shlex.quote(candidate)} > {shlex.quote(output)}\n"
    return {SOLVE_SH: script.encode()}
