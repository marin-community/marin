# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a TaskTrove math scorer's ``test.sh`` only under the interpreter and packages it was pinned to.

Usage: ``python3 math_grade.py python==3.11 sympy==1.13.3 ...``. The ``python`` pin names the
interpreter's major.minor version; every other pin names an installed distribution. The source
scorers turn every exception into a zero reward, so a missing or different package would silently
fail every answer. On a mismatch this exits nonzero before the scorer runs, leaving no reward file,
which the grader reports as a grading failure.
"""

import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def mismatches(pins: list[str]) -> list[str]:
    """One message per ``name==version`` pin the running environment does not satisfy."""
    problems = []
    for pin in pins:
        name, separator, expected = pin.partition("==")
        if not separator:
            raise ValueError(f"Pins take the form name==version: {pin!r}")
        if name == "python":
            observed = f"{sys.version_info.major}.{sys.version_info.minor}"
        else:
            try:
                observed = version(name)
            except PackageNotFoundError:
                observed = "missing"
        if observed != expected:
            problems.append(f"{name}=={expected} required; found {observed}")
    return problems


def main() -> int:
    problems = mismatches(sys.argv[1:])
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 2
    return subprocess.run(["bash", str(Path(__file__).with_name("test.sh"))], check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
