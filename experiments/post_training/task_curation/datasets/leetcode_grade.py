# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade a code reply with the shipped LeetCode check() harness: 1 when every assertion passes.

``config.json`` carries the row's ``starter_code``, ``entry_point`` and ``test`` source
verbatim. The candidate's last fenced block completes the starter code; the harness runs
exactly as the source ships it. Candidate-code failures (a failed assertion, an exception
raised by the candidate's function) score 0. Harness defects exit nonzero: they are
infrastructure errors, never zeros.
"""

import json
import re
from pathlib import Path

config = json.loads(Path("/tests/config.json").read_text())
answer = Path("/app/answer.txt").read_text()

blocks = re.findall(r"```(?:\w+)?\n(.*?)```", answer, re.DOTALL)
code = blocks[-1] if blocks else answer

namespace: dict = {}
exec(config["starter_code"], namespace)
try:
    exec(code, namespace)
except Exception:
    print(0)
    raise SystemExit(0) from None
try:
    candidate = eval(config["entry_point"], namespace)
except Exception:
    print(0)
    raise SystemExit(0) from None
exec(config["test_source"], namespace)
check = namespace["check"]
try:
    check(candidate)
except AssertionError:
    print(0)
    raise SystemExit(0) from None
except Exception:
    print(0)
    raise SystemExit(0) from None
print(1)
