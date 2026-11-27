# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade a code reply against the shipped AceCode assert cases: 1 when every case passes.

``config.json`` carries the row's assert strings verbatim. The candidate's last fenced block
is exec'd, then each assert case runs in the same namespace. A failed or erroring case scores
0; harness defects exit nonzero as infrastructure errors.
"""

import json
import re
from pathlib import Path

config = json.loads(Path("/tests/config.json").read_text())
answer = Path("/app/answer.txt").read_text()

blocks = re.findall(r"```(?:\w+)?\n(.*?)```", answer, re.DOTALL)
code = blocks[-1] if blocks else answer

namespace: dict = {}
exec(code, namespace)
cases = config["test_cases"]
assert isinstance(cases, list) and cases, "the shipped case list must be nonempty"
for case in cases:
    try:
        exec(case, namespace)
    except Exception:
        print(0)
        raise SystemExit(0) from None
print(1)
