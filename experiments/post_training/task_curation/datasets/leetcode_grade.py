# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade a code reply with the shipped LeetCode check() harness: 1 when every assertion passes.

``config.json`` carries the row's ``preamble`` (the source's own runtime environment, verbatim
native prompt code), ``entry_point``, ``test_source`` and the ``starter_code`` provenance. The
candidate's last fenced block is the complete solution; the harness runs exactly as the source
ships it. Candidate-code failures (a failed assertion, an exception raised by the candidate's
function) score 0. Harness defects exit nonzero: they are infrastructure errors, never zeros.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from leetcode_harness import harness_reward

config = json.loads(Path("/tests/config.json").read_text())
answer = Path("/app/answer.txt").read_text()
print(harness_reward(config, answer))
