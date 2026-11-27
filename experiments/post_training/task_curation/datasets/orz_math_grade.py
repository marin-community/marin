# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade an ORZ math reply: extract the final boxed answer and verify it against the reference.

The ORZ collection's ground truths are exact math answers. The candidate's last
``\\boxed{...}`` is extracted (falling back to the text after the last ``Answer:`` marker);
``math_verify`` compares it to the ground truth as mathematics, with an exact string
fallback. Anything unextractable or unverifiable scores 0; harness defects exit nonzero.
"""

import json
import re
import sys
from pathlib import Path

from math_verify import parse, verify

BOXED = re.compile(r"\\boxed\{")
ANSWER_MARKER = re.compile(r"Answer:\s*(.+?)\s*$", re.DOTALL)


def extract_candidate(reply: str) -> str | None:
    """The candidate's final answer: the last balanced \\boxed{...}, else the Answer: tail."""
    matches = list(BOXED.finditer(reply))
    if matches:
        start = matches[-1].end()
        depth = 1
        for index in range(start, len(reply)):
            if reply[index] == "{":
                depth += 1
            elif reply[index] == "}":
                depth -= 1
                if depth == 0:
                    return reply[start:index]
        return None
    marker = ANSWER_MARKER.search(reply.strip())
    if marker:
        return marker.group(1).strip()
    return None


def score(reply: str, ground_truth: str) -> int:
    candidate = extract_candidate(reply)
    if candidate is None:
        return 0
    try:
        # math_verify parses LaTeX correctly when the target is wrapped in math delimiters.
        if verify(parse(ground_truth), parse(f"${candidate}$")):
            return 1
    except Exception:
        pass
    return int(candidate.strip().casefold() == ground_truth.strip().casefold())


def main() -> None:
    config = json.loads(Path("/tests/config.json").read_text())
    reply = Path("/app/answer.txt").read_text()
    print(score(reply, config["ground_truth"]))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"orz grader harness defect: {error}", file=sys.stderr)
        raise SystemExit(1) from None
