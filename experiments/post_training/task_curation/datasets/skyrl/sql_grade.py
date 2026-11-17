# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score a SQL reply with the vendored SkyRL text-to-SQL comparator: 1 when its result sets match the reference.

``ground_truth`` in ``/tests/config.json`` is the comparator's prepared ground truth. The comparator
runs both queries on the seeded database and on a copy with every third row deleted. A ground truth
it cannot run fails the grader instead of scoring the reply.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "/tests")

import text_to_sql_scoring


def main() -> None:
    config = json.loads(Path("/tests/config.json").read_text())
    answer = Path("/app/answer.txt").read_text()
    outcome, detail = text_to_sql_scoring.grade(config["ground_truth"], text_to_sql_scoring.extract_sql(answer))
    if outcome == text_to_sql_scoring.GradeOutcome.INFRA:
        raise SystemExit(f"The comparator cannot grade this task: {detail}")
    print(1.0 if outcome == text_to_sql_scoring.GradeOutcome.MATCH else 0.0)


if __name__ == "__main__":
    main()
