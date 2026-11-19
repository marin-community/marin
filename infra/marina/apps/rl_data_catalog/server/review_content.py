# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select a task's exact requests and responses from its original reviewer batch."""

import json
from typing import Any


def task_request_evidence(batch: dict) -> dict[str, dict[str, Any]]:
    """Index exact requests and raw provider rows once by the reviewed task IDs."""
    evidence: dict[str, dict[str, Any]] = {}
    for attempt in batch["attempts"]:
        targets = attempt["query_task_ids"]
        requests = attempt["requests"]
        for request in requests["requests"]:
            for task in targets[request["custom_id"]]:
                evidence.setdefault(task, {"input": [], "output": []})["input"].append(
                    {"attempt": attempt["number"], "request": request}
                )
        for observation in requests["observations"]:
            for line in observation["output"].splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                for task in targets[row["custom_id"]]:
                    evidence.setdefault(task, {"input": [], "output": []})["output"].append(
                        {"attempt": attempt["number"], "response": row, "raw": line}
                    )
    return evidence
