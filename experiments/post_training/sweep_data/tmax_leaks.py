# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Find TMax environments whose image holds verifier assets the agent can read.

A TMax environment is built from container.def/setup.sh; test_final_state.py runs inside the same container.
An "answer path" is a path created by the build whose name suggests a verifier asset (truth, expected, oracle,
solution, verifier, answer, reference) and that test_final_state.py also reads. It is hidden when the task
instruction names neither the path nor its file name, so the agent has no legitimate reason to use it.
"""

import json
import re
import zipfile

ANSWER_PATH = re.compile(
    r"/[A-Za-z0-9_./-]*(?:truth|expected|oracle|solution|verifier|answer|reference)[A-Za-z0-9_./-]*", re.IGNORECASE
)


def answer_paths(archive: zipfile.ZipFile) -> dict[str, dict[str, list[str]]]:
    """Return {task: {"hidden": [...], "visible": [...]}} for every task with at least one answer path."""
    names = set(archive.namelist())
    tasks = sorted({name.split("/", 1)[0] for name in names if name.endswith("/task.json")})
    found = {}
    for task in tasks:
        build = "".join(
            archive.read(f"{task}/{name}").decode(errors="ignore")
            for name in ("container.def", "setup.sh")
            if f"{task}/{name}" in names
        )
        tests = archive.read(f"{task}/test_final_state.py").decode(errors="ignore")
        created = {path.rstrip("/.") for path in ANSWER_PATH.findall(build)}
        paths = sorted(path for path in created if len(path) > 6 and path in tests)
        if not paths:
            continue
        description = json.loads(archive.read(f"{task}/task.json"))["description"]
        hidden = [path for path in paths if path not in description and path.rsplit("/", 1)[-1] not in description]
        found[task] = {"hidden": hidden, "visible": [path for path in paths if path not in hidden]}
    return found


def references_hidden_asset(row: dict, hidden: list[str]) -> bool:
    """True when any bash tool call in a tmax-sft trajectory mentions one of its task's hidden answer paths."""
    commands = " ".join(
        str(call["function"]["arguments"])
        for message in row["messages"]
        if message["role"] == "assistant"
        for call in message["tool_calls"] or []
    )
    return any(path in commands for path in hidden)
