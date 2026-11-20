# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check frozen TaskSpecs with the selected runtime's actual loader and registry.

Run in a subprocess with PYTHONPATH set to the pinned runtime archive. Input and
output are JSON lines; output contains counts and schema paths, never task text.
"""

import json
import sys
from collections import Counter

from pydantic import ValidationError
from rolloutengine.harness.assistant_prefix import AssistantPrefixHarness
from skyrl_gym.rollout_specs import TOOL_PROVIDERS, rollout_spec, shell_harness
from taskcompendium.models import TaskSpec, VerifyitGrader, judge_spec
from verifyit.grade import grader_for


def check_task(serialized: str, session: dict, judges: frozenset[str]) -> tuple[set[str], list[str]]:
    document = json.loads(serialized)
    failures: set[str] = set()
    fields: list[str] = []
    # Registry failures remain visible even when this runtime rejects the schema.
    for requirement in document["environment_requirements"].get("tool_providers", {}).values():
        if requirement["action_interface"] not in TOOL_PROVIDERS:
            failures.add("CONSUMER_TOOL_UNREGISTERED")
    try:
        task = TaskSpec.model_validate_json(serialized)
    except ValidationError as error:
        failures.add("CONSUMER_SCHEMA_REJECTED")
        fields = [".".join(map(str, item["loc"])) for item in error.errors(include_input=False)]
        return failures, fields
    judge = judge_spec(task.grader)
    if judge is not None and judge.model not in judges:
        failures.add("CONSUMER_JUDGE_UNREGISTERED")
    if isinstance(task.grader, VerifyitGrader):
        try:
            grader_for(task.grader.mode)
        except (ValueError, ImportError, KeyError):
            failures.add("CONSUMER_JUDGE_UNREGISTERED")
    try:
        lowered = rollout_spec(task, session)
        if session["task_session"] == "assistant_prefix":
            AssistantPrefixHarness().check(task, lowered.limits)
        else:
            shell_harness(lowered.harness).check(task, lowered.limits)
    except (ValueError, NotImplementedError, KeyError):
        failures.add("CONSUMER_LOADER_REJECTED")
    return failures, fields


def main() -> None:
    config = json.loads(sys.stdin.readline())
    failures: Counter[str] = Counter()
    fields: Counter[str] = Counter()
    checked = failed = 0
    for line in sys.stdin:
        reasons, rejected = check_task(json.loads(line), config["session"], frozenset(config["judges"]))
        checked += 1
        failed += bool(reasons)
        failures.update(reasons)
        fields.update(rejected)
    print(json.dumps({"checked_tasks": checked, "failed_tasks": failed, "failures": failures, "schema_paths": fields}))


if __name__ == "__main__":
    main()
