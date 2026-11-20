# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check bespoke needs in the pinned consumer; report global schema drift separately."""

import json
import sys
from collections import Counter

from pydantic import ValidationError
from rolloutengine.grading import judge_connection
from skyrl_gym.rollout_specs import rollout_spec, shell_harness
from taskcompendium.models import TaskSpec
from verifyit.modes.grade_judge import JudgeConnection


def check_task(record: dict, config: dict) -> tuple[set[str], list[str]]:
    failures: set[str] = set()
    fields: list[str] = []
    judge_binding = config.get("judge_binding")
    try:
        task = TaskSpec.model_validate_json(record["task"])
    except ValidationError as error:
        task = None
        fields = [".".join(map(str, item["loc"])) for item in error.errors(include_input=False)]
    lowered = None
    if task is not None and record["needs"]:
        try:
            lowered = rollout_spec(task, config["session"])
            if config["session"]["task_session"] == "shellbox" and "tool_provider" in record["needs"]:
                shell_harness(lowered.harness, retrieval_service=config["retrieval_service"]).check(task, lowered.limits)
        except (ValueError, NotImplementedError, KeyError):
            failures.add(
                "CONSUMER_TOOL_UNREGISTERED" if "tool_provider" in record["needs"] else "CONSUMER_EXECUTION_UNSUPPORTED"
            )
    if "custom_judge" in record["needs"]:
        model = record["judge_model"] or config["default_judge"]
        if not (model and judge_binding and judge_binding.get("base_url") and judge_binding.get("api_key_env")):
            failures.add("CONSUMER_JUDGE_UNREGISTERED")
        elif lowered is not None:
            try:
                judge_connection(
                    lowered.model_copy(
                        update={
                            "judge_connection": JudgeConnection(
                                judge_binding["base_url"],
                                "consumer-binding-probe",
                                default_model=config["default_judge"],
                            )
                        }
                    )
                )
            except ValueError:
                failures.add("CONSUMER_JUDGE_UNREGISTERED")
    if "execution" in record["needs"]:
        failures.add("CONSUMER_EXECUTION_UNSUPPORTED")
    if "service" in record["needs"] and not config["retrieval_service"]:
        failures.add("CONSUMER_SERVICE_UNREGISTERED")
    return failures, fields


def main() -> None:
    config = json.loads(sys.stdin.readline())
    failures: Counter[str] = Counter()
    fields: Counter[str] = Counter()
    needs: Counter[str] = Counter()
    checked = failed = bespoke = schema_rejected = 0
    for line in sys.stdin:
        record = json.loads(line)
        reasons, rejected = check_task(record, config)
        checked += 1
        bespoke += bool(record["needs"])
        failed += bool(reasons)
        schema_rejected += bool(rejected)
        failures.update(reasons)
        fields.update(rejected)
        needs.update(record["needs"])
    print(
        json.dumps(
            {
                "checked_tasks": checked,
                "bespoke_tasks": bespoke,
                "failed_tasks": failed,
                "failures": failures,
                "requirements": needs,
                "global_schema_rejected_tasks": schema_rejected,
                "schema_paths": fields,
            }
        )
    )


if __name__ == "__main__":
    main()
