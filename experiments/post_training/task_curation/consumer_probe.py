# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check bespoke needs in the pinned consumer; report global schema drift separately."""

import json
import sys
from collections import Counter

from pydantic import ValidationError
from skyrl_gym.rollout_specs import TOOL_PROVIDERS
from taskcompendium.models import ProviderRequirement, TaskSpec


def check_task(record: dict, config: dict) -> tuple[set[str], list[str]]:
    document = json.loads(record["task"])
    failures: set[str] = set()
    fields: list[str] = []
    for requirement in document["environment_requirements"].get("tool_providers", {}).values():
        interface = requirement["action_interface"]
        if interface not in TOOL_PROVIDERS:
            failures.add("CONSUMER_TOOL_UNREGISTERED")
            continue
        try:
            TOOL_PROVIDERS[interface].tools(ProviderRequirement.model_validate(requirement))
        except (ValidationError, ValueError, NotImplementedError, KeyError):
            failures.add("CONSUMER_TOOL_UNREGISTERED")
    if "custom_judge" in record["needs"]:
        model = record["judge_model"] or config["default_judge"]
        if not model or model not in config["judges"]:
            failures.add("CONSUMER_JUDGE_UNREGISTERED")
    if "execution" in record["needs"]:
        failures.add("CONSUMER_EXECUTION_UNSUPPORTED")
    if "service" in record["needs"] and not config["session"].get("retrieval_service"):
        failures.add("CONSUMER_SERVICE_UNREGISTERED")
    try:
        TaskSpec.model_validate_json(record["task"])
    except ValidationError as error:
        fields = [".".join(map(str, item["loc"])) for item in error.errors(include_input=False)]
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
