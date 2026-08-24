# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared Kubernetes task-output uploader contract."""

from collections.abc import Mapping

from rigging.timing import Duration

from iris.cluster.config import TaskOutputDestination, TaskOutputPolicy

OUTPUT_CONTAINER_NAME = "output-uploader"
OUTPUT_CONTROL_VOLUME_NAME = "output-control"
OUTPUT_CONTROL_PATH = "/iris/output-control"
OUTPUT_RELEASE_PATH = f"{OUTPUT_CONTROL_PATH}/release"

TASK_ID_ENV = "IRIS_TASK_ID"
ATTEMPT_UID_ENV = "IRIS_ATTEMPT_UID"
DESTINATION_ENV = "IRIS_TASK_OUTPUT_DESTINATION"
TTL_DAYS_ENV = "IRIS_TASK_OUTPUT_TTL_DAYS"
MAX_BYTES_ENV = "IRIS_TASK_OUTPUT_MAX_BYTES"
MAX_ENTRIES_ENV = "IRIS_TASK_OUTPUT_MAX_ENTRIES"
TIMEOUT_ENV = "IRIS_TASK_OUTPUT_TIMEOUT_SECONDS"
SOURCE_PREFIX_ENV = "IRIS_TASK_OUTPUT_SOURCE_PREFIX"


def output_uploader_environment(
    task_id_wire: str,
    attempt_uid: str,
    policy: TaskOutputPolicy,
    source_prefix: str | None,
) -> list[dict[str, object]]:
    """Encode cluster output policy for the uploader container."""
    env: list[dict[str, object]] = [
        {"name": TASK_ID_ENV, "value": task_id_wire},
        {"name": ATTEMPT_UID_ENV, "value": attempt_uid},
        {"name": DESTINATION_ENV, "value": str(policy.destination)},
        {"name": TTL_DAYS_ENV, "value": str(policy.ttl_days)},
        {"name": MAX_BYTES_ENV, "value": str(policy.max_bytes)},
        {"name": MAX_ENTRIES_ENV, "value": str(policy.max_entries)},
        {"name": TIMEOUT_ENV, "value": str(policy.finalization_timeout.to_seconds())},
    ]
    if source_prefix:
        env.append({"name": SOURCE_PREFIX_ENV, "value": source_prefix})
    return env


def output_policy_from_environment(environment: Mapping[str, str]) -> TaskOutputPolicy:
    """Decode uploader policy from its container environment."""
    return TaskOutputPolicy(
        destination=TaskOutputDestination(environment[DESTINATION_ENV]),
        ttl_days=int(environment[TTL_DAYS_ENV]),
        max_bytes=int(environment[MAX_BYTES_ENV]),
        max_entries=int(environment[MAX_ENTRIES_ENV]),
        finalization_timeout=Duration.from_seconds(float(environment[TIMEOUT_ENV])),
    )
