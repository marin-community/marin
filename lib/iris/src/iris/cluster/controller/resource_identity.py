# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical resource operations over controller state and backend observations."""

import uuid

from rigging.timing import Timestamp

from iris.cluster.controller.persistence import reads
from iris.cluster.federation.protocol import FederationDirection
from iris.cluster.types import LOCAL_CLUSTER
from iris.resources.identity import (
    JobIdentity,
    ResourceKey,
    ResourceKind,
)
from iris.resources.names import JobName

_RESOURCE_UID_NAMESPACE = uuid.UUID("2c72b7f4-a156-5d27-8b58-7de28d5ec4cc")
_RESOURCE_UID_PREFIX = "iris-resource-v2"


def _uid(kind: ResourceKind, *parts: object) -> str:
    name = "\0".join((_RESOURCE_UID_PREFIX, kind.value, *(str(part) for part in parts)))
    return str(uuid.uuid5(_RESOURCE_UID_NAMESPACE, name))


def _job_uid(
    cluster_id: str,
    job_id: JobName,
    submitted_at: Timestamp,
    *,
    handoff_nonce: str = "",
) -> str:
    incarnation = handoff_nonce if job_id.is_root and handoff_nonce else submitted_at.epoch_ms()
    return _uid(ResourceKind.JOB, cluster_id, job_id.to_wire(), incarnation)


def _task_uid(job_uid: str, task_id: JobName) -> str:
    _, task_index = task_id.require_task()
    return _uid(ResourceKind.TASK, job_uid, task_index)


def _authority_cluster(
    cluster_id: str,
    row: reads.JobCoordinates | reads.JobRecord,
) -> str:
    if row.direction == int(FederationDirection.RECEIVED):
        return str(row.peer_id)
    return cluster_id


def _job_identity(
    cluster_id: str,
    row: reads.JobCoordinates | reads.JobRecord,
) -> JobIdentity:
    authority = _authority_cluster(cluster_id, row)
    return JobIdentity(
        ResourceKey(authority, ResourceKind.JOB, row.job_id.to_wire()),
        _job_uid(
            authority,
            row.job_id,
            row.submitted_at_ms,
            handoff_nonce=str(row.handoff_nonce or ""),
        ),
    )


def _execution_cluster(cluster_id: str, stored: str) -> str:
    return cluster_id if stored == LOCAL_CLUSTER else stored


def _opaque_uid(value: str) -> str:
    return uuid.uuid5(_RESOURCE_UID_NAMESPACE, value).hex
