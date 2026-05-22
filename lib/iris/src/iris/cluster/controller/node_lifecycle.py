# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Controller-level node lifecycle event vocabulary."""

from enum import StrEnum


class NodeLifecycleSource(StrEnum):
    """Where Iris learned about a node lifecycle transition."""

    GCP_TPU_API = "gcp_tpu_api"
    GCP_QUEUED_RESOURCE = "gcp_queued_resource"
    GCP_AUDIT_LOG = "gcp_audit_log"
    IRIS_AUTOSCALER = "iris_autoscaler"
    IRIS_BOOTSTRAP = "iris_bootstrap"
    IRIS_WORKER_HEALTH = "iris_worker_health"


class NodeLifecycleReason(StrEnum):
    """Normalized reason for a node/slice lifecycle event."""

    GCP_PREEMPTION = "gcp_preemption"
    CLOUD_DELETED_OR_MISSING = "cloud_deleted_or_missing"
    CLOUD_DELETING = "cloud_deleting"
    QUEUED_RESOURCE_FAILED = "queued_resource_failed"
    QUEUED_RESOURCE_SUSPENDED = "queued_resource_suspended"
    IRIS_SCALE_DOWN = "iris_scale_down"
    BOOTSTRAP_FAILED = "bootstrap_failed"
    HEARTBEAT_LOST = "heartbeat_lost"
    HEALTH_PROBE_FAILED = "health_probe_failed"


class NodeLifecycleConfidence(StrEnum):
    """How strongly the event reason is supported by the observed signal."""

    REPORTED = "reported"
    CONTROLLER_ACTION = "controller_action"
    INFERRED = "inferred"
    UNKNOWN = "unknown"


NODE_LIFECYCLE_SOURCES = tuple(source.value for source in NodeLifecycleSource)
NODE_LIFECYCLE_REASONS = tuple(reason.value for reason in NodeLifecycleReason)
NODE_LIFECYCLE_CONFIDENCES = tuple(confidence.value for confidence in NodeLifecycleConfidence)
