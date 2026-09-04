# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Runtime capabilities consumed by controller operations."""

from typing import Protocol

from iris.cluster.controller import admin, attempts, backend_status, federation_service, jobs, tasks, workers


class ControllerRuntime(
    admin.CheckpointRuntime,
    attempts.AttemptRuntime,
    backend_status.BackendStatusRuntime,
    federation_service.FederationRuntime,
    jobs.JobRuntime,
    tasks.TaskRuntime,
    workers.WorkerRuntime,
    Protocol,
):
    """Combined structural type accepted by the RPC composition root."""
