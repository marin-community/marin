# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The finelog client and tables shared across the controller's components.

Built once, before the task backend and autoscaler, so the finelog ``Table``
handles those components write to are constructor arguments rather than
post-construction injections. The stack owns the optional in-process log server
(started in tests and local mode when no external server is configured) so the
controller can tear everything down through a single :meth:`LogStack.close`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from finelog.client import LogClient
from finelog.client.log_client import Table
from finelog.deploy.config import INTRA_CLUSTER_CIDRS, CidrAuthLayer, auth_policy_json
from finelog.embedded import require_embedded_server
from rigging.auth import BearerTokenInjector, TokenProvider

from iris.cluster.platforms.types import resolve_external_host
from iris.cluster.stats.tables import (
    JOB_CONFIG_NAMESPACE,
    JOB_CONFIG_STORAGE_POLICY,
    JOB_NAMESPACE,
    JOB_STORAGE_POLICY,
    JOB_TASK_NAMESPACE,
    JOB_TASK_STORAGE_POLICY,
    PROFILE_NAMESPACE,
    PROVISIONING_NAMESPACE,
    TASK_ATTEMPT_NAMESPACE,
    TASK_ATTEMPT_STORAGE_POLICY,
    TASK_EVENT_NAMESPACE,
    TASK_EVENT_STORAGE_POLICY,
    TASK_STATE_NAMESPACE,
    TASK_STATS_NAMESPACE,
    WORKER_STATS_NAMESPACE,
    IrisJob,
    IrisJobConfig,
    IrisJobTask,
    IrisProfile,
    IrisProvisioning,
    IrisTaskAttempt,
    IrisTaskStat,
    IrisTaskState,
    IrisWorkerStat,
    TaskEventRow,
)

logger = logging.getLogger(__name__)


@dataclass
class LogStack:
    """A finelog ``LogClient`` plus the tables the controller's components write to.

    ``server`` holds the in-process ``finelog_server`` when one was started for
    this stack (tests and local mode); it is ``None`` when connecting to an
    externally-hosted server.
    """

    client: LogClient
    address: str
    task_stats_table: Table
    task_event_table: Table
    profile_table: Table
    provisioning_table: Table
    # iris.worker rows are emitted by worker-daemon clusters.
    worker_stats_table: Table
    # iris.task_state rows from the controller's periodic per-root-job aggregate.
    task_state_table: Table
    # iris.controller_db.*: durable copies of the controller DB's jobs, job_config, tasks and task_attempts.
    job_table: Table
    job_config_table: Table
    job_task_table: Table
    task_attempt_table: Table
    server: Any = None

    def close(self) -> None:
        self.client.close()
        if self.server is not None:
            self.server.stop()


def build_log_stack(
    *,
    log_service_address: str,
    local_log_dir: Path,
    host: str,
    worker_token_provider: TokenProvider | None,
) -> LogStack:
    """Connect to the log service (starting an in-process server if needed) and resolve tables.

    When ``log_service_address`` is empty, start a bundled native ``finelog_server``
    under ``local_log_dir`` and connect to it; otherwise connect to the externally
    hosted server. ``worker_token_provider``, when set, is attached as a bearer token so the
    log server accepts controller-originated PushLogs/FetchLogs.
    """
    server = None
    address = log_service_address
    if not address:
        # The embedded fallback is a real network endpoint: this controller and its
        # workers reach it over the pod network (a non-loopback in-cluster address),
        # and a port-forward reaches it over loopback. finelog is default-deny and an
        # unset policy trusts loopback only, so without a cidr layer it would reject
        # every non-loopback in-cluster client. Trust the private network + loopback,
        # the same posture a real finelog deploy uses.
        auth_policy = auth_policy_json((CidrAuthLayer(cidrs=INTRA_CLUSTER_CIDRS),))
        server = require_embedded_server()(log_dir=str(local_log_dir), host=host, auth_policy=auth_policy)
        address = f"http://{resolve_external_host(host)}:{server.port}"
        logger.info("Local log server ready at %s (log_dir=%s)", address, local_log_dir)

    interceptors = (BearerTokenInjector(worker_token_provider, "authorization"),) if worker_token_provider else ()
    client = LogClient.connect(address, interceptors=interceptors)
    return LogStack(
        client=client,
        address=address,
        task_stats_table=client.get_table(TASK_STATS_NAMESPACE, IrisTaskStat),
        task_event_table=client.get_table(TASK_EVENT_NAMESPACE, TaskEventRow, storage_policy=TASK_EVENT_STORAGE_POLICY),
        profile_table=client.get_table(PROFILE_NAMESPACE, IrisProfile),
        provisioning_table=client.get_table(PROVISIONING_NAMESPACE, IrisProvisioning),
        worker_stats_table=client.get_table(WORKER_STATS_NAMESPACE, IrisWorkerStat),
        task_state_table=client.get_table(TASK_STATE_NAMESPACE, IrisTaskState),
        job_table=client.get_table(JOB_NAMESPACE, IrisJob, storage_policy=JOB_STORAGE_POLICY),
        job_config_table=client.get_table(JOB_CONFIG_NAMESPACE, IrisJobConfig, storage_policy=JOB_CONFIG_STORAGE_POLICY),
        job_task_table=client.get_table(JOB_TASK_NAMESPACE, IrisJobTask, storage_policy=JOB_TASK_STORAGE_POLICY),
        task_attempt_table=client.get_table(
            TASK_ATTEMPT_NAMESPACE, IrisTaskAttempt, storage_policy=TASK_ATTEMPT_STORAGE_POLICY
        ),
        server=server,
    )
