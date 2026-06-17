#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Allocate and use development CoreWeave H100 pods on Iris-managed clusters."""

from __future__ import annotations

import getpass
import json
import logging
import os
import shlex
import subprocess
import time
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import click
from iris.client import IrisClient, JobAlreadyExists
from iris.cluster.backends.k8s.tasks import _LABEL_TASK_ID, _sanitize_label_value
from iris.cluster.config import IrisConfig
from iris.cluster.types import Entrypoint, JobName, ResourceSpec, gpu_device
from iris.rpc import config_pb2, job_pb2

logger = logging.getLogger(__name__)

HOLDER_COMMAND = (
    "import signal, sys, time; "
    "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0)); "
    "signal.signal(signal.SIGINT, lambda *_: sys.exit(0)); "
    "print('iris dev coreweave holder ready', flush=True); "
    "time.sleep(365 * 24 * 60 * 60)"
)

STATE_DIR = Path.home() / ".cache" / "marin" / "dev_coreweave_iris"
DEFAULT_GPU_COUNT = 8
TASK_CONTAINER = "task"
GPU_VARIANT = "H100"

TERMINAL_JOB_STATES = {
    job_pb2.JOB_STATE_FAILED,
    job_pb2.JOB_STATE_KILLED,
    job_pb2.JOB_STATE_UNSCHEDULABLE,
    job_pb2.JOB_STATE_WORKER_FAILED,
}
INACTIVE_JOB_STATES = TERMINAL_JOB_STATES | {job_pb2.JOB_STATE_SUCCEEDED}


@dataclass(frozen=True)
class PodRef:
    """The k8s pod backing a dev CoreWeave session."""

    namespace: str
    pod_name: str
    container: str = TASK_CONTAINER


@dataclass(frozen=True)
class CoreweaveTarget:
    """Cluster-level kubectl target. Empty kubeconfig_path => kubectl default resolution."""

    namespace: str
    kubeconfig_path: str


@dataclass(frozen=True)
class DevCoreweaveState:
    """Persisted local state for an active dev CoreWeave session."""

    session_name: str
    config_file: str
    job_id: str
    gpu_count: int
    target: CoreweaveTarget
    pod: PodRef

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> DevCoreweaveState:
        data = json.loads(raw)
        return cls(
            session_name=data["session_name"],
            config_file=data["config_file"],
            job_id=data["job_id"],
            gpu_count=data["gpu_count"],
            target=CoreweaveTarget(**data["target"]),
            pod=PodRef(**data["pod"]),
        )
