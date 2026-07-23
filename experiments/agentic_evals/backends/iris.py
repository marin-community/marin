# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""IrisBackend — adapter for Marin's Iris TPU/GPU cluster.

Thin wrapper around ``iris.client.IrisClient`` that builds the task spec
(entrypoint, resources, environment) and submits an eval job. Adapted from
OT-Agent ``hpc/iris/launcher.py`` — captures the essential task-spec
construction without the full region-discovery / output-mode machinery.

Requires the ``[iris]`` extra (``iris`` client + ``rigging``).
"""

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


class IrisBackend:
    """Submit eval jobs to a Marin Iris cluster via IrisClient."""

    def __init__(
        self,
        *,
        workspace: Optional[Path] = None,
        cluster_config: Optional[str] = None,
        task_image: Optional[str] = None,
    ):
        self.workspace = Path(workspace) if workspace else Path.cwd()
        self.cluster_config = cluster_config
        self.task_image = task_image

    def _resolve_controller(self, cluster_config_path: str):
        """Connect to the Iris controller and return the client and endpoint handle."""
        from iris.cli.connect import connect_controller
        from iris.client import IrisClient

        endpoint = connect_controller(config_file=Path(cluster_config_path))
        client = IrisClient.remote(endpoint.url, workspace=self.workspace, credentials=endpoint.credentials)
        return client, endpoint, endpoint.url

    def submit(
        self,
        *,
        command: List[str],
        job_name: str,
        env_vars: Dict[str, str],
        accelerator: str = "v6e-4",
        replicas: int = 1,
        cpu: float = 8.0,
        memory: str = "256GB",
        disk: str = "100GB",
        task_image: Optional[str] = None,
        priority: str = "interactive",
        max_retries: int = 0,
        timeout: int = 0,
        secrets_env: Optional[str] = None,
        dry_run: bool = False,
        no_wait: bool = False,
        target_cluster: Optional[str] = None,
    ) -> Any:
        """Submit an eval job to Iris.

        Args:
            command: The worker command (e.g. ``["python", "run_eval.py", ...]``).
            job_name: Unique job name.
            env_vars: Environment variables for the task container.
            accelerator: TPU/GPU spec (e.g. ``"v6e-4"``, ``"H100x8"``).
            replicas: Number of task replicas (1 per VM for multi-host TPU).
            cpu: CPU cores for the entrypoint task.
            memory: Memory spec (e.g. ``"256GB"``).
            disk: Ephemeral disk (e.g. ``"100GB"``).
            task_image: Container image override.
            priority: Priority band: ``"production"``, ``"interactive"``, ``"batch"``.
            max_retries: Max retries on failure.
            timeout: Job timeout in seconds (0 = no timeout).
            secrets_env: Path to a KEY=VALUE env file to load.
            dry_run: If True, print but don't submit.
            no_wait: If True, submit and detach (don't stream logs).

        Returns:
            Exit code (0 on success) or job handle (if no_wait).
        """
        # Load secrets env into os.environ on the launch host
        if secrets_env:
            self._load_secrets_env(secrets_env, env_vars)

        image = task_image or self.task_image

        print(f"[iris] Job:        {job_name}", flush=True)
        print(f"[iris] Accelerator: {accelerator}", flush=True)
        print(f"[iris] Image:      {image or '(default)'}", flush=True)
        print(f"[iris] Command:    {shlex.join(command)}", flush=True)

        if dry_run:
            print("[iris] --dry-run: not submitting", flush=True)
            return 0

        from iris.cli.job import (
            build_job_constraints,
            build_resources,
            build_tpu_alternatives,
            parse_gpu_spec,
            resolve_multinode_defaults,
        )
        from iris.cluster.types import Entrypoint, EnvironmentSpec
        from iris.rpc import job_pb2

        cluster_config_path = self.cluster_config
        if not cluster_config_path:
            raise ValueError("IrisBackend requires cluster_config (path to iris cluster YAML)")

        client, tunnel_ctx, _ = self._resolve_controller(cluster_config_path)

        try:
            entrypoint = Entrypoint.from_command(*command)

            try:
                parse_gpu_spec(accelerator)
                gpu, tpu = accelerator, None
            except ValueError:
                gpu, tpu = None, accelerator
            resources = build_resources(tpu=tpu, gpu=gpu, cpu=cpu, memory=memory, disk=disk)
            replicas, coscheduling = resolve_multinode_defaults(tpu=tpu, gpu=gpu, replicas=replicas)
            constraints = build_job_constraints(
                resources_proto=resources.to_proto(),
                tpu_variants=build_tpu_alternatives(tpu),
                replicas=replicas,
                regions=None,
                zone=None,
                preemptible=None,
                target_cluster=target_cluster,
            )

            _PRIO = {
                "production": job_pb2.PRIORITY_BAND_PRODUCTION,
                "interactive": job_pb2.PRIORITY_BAND_INTERACTIVE,
                "batch": job_pb2.PRIORITY_BAND_BATCH,
            }
            priority_band = _PRIO.get(priority, job_pb2.PRIORITY_BAND_UNSPECIFIED)

            task_env_vars = dict(env_vars)
            extras = task_env_vars.pop("_iris_extras", None) or ["datagen-tpu"]

            job = client.submit(
                entrypoint=entrypoint,
                name=job_name,
                resources=resources,
                environment=EnvironmentSpec(env_vars=task_env_vars, extras=extras),
                constraints=constraints,
                coscheduling=coscheduling,
                replicas=replicas,
                max_retries_failure=max_retries,
                task_image=image,
                priority_band=priority_band,
                timeout=self._duration(timeout) if timeout else None,
            )

            full_job_id = str(job.job_id)
            print(f"[iris] Submitted: {full_job_id}", flush=True)

            if no_wait:
                return job

            try:
                status = job.wait(stream_logs=True, timeout=float("inf"))
                return 0 if status.state == job_pb2.JOB_STATE_SUCCEEDED else 1
            except KeyboardInterrupt:
                print(f"[iris] Terminating job {full_job_id}...", file=sys.stderr, flush=True)
                client.terminate_job(job.job_id)
                return 130
        finally:
            tunnel_ctx.__exit__(None, None, None)

    def _load_secrets_env(self, secrets_env: str, env_vars: Dict[str, str]) -> None:
        """Load KEY=VALUE lines from a secrets file into env_vars."""
        path = Path(secrets_env).expanduser()
        if not path.exists():
            return
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :]
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("'").strip('"')
                if key:
                    os.environ[key] = value
                    env_vars.setdefault(key, value)

    @staticmethod
    def _duration(secs: int):
        from rigging.timing import Duration

        return Duration.from_seconds(secs)

    def query(self, job_id: str) -> Any:
        """Query job status. Requires the iris client."""
        raise NotImplementedError("IrisBackend.query requires a live controller connection")

    def logs(self, job_id: str, *, follow: bool = True) -> Any:
        """Stream job logs. Requires the iris client."""
        raise NotImplementedError("IrisBackend.logs requires a live controller connection")


__all__ = ["IrisBackend"]
