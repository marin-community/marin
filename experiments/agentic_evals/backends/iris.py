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
        """Tunnel to the iris controller and return (client, controller_url)."""
        from iris.client import IrisClient
        from iris.cluster.config import load_config
        from iris.cluster.composer import provider_bundle
        from iris.cluster.local_cluster import LocalCluster
        from iris.cli.main import client_credentials, resolve_cluster_name

        config = load_config(cluster_config_path)
        cluster_name = resolve_cluster_name(config, None, Path(cluster_config_path).stem)
        credentials = client_credentials(config, cluster_name)
        bundle = provider_bundle(config)
        if config.controller.controller_kind() == "local":
            local_cluster = LocalCluster(config)
            controller_address = local_cluster.start()
        else:
            controller_address = (
                config.controller_address()
                or bundle.controller.discover_controller(config.controller)
            )
        tunnel_ctx = bundle.controller.tunnel(address=controller_address)
        controller_url = tunnel_ctx.__enter__()
        client = IrisClient.remote(controller_url, workspace=self.workspace, credentials=credentials)
        return client, tunnel_ctx, controller_url

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

        from iris.cluster.types import EnvironmentSpec, Entrypoint
        from iris.cli.job import build_job_constraints
        from iris.rpc import job_pb2

        cluster_config_path = self.cluster_config
        if not cluster_config_path:
            raise ValueError("IrisBackend requires cluster_config (path to iris cluster YAML)")

        client, tunnel_ctx, _ = self._resolve_controller(cluster_config_path)

        try:
            # Build resources from the accelerator spec. This is a simplified
            # adapter — the full OT-Agent launcher resolves TPU topology +
            # chip counts; here we pass the spec through to iris's resource
            # builder via the client API.
            entrypoint = Entrypoint.from_command(*command)

            # Build constraints (simplified — real launcher resolves TPU variants
            # and multinode coscheduling from the accelerator spec).
            constraints = build_job_constraints(
                resources_proto=None,
                tpu_variants=[accelerator] if not accelerator.startswith("H100") else [],
                replicas=replicas,
                regions=None,
                zone=None,
                preemptible=None,
            )

            _PRIO = {
                "production": job_pb2.PRIORITY_BAND_PRODUCTION,
                "interactive": job_pb2.PRIORITY_BAND_INTERACTIVE,
                "batch": job_pb2.PRIORITY_BAND_BATCH,
            }
            priority_band = _PRIO.get(priority, job_pb2.PRIORITY_BAND_UNSPECIFIED)

            extras = env_vars.pop("_iris_extras", None) or ["datagen-tpu"]

            job = client.submit(
                entrypoint=entrypoint,
                name=job_name,
                resources=None,
                environment=EnvironmentSpec(env_vars=env_vars, extras=extras),
                constraints=constraints,
                coscheduling=None,
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
                line = line[len("export "):]
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
