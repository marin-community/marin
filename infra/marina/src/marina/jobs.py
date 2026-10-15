# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute application jobs from a snapshot of released packages."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from uuid import UUID

from sqlalchemy.engine import Engine

from marina.applets import AppletStore, applet_role, applet_schema, materialize_package
from marina.db import database_from_env, engine_for_role
from marina.runtime import RuntimeConfig

JOB_APPLET_ENV = "MARINA_JOB_APPLET_ID"
JOB_SCHEMA_ENV = "MARINA_JOB_SCHEMA"
JOB_VERSION_ENV = "MARINA_JOB_VERSION"


def job_engine() -> Engine:
    """Connect as the role of the immutable package selected by the job runner."""
    database = database_from_env(os.environ)
    if database is None:
        raise RuntimeError("No database configured for the application job")
    applet_id = UUID(os.environ[JOB_APPLET_ENV])
    return engine_for_role(database, os.environ[JOB_SCHEMA_ENV], applet_role(applet_id))


def execute_jobs(store: AppletStore, runtime: RuntimeConfig, runner_name: str) -> None:
    """Snapshot and materialize every selected module before running any child process."""
    runner = runtime.runner(runner_name)
    selected = [
        (snapshot, job)
        for snapshot in store.snapshots(runner.name)
        for job in snapshot.package.manifest.jobs
        if job.runner == runner.name
    ]
    for snapshot, job in selected:
        runtime.validate_jobs(snapshot.applet_id, (job,))
    if sum(job.timeout for _, job in selected) > runner.timeout:
        raise ValueError(f"Released jobs exceed runner {runner.name}'s total timeout")
    failures = []
    with tempfile.TemporaryDirectory(prefix="marina-jobs-") as directory:
        prepared = []
        for snapshot, job in selected:
            root = Path(directory) / f"applet_{snapshot.applet_id.hex}_v{snapshot.version}"
            materialize_package(root, snapshot.package)
            environment = dict(os.environ)
            for secret in set(runner.secrets) - set(job.secrets):
                environment.pop(secret, None)
            manifest = snapshot.package.manifest
            environment[JOB_APPLET_ENV] = str(snapshot.applet_id)
            environment[JOB_VERSION_ENV] = str(snapshot.version)
            environment[JOB_SCHEMA_ENV] = (
                manifest.name.replace("-", "_") if manifest.name else applet_schema(snapshot.applet_id)
            )
            prepared.append((snapshot, job, root, environment))
        for snapshot, job, root, environment in prepared:
            qualified = f"{snapshot.applet_id}/v/{snapshot.version}:{job.name}"
            try:
                result = subprocess.run(
                    [sys.executable, "-I", "-m", "marina.applet_job", str(root), job.module, *job.args],
                    cwd=root,
                    env=environment,
                    timeout=job.timeout,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as error:
                failures.append(f"{qualified}: {error}")
                continue
            if result.returncode:
                failures.append(f"{qualified}: exit {result.returncode}")
    if failures:
        raise RuntimeError("Job failures: " + "; ".join(failures))
