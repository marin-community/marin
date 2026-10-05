# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure what shellbox's Iris machine backend does for Taskforge, from inside an Iris task.

Run as a small Iris job from the worktree root (shellbox is not a root workspace member, so it
comes from its source tree):

    uv run iris --cluster=marin job run --no-wait --job-name taskforge-iris-machine-probe \
      --zone us-west4-a --sync-package marin-iris --cpu 1 --memory 2GB \
      -e HF_TOKEN probe-not-a-secret -e WANDB_API_KEY "" -- \
      env PYTHONPATH=lib/shellbox/src python lib/taskforge/scripts/iris_machine_probe.py

Sandboxes are child jobs and inherit the probe job's zone, so the probe job itself is placed with
``job run --zone``: a sandbox-level zone constraint that disagrees with the parent's zone is
unschedulable, and gVisor sandboxes start only in some zones (``us-west4-a`` was observed working).

It prints one JSON object per check on stdout (``PROBE {...}``), so ``iris job logs`` is the
evidence. Checks:

- ``factory_create``: ``IrisMachineFactory.create`` exactly as shipped.
- ``create_latency``: sandboxes created with the shipped submission but a corrected readiness
  poll (``fixed_create``; see docs/upstream/shellbox/iris-machine.patch), sequentially and
  concurrently, plus run/upload/download/close latencies.
- ``network``: DNS and TCP egress inside the sandbox, and whether ``unshare -rn`` gives a command
  an empty network namespace (a way to enforce DENY on a cluster whose sandboxes do have egress).
- ``environment``: which submitter variables reach the sandbox (names and lengths only).
- ``ttl``: a sandbox created with a short ``job_ttl``, used every 30 s until a command fails.
- ``private_image``: a sandbox from an image in the authenticated task registry.
"""

import argparse
import asyncio
import json
import os
import time
import uuid
from dataclasses import replace
from pathlib import Path

from iris.cli.connect import connect_controller
from iris.client import IrisClient
from iris.client.workload import TaskState
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec
from iris.rpc import job_pb2
from iris.rpc.compression import IRIS_RPC_COMPRESSIONS
from iris.rpc.controller_connect import ControllerServiceClientSync
from rigging.timing import Duration
from shellbox.backends.iris import machine as iris_backend
from shellbox.backends.iris.machine import (
    DEFAULT_MEMORY_MB,
    RPC_PADDING_SECONDS,
    IrisMachine,
    IrisMachineFactory,
)
from shellbox.image import RegistryImage
from shellbox.machine import Command, MachineSpec, NetworkPolicy

IMAGE = "docker.io/library/ubuntu:24.04"
PRIVATE_IMAGE = (
    "envreg.208261-marin-gpu.coreweave.app/capability-infra/taskforge-image-build-smoke"
    "@sha256:3ae832dec16b02ff2f6bf0656cfca1449d42a5bce9de955d48e03efea2f903bc"
)
CONTROLLER_URL_ENV = "IRIS_CONTROLLER_URL"
WAITING = (TaskState.PENDING, TaskState.BUILDING, TaskState.ASSIGNED)
NETWORK_SCRIPT = r"""
getent hosts registry-1.docker.io > /dev/null; echo "dns=$?"
timeout 5 bash -c 'exec 3<>/dev/tcp/1.1.1.1/443' 2> /dev/null; echo "tcp_ip=$?"
unshare -rn bash -c "timeout 3 bash -c 'exec 3<>/dev/tcp/1.1.1.1/443' 2> /dev/null; echo unshare_tcp_ip=\$?"
echo "unshare=$?"
unshare -rn bash -c 'echo hi > /tmp/u.txt && cat /tmp/u.txt' ; echo "unshare_fs=$?"
"""


def emit(check: str, **fields) -> None:
    print("PROBE " + json.dumps({"check": check, **fields}, default=str), flush=True)


def fixed_create(factory: IrisMachineFactory, spec: MachineSpec, *, poll: float = 0.5) -> IrisMachine:
    """``IrisMachineFactory._create_sync`` with the readiness poll compared against ``TaskState``."""
    assert isinstance(spec.source, RegistryImage)
    endpoint = connect_controller(cluster_name=factory.cluster, controller_url=factory.controller_url)
    client = IrisClient.remote(endpoint.url, workspace=None, credentials=endpoint.credentials)
    rpc = ControllerServiceClientSync(
        address=endpoint.url,
        timeout_ms=RPC_PADDING_SECONDS * 1000,
        interceptors=endpoint.credentials.interceptors() if endpoint.credentials is not None else [],
        accept_compression=IRIS_RPC_COMPRESSIONS,
        send_compression=None,
    )
    job = client.submit(
        entrypoint=Entrypoint.from_command("sleep", "infinity"),
        name=f"shellbox-{uuid.uuid4().hex}",
        environment=EnvironmentSpec(setup_scripts=[]),
        resources=ResourceSpec(
            cpu=spec.cpus or 1,
            memory=(spec.memory_mb or DEFAULT_MEMORY_MB) * 1024 * 1024,
            disk=(spec.storage_mb or factory.disk_mb) * 1024 * 1024,
        ),
        task_image=spec.source.reference,
        container_profile=job_pb2.CONTAINER_PROFILE_GVISOR,
        scheduling_timeout=Duration.from_seconds(factory.scheduling_timeout),
        timeout=Duration.from_seconds(factory.job_ttl),
        max_retries_failure=0,
        max_retries_preemption=0,
    )
    deadline = time.monotonic() + factory.scheduling_timeout
    try:
        while time.monotonic() < deadline:
            tasks = job.tasks()
            if tasks:
                status = tasks[0].status()
                if status.state == TaskState.RUNNING:
                    emit("placement", job=str(job.job_id), worker=status.worker_id, cluster=status.execution_cluster_id)
                    machine = IrisMachine(endpoint, client, rpc, job, tasks[0].task_id.to_wire(), spec)
                    created = machine._exec_sync(["mkdir", "-p", spec.workdir])
                    if created.exit_code:
                        raise RuntimeError(f"mkdir {spec.workdir} failed: {created.stderr}")
                    return machine
                if status.state not in WAITING:
                    raise RuntimeError(
                        f"sandbox task {status.state} before running on {status.worker_id}: {status.error_message[:400]}"
                    )
            time.sleep(poll)
        raise TimeoutError(f"sandbox did not start within {factory.scheduling_timeout} seconds")
    except BaseException:
        job.cancel()
        client.shutdown()
        endpoint.close()
        raise


async def timed(coro):
    start = time.monotonic()
    result = await coro
    return result, round(time.monotonic() - start, 2)


async def create(factory: IrisMachineFactory, spec: MachineSpec) -> tuple[IrisMachine, float]:
    return await timed(asyncio.to_thread(fixed_create, factory, spec))


async def check_factory_create(factory: IrisMachineFactory, spec: MachineSpec) -> None:
    start = time.monotonic()
    try:
        machine = await factory.create(spec)
    except Exception as error:
        emit("factory_create", ok=False, seconds=round(time.monotonic() - start, 2), error=repr(error)[:500])
        return
    emit("factory_create", ok=True, seconds=round(time.monotonic() - start, 2))
    await machine.close()


async def check_latency(factory: IrisMachineFactory, spec: MachineSpec, sequential: int, concurrent: int) -> None:
    for index in range(sequential):
        start = time.monotonic()
        try:
            machine, seconds = await create(factory, spec)
        except Exception as error:
            emit(
                "create_latency",
                mode="sequential",
                index=index,
                seconds=round(time.monotonic() - start, 2),
                error=repr(error)[:600],
            )
            continue
        _, run_seconds = await timed(machine.run(Command(("true",))))
        _, close_seconds = await timed(machine.close())
        emit("create_latency", mode="sequential", index=index, create=seconds, run=run_seconds, close=close_seconds)
    results = await asyncio.gather(*(create(factory, spec) for _ in range(concurrent)), return_exceptions=True)
    seconds = [r[1] for r in results if not isinstance(r, BaseException)]
    errors = [repr(r)[:300] for r in results if isinstance(r, BaseException)]
    emit("create_latency", mode="concurrent", count=concurrent, create=seconds, errors=errors)
    for item in results:
        if not isinstance(item, BaseException):
            await item[0].close()


async def check_machine(factory: IrisMachineFactory, spec: MachineSpec, tmp: str) -> None:
    machine, seconds = await create(factory, spec)
    try:
        network, network_seconds = await timed(machine.run(Command(("bash", "-c", NETWORK_SCRIPT), timeout=60)))
        emit(
            "network",
            create=seconds,
            seconds=network_seconds,
            stdout=network.stdout.decode(),
            stderr=network.stderr.decode()[-500:],
        )
        names = ("HF_TOKEN", "WANDB_API_KEY", "AWS_SECRET_ACCESS_KEY", "CW_KEY_SECRET", "IRIS_JOB_ENV")
        script = "; ".join(f'printf "{name}=%s\\n" "${{#{name}}}"' for name in names)
        env = await machine.run(Command(("bash", "-c", script)))
        emit("environment", lengths=env.stdout.decode())
        payload = os.urandom(512 * 1024)
        source = f"{tmp}/upload.bin"
        with open(source, "wb") as handle:
            handle.write(payload)
        _, upload_seconds = await timed(machine.upload(Path(source), "/tmp/upload.bin"))
        _, download_seconds = await timed(machine.download("/tmp/upload.bin", Path(f"{tmp}/download.bin")))
        with open(f"{tmp}/download.bin", "rb") as handle:
            round_trip = handle.read() == payload
        runs = []
        for _ in range(5):
            _, run_seconds = await timed(machine.run(Command(("echo", "hi"))))
            runs.append(run_seconds)
        emit("transfer", bytes=len(payload), upload=upload_seconds, download=download_seconds, ok=round_trip, runs=runs)
    finally:
        await machine.close()


async def check_ttl(factory: IrisMachineFactory, spec: MachineSpec, ttl: int) -> None:
    short = IrisMachineFactory(controller_url=factory.controller_url, job_ttl=ttl)
    machine, seconds = await create(short, spec)
    started = time.monotonic()
    first = await machine.run(Command(("echo", "alive")))
    emit("ttl", phase="before", create=seconds, exit=first.exit_code, stdout=first.stdout.decode().strip())
    await asyncio.sleep(max(0, ttl + 30 - (time.monotonic() - started)))
    while True:
        state = await asyncio.to_thread(machine.job.status)
        elapsed = round(time.monotonic() - started, 1)
        try:
            second = await machine.run(Command(("echo", "alive")))
            emit("ttl", phase="after", elapsed=elapsed, job_state=state.state, exit=second.exit_code)
        except Exception as error:
            emit("ttl", phase="after", elapsed=elapsed, job_state=state.state, error=repr(error)[:300])
            break
        if elapsed > ttl + 600:
            break
        await asyncio.sleep(30)
    await machine.close()


async def check_private_image(factory: IrisMachineFactory) -> None:
    spec = MachineSpec(source=RegistryImage(PRIVATE_IMAGE), network=NetworkPolicy.ALLOW)
    short = IrisMachineFactory(controller_url=factory.controller_url, scheduling_timeout=300)
    start = time.monotonic()
    try:
        machine = await asyncio.to_thread(fixed_create, short, spec)
    except Exception as error:
        emit("private_image", ok=False, seconds=round(time.monotonic() - start, 2), error=repr(error)[:800])
        return
    result = await machine.run(Command(("/opt/task/run.sh",)))
    emit("private_image", ok=True, seconds=round(time.monotonic() - start, 2), stdout=result.stdout.decode())
    await machine.close()


async def guarded(name: str, coro) -> None:
    """Report a failed check and continue: this script's output is the evidence."""
    try:
        await coro
    except Exception as error:
        emit(name, error=repr(error)[:800])


async def main(args: argparse.Namespace) -> None:
    if args.chunk_bytes:
        iris_backend.TRANSFER_CHUNK_BYTES = args.chunk_bytes
    factory = IrisMachineFactory(controller_url=os.environ[CONTROLLER_URL_ENV])
    spec = MachineSpec(source=RegistryImage(IMAGE), network=NetworkPolicy.ALLOW)
    emit(
        "start",
        controller_url=factory.controller_url,
        image=IMAGE,
        chunk_bytes=iris_backend.TRANSFER_CHUNK_BYTES,
    )
    await check_factory_create(factory, spec)
    await check_factory_create(factory, replace(spec, network=NetworkPolicy.DENY))
    await check_latency(factory, spec, args.sequential, args.concurrent)
    for attempt in range(2):
        await guarded(f"machine_attempt_{attempt}", check_machine(factory, spec, args.tmp))
    await guarded("private_image", check_private_image(factory))
    await guarded("ttl", check_ttl(factory, spec, args.ttl))
    emit("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequential", type=int, default=6)
    parser.add_argument("--concurrent", type=int, default=8)
    parser.add_argument("--ttl", type=int, default=90)
    parser.add_argument("--tmp", default="/tmp")
    parser.add_argument("--chunk-bytes", type=int, default=None, help="override shellbox TRANSFER_CHUNK_BYTES")
    asyncio.run(main(parser.parse_args()))
