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
evidence. Every sandbox comes from ``IrisMachineFactory.create`` as shipped. Checks:

- ``factory_create``: create and close latency under each network policy.
- ``network``: DNS and TCP egress inside a sandbox under each network policy.
- ``environment``: which submitter variables reach the sandbox (names and lengths only).
- ``transfer``: upload and download of 512 KiB at the transfer chunk size (``--chunk-bytes``), and
  per-command latency.
- ``ttl``: a sandbox created with a short ``job_ttl``, used every 30 s until a command fails;
  ``terminated`` says whether the failure was shellbox's typed ``MachineTerminated``.
- ``private_image``: a sandbox from an image in the authenticated task registry.
- ``grader_base``: with ``--grader-image`` (``taskforge.sandbox.images.GRADER_BASE_IMAGE``), a sandbox from the
  script-grader base image: its user, and whether ``python3``, ``setsid`` and ``tar`` are on ``PATH``.
"""

import argparse
import asyncio
import json
import os
import time
from dataclasses import replace
from pathlib import Path

from shellbox.backends.iris import machine as iris_backend
from shellbox.backends.iris.machine import IrisMachine, IrisMachineFactory
from shellbox.image import RegistryImage
from shellbox.machine import Command, MachineSpec, MachineTerminated, NetworkPolicy

IMAGE = "docker.io/library/ubuntu:24.04"
PRIVATE_IMAGE = (
    "envreg.208261-marin-gpu.coreweave.app/capability-infra/taskforge-image-build-smoke"
    "@sha256:3ae832dec16b02ff2f6bf0656cfca1449d42a5bce9de955d48e03efea2f903bc"
)
CONTROLLER_URL_ENV = "IRIS_CONTROLLER_URL"
GRADER_BASE_SCRIPT = 'echo "uid=$(id -u) pwd=$(pwd)"; for tool in python3 setsid tar; do command -v "$tool"; done'

NETWORK_SCRIPT = r"""
getent hosts registry-1.docker.io > /dev/null; echo "dns=$?"
timeout 5 bash -c 'exec 3<>/dev/tcp/1.1.1.1/443' 2> /dev/null; echo "tcp_ip=$?"
"""


def emit(check: str, **fields) -> None:
    print("PROBE " + json.dumps({"check": check, **fields}, default=str), flush=True)


async def timed(coro):
    start = time.monotonic()
    result = await coro
    return result, round(time.monotonic() - start, 2)


async def create(factory: IrisMachineFactory, spec: MachineSpec) -> tuple[IrisMachine, float]:
    return await timed(factory.create(spec))


async def check_factory_create(factory: IrisMachineFactory, spec: MachineSpec) -> None:
    start = time.monotonic()
    try:
        machine, seconds = await create(factory, spec)
    except Exception as error:
        emit(
            "factory_create",
            network=spec.network,
            ok=False,
            seconds=round(time.monotonic() - start, 2),
            error=repr(error)[:500],
        )
        return
    _, close_seconds = await timed(machine.close())
    emit("factory_create", network=spec.network, ok=True, create=seconds, close=close_seconds)


async def check_network(factory: IrisMachineFactory, spec: MachineSpec) -> None:
    machine, seconds = await create(factory, spec)
    try:
        network, network_seconds = await timed(machine.run(Command(("bash", "-c", NETWORK_SCRIPT), timeout=60)))
        emit(
            "network",
            network=spec.network,
            create=seconds,
            seconds=network_seconds,
            stdout=network.stdout.decode(),
            stderr=network.stderr.decode()[-500:],
        )
    finally:
        await machine.close()


async def check_machine(factory: IrisMachineFactory, spec: MachineSpec, tmp: str) -> None:
    machine, seconds = await create(factory, spec)
    try:
        names = ("HF_TOKEN", "WANDB_API_KEY", "AWS_SECRET_ACCESS_KEY", "CW_KEY_SECRET", "IRIS_JOB_ENV")
        script = "; ".join(f'printf "{name}=%s\\n" "${{#{name}}}"' for name in names)
        env = await machine.run(Command(("bash", "-c", script)))
        emit("environment", create=seconds, lengths=env.stdout.decode())
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
        emit(
            "transfer",
            bytes=len(payload),
            chunk_bytes=iris_backend.TRANSFER_CHUNK_BYTES,
            upload=upload_seconds,
            download=download_seconds,
            ok=round_trip,
            runs=runs,
        )
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
            emit(
                "ttl",
                phase="after",
                elapsed=elapsed,
                job_state=state.state,
                terminated=isinstance(error, MachineTerminated),
                error=repr(error)[:300],
            )
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
        machine = await short.create(spec)
    except Exception as error:
        emit("private_image", ok=False, seconds=round(time.monotonic() - start, 2), error=repr(error)[:800])
        return
    result = await machine.run(Command(("/opt/task/run.sh",)))
    emit("private_image", ok=True, seconds=round(time.monotonic() - start, 2), stdout=result.stdout.decode())
    await machine.close()


async def check_grader_base(factory: IrisMachineFactory, image: str) -> None:
    spec = MachineSpec(source=RegistryImage(image), workdir="/app", network=NetworkPolicy.DENY)
    start = time.monotonic()
    machine = await factory.create(spec)
    result = await machine.run(Command(("sh", "-c", GRADER_BASE_SCRIPT)))
    emit("grader_base", seconds=round(time.monotonic() - start, 2), stdout=result.stdout.decode())
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
    for policy in NetworkPolicy:
        await check_factory_create(factory, replace(spec, network=policy))
        await guarded(f"network_{policy}", check_network(factory, replace(spec, network=policy)))
    for attempt in range(2):
        await guarded(f"machine_attempt_{attempt}", check_machine(factory, spec, args.tmp))
    await guarded("private_image", check_private_image(factory))
    if args.grader_image:
        await guarded("grader_base", check_grader_base(factory, args.grader_image))
    await guarded("ttl", check_ttl(factory, spec, args.ttl))
    emit("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ttl", type=int, default=90)
    parser.add_argument("--tmp", default="/tmp")
    parser.add_argument("--chunk-bytes", type=int, default=None, help="override shellbox TRANSFER_CHUNK_BYTES")
    parser.add_argument("--grader-image", default=None, help="GRADER_BASE_IMAGE, to check it on gVisor")
    asyncio.run(main(parser.parse_args()))
