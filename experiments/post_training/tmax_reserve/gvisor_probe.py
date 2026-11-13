# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Probe Shellbox's Iris gVisor sandboxes for Terminus-style TMax trials, and kaniko builds of TMax images.

Runs inside one Iris CPU job on a CoreWeave cluster. Nothing is published:

- Sandboxes use a public, digest-pinned Terminal-Bench base image that holds no TMax content. Each round
  starts ``--sandboxes`` sandboxes at once with egress denied. Each sandbox runs four checks:
  - ``setsid``: a detached process group is killed by group.
  - ``tmux``: the session persists across separate execs, as Terminus-2 uses it.
  - ``pytest``: pinned wheels are uploaded and a test file runs.
  - ``egress``: HTTP(S) and DNS fail.
  The round records the start latency, then closes every sandbox and checks that its job ended.
- Three TMax Dockerfiles are built with kaniko ``--no-push`` through Taskforge's build job. The image
  tarball goes only to the cluster's private temporary bucket, as the build job's output archive.

Every job runs on the cluster whose controller started this one. Run it where gVisor starts: on
cw-us-east-02a sandbox pods fail in ``FailedCreatePodSandBox`` (Echo incident 631, issue 10090).
"""

import argparse
import asyncio
import gzip
import hashlib
import io
import json
import logging
import statistics
import tarfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from iris.cli.connect import connect_controller
from iris.client import IrisClient, Job
from iris.cluster.client import get_job_info
from iris.cluster.types import Entrypoint, EnvironmentSpec, JobName, ResourceSpec
from iris.resources.state import is_job_finished
from iris.rpc import job_pb2
from rigging.filesystem.buckets import filesystem_for
from rigging.secrets import COREWEAVE_SECRET_NAMES, SecretBehavior, get_secret
from rigging.timing import Duration
from shellbox.backends.iris.machine import IrisMachineFactory
from shellbox.image import RegistryImage
from shellbox.machine import Command, ExitReason, MachineSpec, NetworkPolicy
from taskforge.sandbox.image_builder import (
    BUILD_IMAGE,
    BUILD_SCRIPT,
    BUILD_SETUP_FAILED_EXIT,
    CONTEXT_FILE,
    CRANE_SHA256,
    CRANE_URL,
    JOB_TIMEOUT,
    KANIKO_IMAGE,
    SCHEDULING_TIMEOUT,
    BuildFailed,
    output_archive,
    wait_succeeded,
)

logger = logging.getLogger(__name__)

# Terminal-Bench's public Python base (Debian 12, tmux 3.3a, setsid, bash, Python 3.13); no pytest, no curl.
PROBE_IMAGE = (
    "ghcr.io/laude-institute/t-bench/python-3-13@sha256:236734f0cafcce942ca09316d57236c2273a2b5411e116454a22cf6d718d95f5"
)
HERE = Path(__file__).parent
WHEEL_DIR = HERE / "tasks" / "probe-wheels"
WHEELS = {
    "iniconfig-2.0.0-py3-none-any.whl": "b6a85871a79d2e3b22d2d1b94ac2824226a63c6b741c88f7ae975f18b6778374",
    "packaging-24.2-py3-none-any.whl": "09abb1bccd265c01f4a3aa3f7a7db064b36514d2cba19a2f694fe6150451a759",
    "pluggy-1.5.0-py3-none-any.whl": "44e1ad92c8ca002de6377e165f3e0f1be63266ab4d554740532335b9d75ea669",
    "pytest-8.3.4-py3-none-any.whl": "50e16d954148559c9a74109af1eaf0c945ba2d8f30f0a3d3335edde19788b6f6",
}
TASK_DIR = HERE / "tasks" / "pilot-200"
# Smallest context, largest context (fixture COPYs), and one whose JSON-form COPY was rewritten.
BUILD_TASKS = ("task_009761_7f37c29a", "task_004317_5fa10315", "task_000110_cca950c9")
EGRESS_URLS = ("https://pypi.org/simple/", "http://archive.ubuntu.com/ubuntu/", "https://github.com/")
TEST_FILE = """\
from pathlib import Path


def test_round_trip(tmp_path):
    path = tmp_path / "x.txt"
    path.write_text("ok")
    assert path.read_text() == "ok"
    assert Path("/app").is_dir()
"""
UNPACK = (
    "import sys, zipfile, pathlib\n"
    "for wheel in sorted(pathlib.Path('/opt/probe-wheels').glob('*.whl')):\n"
    "    zipfile.ZipFile(wheel).extractall('/opt/pt')\n"
)
# One target per exec: urllib's timeout does not bound the DNS lookup, which can hang when egress drops packets.
EGRESS_URL = """\
import sys, urllib.request
try:
    urllib.request.urlopen(sys.argv[1], timeout=5)
    print("reached")
except Exception as error:
    print(type(error).__name__)
"""
EGRESS_DNS = """\
import socket
try:
    socket.getaddrinfo("pypi.org", 443)
    print("resolved")
except OSError as error:
    print(type(error).__name__)
"""
EGRESS_TIMEOUT = 30
# kaniko unpacks the base image over the root it runs in and keeps files already there. Taskforge runs it in
# an Alpine bash image, so Alpine's /etc/protocols and /etc/ssl/openssl.cnf stay in the build root, and an
# apt install of netbase or openssl stops at a dpkg conffile prompt. Like the official kaniko image, leave
# only the pod-provided name-resolution files in /etc before exec'ing kaniko.
KANIKO_EXEC = 'SSL_CERT_DIR="$K/ssl/certs" exec "$K/executor"'
CLEAR_ETC = "find /etc -mindepth 1 -maxdepth 1 ! -name resolv.conf ! -name hosts ! -name hostname -exec rm -rf {} +\n"
if BUILD_SCRIPT.count(KANIKO_EXEC) != 1:
    raise RuntimeError("Taskforge's kaniko build script changed; re-check the /etc clean-up")
PROBE_BUILD_SCRIPT = BUILD_SCRIPT.replace(KANIKO_EXEC, CLEAR_ETC + KANIKO_EXEC)
# kaniko holds layer snapshots in memory; 4 GiB OOMs on pip installs of large wheels.
BUILD_MEMORY = 16 * 1024**3
SETSID = (
    "setsid --fork sh -c 'echo $$ > /tmp/pg; exec sleep 600' </dev/null >/dev/null 2>&1; sleep 1; "
    "pg=$(cat /tmp/pg); kill -0 $pg && kill -TERM -- -$pg; sleep 1; "
    "if kill -0 $pg 2>/dev/null; then echo ALIVE; else echo KILLED; fi"
)
TMUX_EXECS = (
    "tmux new-session -d -s probe -x 160 -y 40",
    "tmux send-keys -t probe 'echo TMUX_$((6*7))' Enter",
    "for i in $(seq 20); do tmux capture-pane -p -t probe | grep -q TMUX_42 && echo SEEN && exit 0; sleep 0.5; done; "
    "echo MISSING",
)
COMMAND_TIMEOUT = 60
# Bounds an orphaned sandbox if this job dies before closing it (the factory default is 6 h).
SANDBOX_TTL = 900


def _bash(script: str) -> Command:
    return Command(("bash", "-c", script), timeout=COMMAND_TIMEOUT)


def _text(result) -> str:
    return result.stdout.decode(errors="replace").strip()


async def _egress(machine, script: str, *args: str) -> str:
    result = await machine.run(Command(("python3", "-c", script, *args), timeout=EGRESS_TIMEOUT))
    # A target that never answers within the deadline was not reached.
    return "timed_out" if result.reason == ExitReason.TIMED_OUT else _text(result) or f"exit {result.exit_code}"


async def _checks(machine, scratch: Path, row: dict) -> None:
    """Run the checks in order, recording each in ``row`` as it completes."""
    first = time.monotonic()
    await machine.run(Command(("true",), timeout=COMMAND_TIMEOUT))
    row["first_exec_seconds"] = time.monotonic() - first
    row["setsid"] = _text(await machine.run(_bash(SETSID))) == "KILLED"
    tmux = [await machine.run(_bash(script)) for script in TMUX_EXECS]
    row["tmux"] = all(r.exit_code == 0 for r in tmux) and _text(tmux[-1]) == "SEEN"
    await machine.upload(WHEEL_DIR, "/opt/probe-wheels")
    await machine.upload(scratch / "test_probe.py", "/tests/test_probe.py")
    unpacked = await machine.run(Command(("python3", "-c", UNPACK), timeout=COMMAND_TIMEOUT))
    pytest = await machine.run(
        Command(
            ("python3", "-m", "pytest", "-q", "-p", "no:cacheprovider", "/tests/test_probe.py"),
            env={"PYTHONPATH": "/opt/pt"},
            timeout=COMMAND_TIMEOUT,
        )
    )
    row["pytest"] = unpacked.exit_code == 0 and pytest.exit_code == 0 and "1 passed" in _text(pytest)
    egress = {url: await _egress(machine, EGRESS_URL, url) for url in EGRESS_URLS}
    egress["dns"] = await _egress(machine, EGRESS_DNS)
    row["egress"] = egress
    row["egress_blocked"] = all(value not in ("reached", "resolved") for value in egress.values())


async def _sandbox(factory: IrisMachineFactory, scratch: Path, index: int) -> dict:
    row: dict = {"index": index}
    start = time.monotonic()
    try:
        machine = await factory.create(
            MachineSpec(source=RegistryImage(PROBE_IMAGE), workdir="/app", network=NetworkPolicy.DENY, cpus=1)
        )
    except Exception as error:
        row["error"] = f"create: {type(error).__name__}: {error}"
        logger.info("sandbox %s", json.dumps(row))
        return row
    row["start_seconds"] = time.monotonic() - start
    row["job_id"] = str(machine.job.job_id)
    try:
        await _checks(machine, scratch, row)
    except Exception as error:
        row["error"] = f"checks: {type(error).__name__}: {error}"
    finally:
        await machine.close()
    logger.info("sandbox %s", json.dumps(row))
    return row


async def _round(factory: IrisMachineFactory, scratch: Path, sandboxes: int) -> list[dict]:
    return list(await asyncio.gather(*(_sandbox(factory, scratch, i) for i in range(sandboxes))))


def _jobs_terminal(controller_url: str, job_ids: list[str]) -> dict[str, bool]:
    with connect_controller(controller_url=controller_url) as endpoint:
        client = IrisClient.remote(endpoint.url, workspace=None, credentials=endpoint.credentials)
        try:
            return {job_id: is_job_finished(client.job_state(JobName.from_wire(job_id))) for job_id in job_ids}
        finally:
            client.shutdown()


def _context_archive(environment: Path) -> bytes:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for path in sorted(p for p in environment.rglob("*") if p.is_file()):
            info = tarfile.TarInfo(str(path.relative_to(environment)))
            content = path.read_bytes()
            info.size = len(content)
            info.mode = 0o755 if path.stat().st_mode & 0o111 else 0o644
            archive.addfile(info, io.BytesIO(content))
    return gzip.compress(raw.getvalue(), mtime=0)


def _submit_build(client: IrisClient, task: str) -> Job:
    """Submit Taskforge's kaniko build job with no registry secret and no cluster pin.

    ``taskforge``'s ``submit_job`` pins its target cluster, which a regional controller rejects as
    not a federation peer of itself. Like ``submit_job``, inherited variables are blanked and the
    job receives only the CoreWeave storage credentials Iris requires, as secrets.
    """
    job_info = get_job_info()
    inherited = {} if job_info is None else {name: "" for name in job_info.env if name not in COREWEAVE_SECRET_NAMES}
    storage = {name: value for name in COREWEAVE_SECRET_NAMES if (value := get_secret(name)) is not None}
    return client.submit(
        entrypoint=Entrypoint(
            command=["bash", "-c", PROBE_BUILD_SCRIPT],
            workdir_files={CONTEXT_FILE: _context_archive(TASK_DIR / task / "environment")},
        ),
        name=f"tmax-gvisor-probe-build-{uuid.uuid4().hex[:12]}",
        environment=EnvironmentSpec(
            setup_scripts=[],
            env_vars={
                **inherited,
                "CRANE_URL": CRANE_URL,
                "CRANE_SHA256": CRANE_SHA256,
                "KANIKO_IMAGE": KANIKO_IMAGE,
                "CONTEXT_FILE": CONTEXT_FILE,
                "DOCKERFILE": "Dockerfile",
                "BUILD_SETUP_FAILED_EXIT": str(BUILD_SETUP_FAILED_EXIT),
                # kaniko requires a destination name even with --no-push; .invalid never resolves.
                "DESTINATION": f"probe.invalid/tmax-probe:{task.replace('_', '-')}",
            },
            secrets=storage,
            secret_behavior=SecretBehavior.ISOLATE,
        ),
        resources=ResourceSpec(cpu=2, memory=BUILD_MEMORY, disk=20 * 1024**3),
        task_image=BUILD_IMAGE,
        container_profile=job_pb2.CONTAINER_PROFILE_DEFAULT,
        scheduling_timeout=Duration.from_seconds(SCHEDULING_TIMEOUT),
        timeout=Duration.from_seconds(JOB_TIMEOUT),
        max_retries_failure=0,
        max_retries_preemption=0,
    )


def _build(controller_url: str, task: str) -> dict:
    """Build one TMax image with kaniko --no-push; nothing is pushed anywhere."""
    start = time.monotonic()
    with connect_controller(controller_url=controller_url) as endpoint:
        client = IrisClient.remote(endpoint.url, workspace=None, credentials=endpoint.credentials)
        try:
            job = _submit_build(client, task)
            row = {"task": task, "job_id": str(job.job_id)}
            try:
                wait_succeeded(job, BuildFailed)
                archive = output_archive(job)
                row.update(succeeded=True, image_tar_bytes=archive.size_bytes)
            except Exception as error:
                row.update(succeeded=False, error=f"{type(error).__name__}: {error}")
            finally:
                if not is_job_finished(job.state):
                    job.cancel()
            row["seconds"] = time.monotonic() - start
            logger.info("build %s", json.dumps(row))
            return row
        finally:
            client.shutdown()


def _latency(rows: list[dict]) -> dict:
    starts = sorted(row["start_seconds"] for row in rows if "start_seconds" in row)
    if not starts:
        return {"started": 0}
    return {
        "started": len(starts),
        "p50": statistics.median(starts),
        "p90": starts[min(len(starts) - 1, int(0.9 * len(starts)))],
        "max": starts[-1],
    }


def summarize(rounds: list[list[dict]], terminal: dict[str, bool], builds: list[dict], max_p90: float) -> dict:
    """Score the probe against its pass rules."""
    rows = [row for round_rows in rounds for row in round_rows]
    checks = {name: sum(bool(row.get(name)) for row in rows) for name in ("setsid", "tmux", "pytest", "egress_blocked")}
    latencies = [_latency(round_rows) for round_rows in rounds]
    not_terminal = sorted(job_id for job_id, finished in terminal.items() if not finished)
    passed = (
        all(count == len(rows) for count in checks.values())
        and not any("error" in row for row in rows)
        and all(latency.get("p90", float("inf")) <= max_p90 for latency in latencies)
        and not not_terminal
        and all(build["succeeded"] for build in builds)
    )
    return {
        "image": PROBE_IMAGE,
        "sandboxes": len(rows),
        "checks_passed": checks,
        "errors": [row for row in rows if "error" in row],
        "latency_seconds": latencies,
        "first_exec_seconds_max": max((row.get("first_exec_seconds", 0.0) for row in rows), default=0.0),
        "jobs_not_terminal_after_close": not_terminal,
        "builds": builds,
        "passed": passed,
    }


def _check_wheels() -> None:
    for name, digest in WHEELS.items():
        if hashlib.sha256((WHEEL_DIR / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"wheel {name} does not match its pinned sha256")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sandboxes", type=int, required=True, help="Concurrent sandboxes per round.")
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--max-p90-start", type=float, required=True, help="Pass rule for p90 start latency (s).")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upload", required=True, help="Remote directory that receives a copy of --output.")
    args = parser.parse_args()
    _check_wheels()
    job = get_job_info()
    if job is None or job.controller_address is None:
        raise RuntimeError("The probe runs inside an Iris job")
    controller_url = job.controller_address
    args.output.mkdir(parents=True, exist_ok=True)
    scratch = args.output / "inputs"
    scratch.mkdir(exist_ok=True)
    (scratch / "test_probe.py").write_text(TEST_FILE)
    factory = IrisMachineFactory(controller_url=controller_url, job_ttl=SANDBOX_TTL)
    try:
        summary = _probe(factory, controller_url, scratch, args)
    finally:
        fs, path = filesystem_for(args.upload)
        fs.put(str(args.output), path, recursive=True)
    if not summary["passed"]:
        raise SystemExit("gVisor probe did not pass; see probe-summary.json")


def _probe(factory: IrisMachineFactory, controller_url: str, scratch: Path, args: argparse.Namespace) -> dict:
    rounds = []
    with ThreadPoolExecutor(len(BUILD_TASKS)) as pool:
        build_futures = [pool.submit(_build, controller_url, task) for task in BUILD_TASKS]
        for number in range(args.rounds):
            logger.info("round %d: %d sandboxes", number, args.sandboxes)
            rows = asyncio.run(_round(factory, scratch, args.sandboxes))
            (args.output / f"round-{number}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            rounds.append(rows)
        builds = []
        for task, future in zip(BUILD_TASKS, build_futures, strict=True):
            try:
                builds.append(future.result())
            except Exception as error:
                builds.append({"task": task, "succeeded": False, "error": f"submit: {type(error).__name__}: {error}"})
    (args.output / "builds.json").write_text(json.dumps(builds, indent=1))
    job_ids = [row["job_id"] for rows in rounds for row in rows if "job_id" in row]
    terminal = _jobs_terminal(controller_url, job_ids)
    summary = summarize(rounds, terminal, builds, args.max_p90_start)
    (args.output / "probe-summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    main()
