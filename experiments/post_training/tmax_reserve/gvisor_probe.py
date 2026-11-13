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
from iris.client import IrisClient
from iris.cluster.client import get_job_info
from iris.cluster.types import Entrypoint, JobName
from iris.resources.state import is_job_finished
from rigging.filesystem.buckets import filesystem_for
from shellbox.backends.iris.machine import IrisMachineFactory
from shellbox.image import RegistryImage
from shellbox.machine import Command, MachineSpec, NetworkPolicy
from taskforge.sandbox.image_builder import (
    BUILD_IMAGE,
    BUILD_SCRIPT,
    BUILD_SETUP_FAILED_EXIT,
    CONTEXT_FILE,
    CRANE_SHA256,
    CRANE_URL,
    KANIKO_IMAGE,
    BuildFailed,
    output_archive,
    submit_job,
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
EGRESS = """\
import json, socket, sys, urllib.request
urls = sys.argv[1:]
result = {}
for url in urls:
    try:
        urllib.request.urlopen(url, timeout=5)
        result[url] = "reached"
    except Exception as error:
        result[url] = type(error).__name__
try:
    socket.getaddrinfo("pypi.org", 443)
    result["dns"] = "resolved"
except OSError as error:
    result["dns"] = type(error).__name__
print(json.dumps(result))
"""
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


async def _checks(machine, scratch: Path) -> dict:
    checks: dict = {}
    first = time.monotonic()
    await machine.run(Command(("true",), timeout=COMMAND_TIMEOUT))
    checks["first_exec_seconds"] = time.monotonic() - first
    checks["setsid"] = _text(await machine.run(_bash(SETSID))) == "KILLED"
    tmux = [await machine.run(_bash(script)) for script in TMUX_EXECS]
    checks["tmux"] = all(r.exit_code == 0 for r in tmux) and _text(tmux[-1]) == "SEEN"
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
    checks["pytest"] = unpacked.exit_code == 0 and pytest.exit_code == 0 and "1 passed" in _text(pytest)
    egress = json.loads(
        _text(await machine.run(Command(("python3", "-c", EGRESS, *EGRESS_URLS), timeout=COMMAND_TIMEOUT)))
    )
    checks["egress"] = egress
    checks["egress_blocked"] = all(value not in ("reached", "resolved") for value in egress.values())
    return checks


async def _sandbox(factory: IrisMachineFactory, scratch: Path, index: int) -> dict:
    row: dict = {"index": index}
    start = time.monotonic()
    try:
        machine = await factory.create(
            MachineSpec(source=RegistryImage(PROBE_IMAGE), workdir="/app", network=NetworkPolicy.DENY, cpus=1)
        )
    except Exception as error:
        row["error"] = f"create: {type(error).__name__}: {error}"
        return row
    row["start_seconds"] = time.monotonic() - start
    row["job_id"] = str(machine.job.job_id)
    try:
        row.update(await _checks(machine, scratch))
    except Exception as error:
        row["error"] = f"checks: {type(error).__name__}: {error}"
    finally:
        await machine.close()
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


def _build(controller_url: str, target_cluster: str, task: str) -> dict:
    """Build one TMax image with kaniko --no-push; nothing is pushed anywhere."""
    run_id = uuid.uuid4().hex[:12]
    start = time.monotonic()
    with connect_controller(controller_url=controller_url) as endpoint:
        client = IrisClient.remote(endpoint.url, workspace=None, credentials=endpoint.credentials)
        try:
            job = submit_job(
                client,
                name=f"tmax-gvisor-probe-build-{run_id}",
                entrypoint=Entrypoint(
                    command=["bash", "-c", BUILD_SCRIPT],
                    workdir_files={CONTEXT_FILE: _context_archive(TASK_DIR / task / "environment")},
                ),
                image=BUILD_IMAGE,
                env={
                    "CRANE_URL": CRANE_URL,
                    "CRANE_SHA256": CRANE_SHA256,
                    "KANIKO_IMAGE": KANIKO_IMAGE,
                    "CONTEXT_FILE": CONTEXT_FILE,
                    "DOCKERFILE": "Dockerfile",
                    "BUILD_SETUP_FAILED_EXIT": str(BUILD_SETUP_FAILED_EXIT),
                    # kaniko requires a destination name even with --no-push; .invalid never resolves.
                    "DESTINATION": f"probe.invalid/tmax-probe:{task.replace('_', '-')}",
                },
                secrets={},
                target_cluster=target_cluster,
            )
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
    parser.add_argument("--target-cluster", required=True, help="Cluster the kaniko build jobs run on.")
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
    with ThreadPoolExecutor(len(BUILD_TASKS)) as pool:
        build_futures = [pool.submit(_build, controller_url, args.target_cluster, task) for task in BUILD_TASKS]
        rounds = []
        for number in range(args.rounds):
            logger.info("round %d: %d sandboxes", number, args.sandboxes)
            rows = asyncio.run(_round(factory, scratch, args.sandboxes))
            (args.output / f"round-{number}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            rounds.append(rows)
        builds = [future.result() for future in build_futures]
    job_ids = [row["job_id"] for rows in rounds for row in rows if "job_id" in row]
    terminal = _jobs_terminal(controller_url, job_ids)
    summary = summarize(rounds, terminal, builds, args.max_p90_start)
    (args.output / "probe-summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    fs, path = filesystem_for(args.upload)
    fs.put(str(args.output), path, recursive=True)
    if not summary["passed"]:
        raise SystemExit("gVisor probe did not pass; see probe-summary.json")


if __name__ == "__main__":
    main()
