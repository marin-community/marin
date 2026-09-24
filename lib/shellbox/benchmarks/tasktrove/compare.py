# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare offline TaskTrove verifier runs in Docker and the QEMU prototype."""

import argparse
import asyncio
import json
import subprocess
import time
import tomllib
from pathlib import Path

from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths
from shellbox.backends.qemu.environment import QemuEnvironment

VERIFIER_COMMAND = "bash /tests/test.sh"


def docker_command(args: list[str], timeout: int = 600) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)


def docker_memory_bytes(container: str) -> int | None:
    inspect = docker_command(["inspect", "--format", "{{.State.Pid}}", container])
    if inspect.returncode:
        return None
    pid = int(inspect.stdout.strip())
    if pid <= 0:
        return None
    cgroups = (Path("/proc") / str(pid) / "cgroup").read_text().splitlines()
    unified = next((line.split("::", 1)[1] for line in cgroups if line.startswith("0::")), None)
    if unified is None:
        return None
    peak = Path("/sys/fs/cgroup") / unified.lstrip("/") / "memory.peak"
    return int(peak.read_text()) if peak.exists() else None


def qemu_pss_bytes(pid: int) -> int | None:
    path = Path("/proc") / str(pid) / "smaps_rollup"
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        if line.startswith("Pss:"):
            return int(line.split()[1]) * 1024
    return None


def docker_check(task: Path, image: str, check: str, output: Path) -> dict:
    container = f"tasktrove-compare-{task.name}-{check}"
    logs = output / "docker-logs" / task.name / check
    logs.mkdir(parents=True, exist_ok=True)
    mounts = ["-v", f"{task / 'tests'}:/tests:ro", "-v", f"{logs}:/logs/verifier"]
    if (task / "setup_files").is_dir():
        mounts.extend(["-v", f"{task / 'setup_files'}:/setup_files:ro"])
    if check == "oracle":
        mounts.extend(["-v", f"{task / 'solution'}:/solution:ro"])
    started = time.monotonic()
    run = docker_command(["run", "-d", "--network", "none", "--name", container, *mounts, image, "sleep", "infinity"])
    if run.returncode:
        return {"error": run.stderr[-1000:], "start_sec": time.monotonic() - started}
    try:
        ready = docker_command(["exec", container, "true"])
        start_sec = time.monotonic() - started
        if ready.returncode:
            return {"error": ready.stderr[-1000:], "start_sec": start_sec}
        action_start = time.monotonic()
        action = docker_command(
            ["exec", container, "bash", "-c", "bash /solution/solve.sh" if check == "oracle" else "true"]
        )
        action_sec = time.monotonic() - action_start
        verify_start = time.monotonic()
        verify = docker_command(["exec", container, "bash", "-c", VERIFIER_COMMAND])
        verify_sec = time.monotonic() - verify_start
        verdict = logs / "verdict.json"
        return {
            "start_sec": start_sec,
            "action_sec": action_sec,
            "verify_sec": verify_sec,
            "action_exit": action.returncode,
            "verify_exit": verify.returncode,
            "action_stderr": action.stderr[-500:],
            "verify_stderr": verify.stderr[-500:],
            "verdict": json.loads(verdict.read_text()) if verdict.exists() else None,
            "peak_memory_bytes": docker_memory_bytes(container),
        }
    finally:
        docker_command(["rm", "-f", container])


async def qemu_check(task: Path, bundle: Path, check: str, output: Path, guest_memory_mb: int = 512) -> dict:
    with (task / "task.toml").open("rb") as source:
        config = tomllib.load(source)
    paths = TrialPaths(output / "qemu-trials" / task.name / check)
    paths.mkdir()
    env = QemuEnvironment(
        environment_dir=task / "environment",
        environment_name=task.name,
        session_id=f"{task.name}-{check}",
        trial_paths=paths,
        task_env_config=EnvironmentConfig.model_validate(config.get("environment", {})),
        guest_bundle=str(bundle),
        network_policy="deny",
        guest_memory_mb=guest_memory_mb,
    )
    started = time.monotonic()
    try:
        await env.start(False)
        start_sec = time.monotonic() - started
        await env.upload_dir(task / "tests", "/tests")
        if (task / "setup_files").is_dir():
            await env.upload_dir(task / "setup_files", "/setup_files")
        if check == "oracle":
            await env.upload_dir(task / "solution", "/solution")
        action_start = time.monotonic()
        action = await env.exec("bash /solution/solve.sh" if check == "oracle" else "true", timeout_sec=600)
        action_sec = time.monotonic() - action_start
        verify_start = time.monotonic()
        verify = await env.exec(VERIFIER_COMMAND, timeout_sec=600)
        verify_sec = time.monotonic() - verify_start
        verdict_path = output / "qemu-verdicts" / task.name / f"{check}.json"
        verdict_path.parent.mkdir(parents=True, exist_ok=True)
        result = await env.exec("test -f /logs/verifier/verdict.json")
        if result.return_code == 0:
            await env.download_file("/logs/verifier/verdict.json", verdict_path)
        return {
            "start_sec": start_sec,
            "action_sec": action_sec,
            "verify_sec": verify_sec,
            "action_exit": action.return_code,
            "verify_exit": verify.return_code,
            "action_stderr": (action.stderr or "")[-500:],
            "verify_stderr": (verify.stderr or "")[-500:],
            "verdict": json.loads(verdict_path.read_text()) if verdict_path.exists() else None,
            "pss_memory_bytes": qemu_pss_bytes(env.machine.process.pid) if env.machine and env.machine.process else None,
        }
    except Exception as error:
        return {"error": repr(error), "elapsed_sec": time.monotonic() - started}
    finally:
        await env.stop(True)


async def compare(root: Path, limit: int | None, concurrency: int) -> None:
    sample = json.loads((root / "sample.json").read_text())
    results = root / "results.jsonl"
    completed = set()
    if results.exists():
        completed = {
            (row["row_index"], row["check"]) for line in results.read_text().splitlines() if (row := json.loads(line))
        }
    pending = []
    for row in sample["tasks"]:
        for check in ("empty", "oracle") if row["has_solution"] else ("empty",):
            if (row["row_index"], check) not in completed:
                pending.append((row, check))
    semaphore = asyncio.Semaphore(concurrency)

    async def run_check(row: dict, check: str) -> None:
        async with semaphore:
            task = root / "tasks" / f"{row['row_index']:05d}"
            image = f"tasktrove-clean-qemu:{row['dockerfile_id']}"
            bundle = root / "bundles" / row["dockerfile_id"]
            docker = await asyncio.to_thread(docker_check, task, image, check, root)
            qemu = await qemu_check(task, bundle, check, root)
            result = {
                "row_index": row["row_index"],
                "mode": row["mode"],
                "dockerfile_id": row["dockerfile_id"],
                "check": check,
                "docker": docker,
                "qemu": qemu,
            }
            with results.open("a") as destination:
                destination.write(json.dumps(result) + "\n")
            print(
                row["row_index"],
                row["mode"],
                check,
                docker.get("verdict", docker.get("error")),
                qemu.get("verdict", qemu.get("error")),
                flush=True,
            )

    await asyncio.gather(*(run_check(row, check) for row, check in pending[:limit]))
    recorded = [json.loads(line) for line in results.read_text().splitlines()]
    failures = [
        (row["row_index"], row["check"])
        for row in recorded
        if row["docker"].get("error")
        or row["qemu"].get("error")
        or row["docker"].get("verdict") != row["qemu"].get("verdict")
    ]
    if failures:
        raise AssertionError(f"Docker/QEMU comparison failed for {failures}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()
    asyncio.run(compare(args.root, args.limit, args.concurrency))
