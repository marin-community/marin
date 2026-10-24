# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ray node supervisor executed with Python from the isolated TPU vLLM environment."""

import argparse
import importlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


def ray_start_command(*, host: str, address: str, leader: bool, chips: int, ports: dict[str, int]) -> list[str]:
    """Reserve fixed Ray listeners through Iris; let the kernel pick worker ports."""
    command = [
        sys.executable,
        "-m",
        "ray.scripts.scripts",
        "start",
        "--block",
        "--node-ip-address",
        host,
        "--num-cpus",
        "8",
        "--resources",
        json.dumps({"TPU": chips}),
        "--disable-usage-stats",
        "--min-worker-port",
        "0",
        "--max-worker-port",
        "0",
        "--dashboard-agent-listen-port",
        str(ports["ray-dashboard-agent"]),
    ]
    if leader:
        command += [
            "--head",
            "--port",
            str(ports["ray-gcs"]),
            "--include-dashboard=false",
            "--ray-client-server-port",
            str(ports["ray-client"]),
            "--dashboard-port",
            str(ports["ray-dashboard"]),
        ]
    else:
        command += ["--address", address]
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    os.environ["RAY_DEDUP_LOGS"] = "0"
    os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
    # Ray is installed only in this isolated environment, not in the serving coordinator's venv.
    ray = importlib.import_module("ray")

    stopped = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    signal.signal(signal.SIGINT, lambda *_: stopped.set())
    log_path = Path(config["log_path"])
    command = ray_start_command(
        host=config["host"],
        address=config["address"],
        leader=config["leader"],
        chips=config["chips"],
        ports=config["ports"],
    )
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + config["timeout_seconds"]

    def check_alive() -> None:
        if process.poll() is not None:
            raise RuntimeError(f"Ray start exited {process.returncode}:\n{log_path.read_text()[-12000:]}")
        if stopped.is_set():
            raise InterruptedError("Ray node stopped")

    initialized = threading.Event()
    failures: list[Exception] = []

    def initialize() -> None:
        try:
            ray.init(address=config["address"])
        except Exception as exc:
            failures.append(exc)
        finally:
            initialized.set()

    try:
        threading.Thread(target=initialize, daemon=True).start()
        while not initialized.wait(1):
            check_alive()
            if time.monotonic() >= deadline:
                raise TimeoutError("Ray initialization timed out")
        check_alive()
        if failures:
            raise RuntimeError("Ray initialization failed") from failures[0]
        while True:
            check_alive()
            nodes = [node for node in ray.nodes() if node["Alive"] and node["Resources"].get("TPU") == config["chips"]]
            if len(nodes) == config["hosts"]:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Ray rendezvous found {len(nodes)}/{config['hosts']} TPU hosts")
            stopped.wait(1)
        Path(config["ready_path"]).write_text(json.dumps({"hosts": len(nodes)}))
        while not stopped.wait(1):
            check_alive()
    finally:
        ray.shutdown()
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


if __name__ == "__main__":
    main()
