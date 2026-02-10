# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent


@dataclass
class ManagedProcess:
    name: str
    process: subprocess.Popen


def _start_process(name: str, cmd: list[str], env: dict[str, str]) -> ManagedProcess:
    printable = " ".join(cmd)
    print(f"Starting {name}: {printable}")
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env)
    return ManagedProcess(name, proc)


def _resolve_config_path(config_path: str) -> Path:
    path = Path(config_path)
    if not path.is_absolute():
        path = ROOT / path
    return path


def _read_port_from_config(config_path: str) -> int | None:
    path = _resolve_config_path(config_path)
    if not path.exists():
        print(f"Warning: config file '{config_path}' not found (expected at {path}). Falling back to default port 5000.")
        return None
    try:
        with path.open() as fp:
            config_data = yaml.safe_load(fp) or {}
    except Exception as exc:  # pragma: no cover - defensive logging only
        print(f"Warning: failed to parse config '{path}': {exc}. Falling back to default port 5000.")
        return None

    port = config_data.get("port")
    if port is None:
        return None
    try:
        return int(port)
    except (TypeError, ValueError):
        print(f"Warning: invalid 'port' value ({port!r}) in {path}. Falling back to default port 5000.")
        return None


def _stop_processes(processes: list[ManagedProcess]) -> None:
    for managed in processes:
        proc = managed.process
        if proc.poll() is None:
            print(f"Stopping {managed.name}...")
            proc.terminate()

    for managed in processes:
        proc = managed.process
        if proc.poll() is None:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"Forcing {managed.name} to exit...")
                proc.kill()
                proc.wait()


def _monitor(processes: list[ManagedProcess]) -> int:
    exit_code = 0
    try:
        while True:
            for managed in processes:
                ret = managed.process.poll()
                if ret is not None:
                    exit_code = ret
                    raise RuntimeError(f"{managed.name} exited with code {ret}")
            time.sleep(0.3)
    except KeyboardInterrupt:
        print("\nReceived interrupt, shutting down...")
    except RuntimeError as exc:
        print(exc)
    finally:
        _stop_processes(processes)
    return exit_code


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the data-browser backend and frontend together.")
    parser.add_argument(
        "--config",
        default="conf/local.conf",
        help="Path to the server configuration file (default: %(default)s).",
    )
    parser.add_argument(
        "--frontend-cmd",
        default="npm start",
        help="Command used to start the React dev server (default: %(default)s).",
    )
    parser.add_argument(
        "--backend-only",
        action="store_true",
        help="Only start the Flask backend (skip the React dev server).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    processes: list[ManagedProcess] = []

    configured_port = _read_port_from_config(args.config)
    backend_port = configured_port if configured_port is not None else 5000
    print(f"Using backend port {backend_port}" + (" (from config)" if configured_port is not None else " (default)"))

    backend_env = os.environ.copy()
    backend_env["DEV"] = "true"
    backend_cmd = [sys.executable, "server.py", "--config", args.config]
    processes.append(_start_process("backend", backend_cmd, backend_env))

    if not args.backend_only:
        frontend_env = os.environ.copy()
        frontend_env["MARIN_DATA_BROWSER_PROXY_TARGET"] = f"http://localhost:{backend_port}"
        frontend_cmd = shlex.split(args.frontend_cmd)
        processes.append(_start_process("frontend", frontend_cmd, frontend_env))

    def _signal_handler(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _signal_handler)

    return _monitor(processes)


if __name__ == "__main__":
    raise SystemExit(main())
