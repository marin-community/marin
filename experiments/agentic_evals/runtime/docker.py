"""Docker runtime detection and configuration for Harbor Docker backend.

Extracted from OT-Agent ``hpc/docker_runtime.py`` (stripped the cloud/SkyPilot
image-selection functions at the bottom). Supports:
- Native Docker daemon
- Podman with Docker CLI emulation
- Remote Docker via SSH tunnel (for SLURM clusters)
"""

import os
import shutil
import stat
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


class DockerRuntimeType(Enum):
    DOCKER = "docker"
    PODMAN = "podman"
    PODMAN_HPC = "podman_hpc"
    REMOTE = "remote"
    UNAVAILABLE = "unavailable"


@dataclass
class DockerRuntimeConfig:
    runtime_type: DockerRuntimeType
    docker_host: Optional[str] = None
    socket_path: Optional[str] = None
    requires_tunnel: bool = False
    tunnel_port: Optional[int] = None
    extra_env: Dict[str, str] = field(default_factory=dict)


def get_podman_socket_path() -> Optional[str]:
    try:
        user_id = subprocess.run(
            ["id", "-u"], capture_output=True, text=True, check=True
        ).stdout.strip()
        return f"/run/user/{user_id}/podman/podman.sock"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def is_podman_hpc_available() -> bool:
    return shutil.which("podman-hpc") is not None


def _is_podman_docker() -> bool:
    try:
        result = subprocess.run(
            ["docker", "--version"], capture_output=True, text=True, timeout=5
        )
        return "podman" in (result.stdout + result.stderr).lower()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _socket_exists(path: str) -> bool:
    try:
        mode = os.stat(path).st_mode
        return stat.S_ISSOCK(mode)
    except (OSError, FileNotFoundError):
        return False


def detect_docker_runtime() -> DockerRuntimeConfig:
    existing_host = os.environ.get("DOCKER_HOST")
    if existing_host:
        if existing_host.startswith("tcp://"):
            return DockerRuntimeConfig(
                runtime_type=DockerRuntimeType.REMOTE,
                docker_host=existing_host,
                requires_tunnel=True,
            )
        elif existing_host.startswith("unix://"):
            socket_path = existing_host.replace("unix://", "")
            if "podman" in socket_path:
                if is_podman_hpc_available():
                    return DockerRuntimeConfig(
                        runtime_type=DockerRuntimeType.PODMAN_HPC,
                        docker_host=existing_host,
                        socket_path=socket_path,
                        extra_env={"CONTAINER_RUNTIME": "podman_hpc"},
                    )
                return DockerRuntimeConfig(
                    runtime_type=DockerRuntimeType.PODMAN,
                    docker_host=existing_host,
                    socket_path=socket_path,
                )
            else:
                return DockerRuntimeConfig(
                    runtime_type=DockerRuntimeType.DOCKER,
                    docker_host=existing_host,
                    socket_path=socket_path,
                )

    if is_podman_hpc_available():
        podman_socket = get_podman_socket_path()
        if podman_socket:
            return DockerRuntimeConfig(
                runtime_type=DockerRuntimeType.PODMAN_HPC,
                docker_host=f"unix://{podman_socket}",
                socket_path=podman_socket,
                extra_env={
                    "CONTAINER_RUNTIME": "podman_hpc",
                    "_PODMAN_SOCKET_NEEDS_START": "1" if not _socket_exists(podman_socket) else "",
                },
            )

    podman_socket = get_podman_socket_path()

    if _is_podman_docker():
        if podman_socket and _socket_exists(podman_socket):
            return DockerRuntimeConfig(
                runtime_type=DockerRuntimeType.PODMAN,
                docker_host=f"unix://{podman_socket}",
                socket_path=podman_socket,
            )
        if podman_socket:
            return DockerRuntimeConfig(
                runtime_type=DockerRuntimeType.PODMAN,
                docker_host=f"unix://{podman_socket}",
                socket_path=podman_socket,
                extra_env={"_PODMAN_SOCKET_NEEDS_START": "1"},
            )

    if shutil.which("podman") and podman_socket:
        if _socket_exists(podman_socket):
            return DockerRuntimeConfig(
                runtime_type=DockerRuntimeType.PODMAN,
                docker_host=f"unix://{podman_socket}",
                socket_path=podman_socket,
            )
        return DockerRuntimeConfig(
            runtime_type=DockerRuntimeType.PODMAN,
            docker_host=f"unix://{podman_socket}",
            socket_path=podman_socket,
            extra_env={"_PODMAN_SOCKET_NEEDS_START": "1"},
        )

    docker_socket = "/var/run/docker.sock"
    if _socket_exists(docker_socket):
        return DockerRuntimeConfig(
            runtime_type=DockerRuntimeType.DOCKER,
            docker_host=f"unix://{docker_socket}",
            socket_path=docker_socket,
        )

    home = os.path.expanduser("~")
    docker_desktop_socket = f"{home}/.docker/run/docker.sock"
    if _socket_exists(docker_desktop_socket):
        return DockerRuntimeConfig(
            runtime_type=DockerRuntimeType.DOCKER,
            docker_host=f"unix://{docker_desktop_socket}",
            socket_path=docker_desktop_socket,
        )

    return DockerRuntimeConfig(runtime_type=DockerRuntimeType.UNAVAILABLE)


def setup_docker_environment(
    config: DockerRuntimeConfig, env: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    if env is None:
        env = {}
    if config.docker_host:
        env["DOCKER_HOST"] = config.docker_host
    env.update(config.extra_env)
    return env


def try_start_podman_socket(timeout: int = 3) -> bool:
    try:
        subprocess.run(
            ["systemctl", "--user", "start", "podman.socket"],
            capture_output=True, timeout=timeout, check=False,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass
    try:
        subprocess.run(
            ["podman", "system", "service", "--time=0"],
            capture_output=True, timeout=timeout, check=False,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return False


def check_docker_connectivity(timeout: int = 5) -> bool:
    try:
        result = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=timeout, check=False
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def verify_docker_connectivity_with_details(
    runtime: DockerRuntimeConfig, timeout: int = 10
) -> tuple[bool, str]:
    env = os.environ.copy()
    if runtime.docker_host:
        env["DOCKER_HOST"] = runtime.docker_host
    try:
        result = subprocess.run(
            ["docker", "info"], capture_output=True, text=True, timeout=timeout, env=env,
        )
        if result.returncode == 0:
            return True, "Docker daemon is accessible"
        return False, f"Docker info failed: {result.stderr.strip()}"
    except FileNotFoundError:
        return False, "docker command not found"
    except subprocess.TimeoutExpired:
        return False, f"Docker daemon not responding (timeout after {timeout}s)"
    except Exception as e:
        return False, f"Unexpected error: {e}"


def setup_docker_runtime_if_needed(env_type: str, fail_fast: bool = True) -> None:
    """Configure Docker/Podman runtime if using docker backend."""
    import sys

    if env_type.lower() != "docker":
        return

    print("[docker] Detecting Docker/Podman runtime...")
    runtime = detect_docker_runtime()

    if runtime.runtime_type == DockerRuntimeType.UNAVAILABLE:
        error_msg = (
            "Docker backend requested but no Docker/Podman runtime found.\n"
            "[docker] HINT: Install Docker/Podman, or set DOCKER_HOST environment variable."
        )
        print(f"[docker] ERROR: {error_msg}")
        if fail_fast:
            raise RuntimeError(error_msg)
        sys.exit(2)

    if runtime.extra_env.get("_PODMAN_SOCKET_NEEDS_START"):
        print("[docker] Podman socket not running, attempting to start...")
        if try_start_podman_socket(timeout=5):
            runtime = detect_docker_runtime()
            print("[docker] Podman socket started successfully")
        else:
            error_msg = f"Failed to start podman socket at {runtime.socket_path}"
            print(f"[docker] ERROR: {error_msg}")
            if fail_fast:
                raise RuntimeError(error_msg)
            sys.exit(3)

    env = setup_docker_environment(runtime)
    os.environ.update(env)

    runtime_name = runtime.runtime_type.value
    if runtime.runtime_type == DockerRuntimeType.PODMAN_HPC:
        runtime_name = "podman_hpc (NERSC HPC environment)"
    print(f"[docker] Runtime type: {runtime_name}")
    print(f"[docker] DOCKER_HOST: {runtime.docker_host}")

    success, message = verify_docker_connectivity_with_details(runtime, timeout=10)
    if not success:
        error_msg = f"{message}\n[docker] HINT: Check 'docker info' manually."
        print(f"[docker] ERROR: {error_msg}")
        if fail_fast:
            raise RuntimeError(error_msg)
        print("[docker] WARNING: Continuing despite connectivity failure...")
    else:
        print(f"[docker] {message}")
