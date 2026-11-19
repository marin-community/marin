"""Isolated Mochi control: native13 environment under unchanged Ray ownership."""

import fcntl
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

from rigging.network import interface_for_ipv4
from skyrl_train.worker_setup import configure_worker_process as configure_skyrl_worker

CONTROL_KEY = "MOCHI_NATIVE_ENVIRONMENT_CONTROL"


def actor_runtime_env(profile: dict[str, Any], inherited: dict[str, Any] | None) -> dict[str, Any]:
    """Project the native library search before Ray starts an inference worker."""
    result = dict(inherited or {})
    environment = dict(result.get("env_vars", {}))
    for key in list(environment):
        if controlled_name(key, profile):
            environment[key] = ""
    for key, value in profile["values"].items():
        if key not in profile["dynamic_names"]:
            environment[key] = value or ""
    # LD_LIBRARY_PATH must be empty at exec, before loading torch or vLLM.
    environment["LD_LIBRARY_PATH"] = ""
    environment[CONTROL_KEY] = json.dumps(profile, sort_keys=True)
    result["env_vars"] = environment
    result["worker_process_setup_hook"] = "marinskyrl.native_environment_control.configure_worker_process"
    return result


def controlled_name(name: str, profile: dict[str, Any]) -> bool:
    if name in profile["preserve"] or name in profile["dynamic_names"]:
        return False
    return name in profile["runtime_names"] or name.startswith(tuple(profile["runtime_prefixes"]))


def apply_environment(profile: dict[str, Any], host: str) -> None:
    """Apply native values, rebasing only the current host address/interface."""
    for key in list(os.environ):
        if controlled_name(key, profile):
            os.environ.pop(key)
    for key, value in profile["values"].items():
        if value is not None and key not in profile["dynamic_names"]:
            os.environ[key] = value
    os.environ["VLLM_HOST_IP"] = host
    os.environ["GLOO_SOCKET_IFNAME"] = interface_for_ipv4(host)


def record_environment(phase: str, *, rank: int | None = None) -> None:
    """Persist only the nonsecret keys declared in the sealed control profile."""
    if CONTROL_KEY not in os.environ:
        return
    profile = json.loads(os.environ[CONTROL_KEY])
    keys = set(profile["values"]) | {"CUDA_VISIBLE_DEVICES"}
    values = {key: os.environ.get(key) for key in sorted(keys)}
    receipt = {
        "phase": phase,
        "pid": os.getpid(),
        "rank": rank,
        "host": socket.gethostname(),
        "process_name": multiprocessing.current_process().name,
        "environment": values,
        "profile_sha256": hashlib.sha256(os.environ[CONTROL_KEY].encode()).hexdigest(),
    }
    directory = Path(os.environ["IRIS_OUTPUT_DIR"]) / "native-environment-control"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{os.getpid()}-{phase}.json"
    destination.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    print("MOCHI_NATIVE_ENV_RECEIPT=" + json.dumps(receipt, sort_keys=True), flush=True)


def _link(target: Path, destination: Path) -> None:
    try:
        destination.symlink_to(target, target_is_directory=target.is_dir())
    except FileExistsError:
        if destination.resolve() != target.resolve():
            raise RuntimeError(f"Native path alias collision: {destination}") from None


def prepare_paths(profile: dict[str, Any], runtime: Path) -> None:
    """Recreate the native dependency prefix without changing the interpreter."""
    alias = Path(profile["venv_alias"])
    (alias / "lib/python3.12").mkdir(parents=True, exist_ok=True)
    _link(runtime / "lib/python3.12/site-packages", alias / "lib/python3.12/site-packages")
    _link(runtime / "bin", alias / "bin")
    cuda = Path(profile["values"]["CUDA_HOME"])
    _link(cuda / "lib/libcudart.so.13", alias / "lib/libcudart.so")
    # Keep the frozen installation immutable; alias lib64 through the native root.
    if not (cuda / "lib64").exists():
        _link(cuda / "lib", cuda / "lib64")
    config = Path(profile["values"]["AWS_CONFIG_FILE"])
    config.write_text("[default]\ns3 =\n    addressing_style = virtual\n")
    template = Path(profile["native_cli"][profile["native_cli"].index("--chat-template") + 1])
    template.write_text(profile["chat_template"])


def configure_worker_process() -> None:
    """Prepare native paths before the inference actor imports CUDA libraries."""
    if "torch" in sys.modules or "vllm" in sys.modules:
        raise RuntimeError("Native environment control must precede torch/vLLM import")
    if os.environ.get("LD_LIBRARY_PATH"):
        raise RuntimeError("Ray did not clear LD_LIBRARY_PATH before process exec")
    profile = json.loads(os.environ[CONTROL_KEY])
    actual_versions = {name: importlib.metadata.version(name) for name in profile["versions"]}
    if actual_versions != profile["versions"]:
        raise RuntimeError(f"Native control runtime version mismatch: {actual_versions}")
    directory = Path(profile["venv_alias"]).parent
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "native-control.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        marker = directory / "native-control-prepared.json"
        if not marker.exists():
            prepare_paths(profile, Path(sys.prefix))
            subprocess.run(
                [sys.executable, "-m", "marinskyrl.native_environment_cache"],
                check=True,
                timeout=300,
            )
            marker.write_text(json.dumps({"profile_sha256": hashlib.sha256(os.environ[CONTROL_KEY].encode()).hexdigest()}))
        prepared = json.loads(marker.read_text())
        expected_hash = hashlib.sha256(os.environ[CONTROL_KEY].encode()).hexdigest()
        if prepared["profile_sha256"] != expected_hash:
            raise RuntimeError("Native environment preparation belongs to a different control profile")
    host = os.environ["IRIS_ADVERTISE_HOST"]
    apply_environment(profile, host)
    configure_skyrl_worker()
    record_environment("worker-bootstrap")


def finalize_actor_environment(rank: int | None) -> None:
    """Undo actor setup overrides before creating the external-DP EngineCore."""
    if CONTROL_KEY not in os.environ:
        return
    profile = json.loads(os.environ[CONTROL_KEY])
    apply_environment(profile, os.environ["VLLM_HOST_IP"])
    record_environment("actor-before-engine", rank=rank)
