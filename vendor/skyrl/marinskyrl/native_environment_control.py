"""Isolated Mochi control: native13 environment under unchanged Ray ownership."""

import fcntl
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
import shlex
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

from rigging.network import interface_for_ipv4

CONTROL_KEY = "MOCHI_NATIVE_ENVIRONMENT_CONTROL"


def actor_runtime_env(profile: dict[str, Any], inherited: dict[str, Any] | None) -> dict[str, Any]:
    """Project the native library search before Ray starts an inference worker."""
    result = dict(inherited or {})
    environment = dict(result.get("env_vars", {}))
    # Positive values are safe before the worker bootstrap. Absence needs an
    # exec prelude: empty strings can be invalid typed settings in dependencies.
    for key, value in profile["values"].items():
        environment.pop(key, None)
        if value is not None and key not in profile["dynamic_names"]:
            environment[key] = value
    environment["LD_LIBRARY_PATH"] = ""
    environment[CONTROL_KEY] = json.dumps(profile, sort_keys=True)
    result["env_vars"] = environment
    result["py_executable"] = shlex.join([sys.executable, "-m", "marinskyrl.native_environment_worker"])
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
    keys = set(profile["values"]) | set(profile["preserve"])
    modules = {}
    for name in ("huggingface_hub", "transformers", "vllm"):
        module = sys.modules.get(name)
        if module is not None:
            modules[name] = module.__file__
    values = {key: os.environ.get(key) for key in sorted(keys)}
    receipt = {
        "phase": phase,
        "pid": os.getpid(),
        "rank": rank,
        "host": socket.gethostname(),
        "process_name": multiprocessing.current_process().name,
        "environment": values,
        "python_executable": sys.executable,
        "python_prefix": sys.prefix,
        "loaded_module_paths": modules,
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
    """Recreate native CUDA paths without aliasing Python packages or executables."""
    alias = Path(profile["venv_alias"])
    cuda = Path(profile["values"]["CUDA_HOME"])
    cuda.mkdir(parents=True, exist_ok=True)
    source_cuda = runtime / "lib/python3.12/site-packages/nvidia/cu13"
    for item in source_cuda.iterdir():
        if item.name != "lib64":
            _link(item, cuda / item.name)
    _link(cuda / "lib", cuda / "lib64")
    _link(cuda / "lib/libcudart.so.13", alias / "lib/libcudart.so")
    config = Path(profile["values"]["AWS_CONFIG_FILE"])
    config.write_text("[default]\ns3 =\n    addressing_style = virtual\n")
    template = Path(profile["native_cli"][profile["native_cli"].index("--chat-template") + 1])
    template.write_text(profile["chat_template"])


def prepare_worker_process() -> None:
    """Prepare native GPU settings before exec starts the unchanged Python worker."""
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
    record_environment("worker-pre-exec")


def finalize_actor_environment(rank: int | None) -> None:
    """Undo actor setup overrides before creating the external-DP EngineCore."""
    if CONTROL_KEY not in os.environ:
        return
    profile = json.loads(os.environ[CONTROL_KEY])
    apply_environment(profile, os.environ["VLLM_HOST_IP"])
    record_environment("actor-before-engine", rank=rank)
