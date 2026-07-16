"""EVAL-ONLY LocalHarborRunner: Ray + vLLM lifecycle + Harbor exec.

Extracted from OT-Agent ``hpc/local_runner_utils.py``, stripped of all
datagen-only paths (literal_proxy, ingress controller, opencode routing,
RL train-data staging). Trivial helpers from ``hpc.launch_utils``
(``generate_served_model_id``, ``hosted_vllm_alias``, ``maybe_int``,
``shorten_model_name``, ``default_job_name``) are inlined.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import resource
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .args import (
    add_harbor_args,
    add_model_compute_args,
    add_ray_vllm_args,
    add_log_path_args,
)
from .docker import setup_docker_runtime_if_needed
from .vllm_server import build_vllm_cli_args as _build_vllm_cli_args, run_endpoint_health_check
from ..harness.config import (
    load_harbor_config,
    get_harbor_env_from_config,
    resolve_jobs_dir_path,
)
from ..harness.command import (
    build_harbor_command,
    load_endpoint_metadata,
    run_harbor_cli,
)
from ..harness.job_config import load_job_config
from ..harness._compat import get_orchestrator_field
from ..harness.trial_prune import prune_refire_errored_trials
from ..serve.tpu import drop_tpu_unsupported_serve_flags, add_tpu_serve_default_flags


# ---------------------------------------------------------------------------
# Inlined trivial helpers (from hpc.launch_utils)
# ---------------------------------------------------------------------------

_HOSTED_VLLM_PREFIX = "hosted_vllm/"
JOB_NAME_SEP = "__"
MODEL_NAME_MAX_LENGTH = 20


def shorten_model_name(raw: str, max_len: int = MODEL_NAME_MAX_LENGTH) -> str:
    name = raw.strip().rstrip("/")
    if "/" in name:
        name = name.split("/")[-1]
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-_")
    if len(name) > max_len:
        name = name[:max_len].rstrip("-_")
    return name or "model"


def generate_served_model_id(job_name: Optional[str] = None) -> str:
    """Deterministic-per-job served model ID (sha256-derived when job_name given)."""
    if job_name:
        digest_hex = hashlib.sha256(job_name.encode("utf-8")).hexdigest()
        return str(int(digest_hex[:16], 16))[:16]
    return str(int(time.time() * 1_000_000))


def hosted_vllm_alias(served_id: str) -> str:
    if not served_id:
        raise ValueError("served_id must be a non-empty string")
    return f"{_HOSTED_VLLM_PREFIX}{served_id}"


def maybe_int(value: Any) -> Optional[int]:
    """Parse a value as int, returning None if not possible."""
    if value in (None, "", "None"):
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def default_job_name(prefix: str, dataset_label: str, model_label: str) -> str:
    sanitized_dataset = Path(dataset_label).name.replace("/", "-").replace(" ", "_")
    sanitized_model = shorten_model_name(model_label)
    return JOB_NAME_SEP.join([prefix, sanitized_dataset, sanitized_model, _timestamp()])


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------


@dataclass
class ManagedProcess:
    """A subprocess with graceful shutdown support."""

    name: str
    proc: subprocess.Popen
    _log_handle: Optional[object] = field(default=None, repr=False)

    def stop(self, timeout: float = 10.0) -> None:
        if self.proc.poll() is not None:
            return
        try:
            self.proc.terminate()
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        finally:
            if self._log_handle:
                try:
                    self._log_handle.close()
                except Exception:
                    pass


def terminate_processes(processes: List[ManagedProcess]) -> None:
    for proc in processes:
        try:
            proc.stop()
        except Exception:
            pass


DEFAULT_FD_MONITOR_INTERVAL = 120


class FileDescriptorMonitor:
    """Background thread that periodically logs file descriptor usage."""

    def __init__(self, interval_seconds: int = DEFAULT_FD_MONITOR_INTERVAL):
        self.interval = interval_seconds
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _get_fd_usage(self) -> tuple:
        try:
            pid = os.getpid()
            fd_dir = Path(f"/proc/{pid}/fd")
            if fd_dir.exists():
                current_fds = len(list(fd_dir.iterdir()))
            else:
                current_fds = 0
                for fd in range(1024):
                    try:
                        os.fstat(fd)
                        current_fds += 1
                    except OSError:
                        pass
            soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
            percent_used = (current_fds / soft_limit * 100) if soft_limit > 0 else 0
            return current_fds, soft_limit, hard_limit, percent_used
        except Exception:
            return -1, -1, -1, 0.0

    def _log_status(self) -> None:
        current, soft, hard, percent = self._get_fd_usage()
        if current < 0:
            print("[fd-monitor] Unable to read file descriptor usage", flush=True)
            return
        if percent >= 90:
            level = "CRITICAL"
        elif percent >= 75:
            level = "WARNING"
        elif percent >= 50:
            level = "INFO"
        else:
            level = "OK"
        timestamp = time.strftime("%H:%M:%S")
        print(
            f"[fd-monitor] [{timestamp}] {level}: {current:,} / {soft:,} FDs open "
            f"({percent:.1f}% of soft limit)",
            flush=True,
        )

    def _run(self) -> None:
        self._log_status()
        while not self._stop_event.is_set():
            self._stop_event.wait(self.interval)
            if not self._stop_event.is_set():
                self._log_status()

    def start(self) -> None:
        if self._thread is not None:
            return
        if self.interval <= 0:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"[fd-monitor] Started monitoring (every {self.interval}s)", flush=True)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        self._log_status()
        print("[fd-monitor] Stopped", flush=True)


# ---------------------------------------------------------------------------
# Subprocess launchers
# ---------------------------------------------------------------------------


def _open_log_file(log_path: Optional[Path]) -> tuple:
    if os.environ.get("OT_AGENT_INHERIT_SUBPROC_LOGS") == "1":
        return None, None, None
    if log_path:
        log_file = open(log_path, "w", encoding="utf-8", buffering=1)
        return log_file, log_file, log_file
    return None, None, None


def start_ray(
    host: str,
    ray_port: int,
    num_gpus: int,
    num_cpus: int,
    log_path: Optional[Path] = None,
    memory: Optional[int] = None,
    object_store_memory: Optional[int] = None,
) -> ManagedProcess:
    """Start a single-node Ray cluster head."""
    if object_store_memory is None:
        object_store_memory = 40 * 1024 * 1024 * 1024

    cmd = [
        "ray", "start", "--head",
        f"--node-ip-address={host}",
        f"--port={ray_port}",
        f"--num-gpus={num_gpus}",
        f"--num-cpus={num_cpus}",
        "--dashboard-host=0.0.0.0",
        "--block",
    ]
    if memory is not None:
        cmd.append(f"--memory={memory}")
    if object_store_memory is not None:
        cmd.append(f"--object-store-memory={object_store_memory}")

    env = os.environ.copy()
    stdout, stderr, log_file = _open_log_file(log_path)
    popen = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, env=env)
    return ManagedProcess(name="ray", proc=popen, _log_handle=log_file)


def start_vllm_controller(
    model: str,
    host: str,
    ray_port: int,
    api_port: int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    data_parallel_size: int,
    endpoint_path: Path,
    controller_script: Path,
    log_path: Optional[Path] = None,
    served_model_name: Optional[str] = None,
    extra_cli_args: Optional[List[str]] = None,
    extra_env_vars: Optional[dict] = None,
) -> ManagedProcess:
    """Start a vLLM controller process."""
    env = os.environ.copy()
    env["VLLM_MODEL_PATH"] = model
    env["PYTHONUNBUFFERED"] = "1"
    if extra_env_vars:
        env.update(extra_env_vars)

    cmd = [
        sys.executable, str(controller_script),
        "--ray-address", f"{host}:{ray_port}",
        "--host", host,
        "--port", str(api_port),
        "--model", model,
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--pipeline-parallel-size", str(pipeline_parallel_size),
        "--data-parallel-size", str(data_parallel_size),
        "--endpoint-json", str(endpoint_path),
    ]
    if served_model_name:
        cmd.extend(["--served-model-name", served_model_name])
    if extra_cli_args:
        cmd.extend(extra_cli_args)

    stdout, stderr, log_file = _open_log_file(log_path)
    _saved_affinity = None
    try:
        _saved_affinity = os.sched_getaffinity(0)
        all_cpus = set(range(os.cpu_count() or 1))
        if _saved_affinity != all_cpus:
            os.sched_setaffinity(0, all_cpus)
    except (OSError, AttributeError):
        pass

    popen = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, env=env)

    if _saved_affinity is not None:
        try:
            os.sched_setaffinity(0, _saved_affinity)
        except (OSError, AttributeError):
            pass

    return ManagedProcess(name="vllm_controller", proc=popen, _log_handle=log_file)


def start_vllm_iris_controller(
    model: str,
    host: str,
    ray_port: int,
    api_port: int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    data_parallel_size: int,
    endpoint_path: Path,
    controller_script: Path,
    log_path: Optional[Path] = None,
    served_model_name: Optional[str] = None,
    extra_cli_args: Optional[List[str]] = None,
    extra_env_vars: Optional[dict] = None,
) -> ManagedProcess:
    """Start the iris multi-host vLLM controller."""
    env = os.environ.copy()
    env["VLLM_MODEL_PATH"] = model
    env["PYTHONUNBUFFERED"] = "1"
    if extra_env_vars:
        env.update(extra_env_vars)

    cmd = [
        sys.executable, str(controller_script),
        "--host", host,
        "--port", str(api_port),
        "--model", model,
        "--ray-port", str(ray_port),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--pipeline-parallel-size", str(pipeline_parallel_size),
        "--data-parallel-size", str(data_parallel_size),
        "--endpoint-json", str(endpoint_path),
    ]
    if served_model_name:
        cmd.extend(["--served-model-name", served_model_name])
    if extra_cli_args:
        cmd.extend(extra_cli_args)

    stdout, stderr, log_file = _open_log_file(log_path)
    popen = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, env=env)
    return ManagedProcess(name="vllm_iris_controller", proc=popen, _log_handle=log_file)


def wait_for_endpoint(
    endpoint_path: Path,
    controller: ManagedProcess,
    timeout: int = 300,
) -> None:
    """Wait for the vLLM endpoint JSON file to be created."""
    start = time.time()
    while time.time() - start < timeout:
        if controller.proc.poll() is not None:
            raise RuntimeError(
                "vLLM controller exited before writing the endpoint JSON. Check logs."
            )
        if endpoint_path.exists():
            return
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for endpoint JSON at {endpoint_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cli_has_option(*flags: str) -> bool:
    for arg in sys.argv[1:]:
        token = arg.split("=", 1)[0]
        if token in flags:
            return True
    return False


def _extract_injected_jobs_dir(harbor_extra_args: Optional[List[str]]) -> Optional[str]:
    """Return the last ``--jobs-dir`` value injected via ``--harbor_extra_arg``."""
    if not harbor_extra_args:
        return None
    found: Optional[str] = None
    tokens = list(harbor_extra_args)
    for i, tok in enumerate(tokens):
        if tok.startswith("--jobs-dir="):
            found = tok.split("=", 1)[1]
        elif tok == "--jobs-dir" and i + 1 < len(tokens):
            found = tokens[i + 1]
    return found or None


# ---------------------------------------------------------------------------
# LocalHarborRunner base class (eval-only)
# ---------------------------------------------------------------------------


class LocalHarborRunner:
    """Base class for local Harbor runners.

    Encapsulates the eval workflow:
    1. Parse/validate arguments
    2. Start Ray + vLLM (or use iris multi-host serve)
    3. Wait for endpoint
    4. Build + run Harbor command
    5. post_harbor_hook (subclass override for uploads)
    6. Clean up

    Subclasses override JOB_PREFIX, DEFAULT_EXPERIMENTS_SUBDIR,
    DEFAULT_N_CONCURRENT, get_env_type(), get_dataset_label(),
    get_dataset_for_harbor(), post_harbor_hook(), etc.
    """

    JOB_PREFIX: str = "job"
    DEFAULT_EXPERIMENTS_SUBDIR: str = "runs"
    DEFAULT_N_CONCURRENT: int = 16
    DEFAULT_ENDPOINT_FILENAME: str = "vllm_endpoint.json"
    TPU_SERVE_DEFAULT_CLI_ARGS: List[str] = []

    def __init__(self, args: argparse.Namespace, repo_root: Path):
        self.args = args
        self.repo_root = repo_root
        self.processes: List[ManagedProcess] = []
        self._endpoint_json: Optional[Path] = None
        self._endpoint_meta: Optional[Dict[str, Any]] = None
        self._harbor_job_name: Optional[str] = None
        self._fd_monitor: Optional[FileDescriptorMonitor] = None

    @classmethod
    def add_common_arguments(cls, parser: argparse.ArgumentParser) -> None:
        add_harbor_args(parser, config_required=True)
        add_model_compute_args(
            parser,
            model_required=False,
            default_n_concurrent=cls.DEFAULT_N_CONCURRENT,
            default_n_attempts=1,
            n_attempts_help="Times to run each task for repeated trials (default: 1).",
        )
        add_ray_vllm_args(parser)
        add_log_path_args(parser)

        parser.add_argument("--cpus", type=int, help="CPUs to expose to Ray.")
        parser.add_argument("--endpoint-json", help="Optional endpoint JSON path.")
        parser.add_argument(
            "--fd_monitor_interval",
            type=int,
            default=DEFAULT_FD_MONITOR_INTERVAL,
            metavar="SECONDS",
            help=f"Interval for file descriptor monitoring (default: {DEFAULT_FD_MONITOR_INTERVAL}s). Set to 0 to disable.",
        )
        parser.add_argument("--fd-monitor-interval", dest="fd_monitor_interval", help=argparse.SUPPRESS)

    def get_env_type(self) -> str:
        raise NotImplementedError("Subclasses must implement get_env_type()")

    def get_dataset_label(self) -> str:
        raise NotImplementedError("Subclasses must implement get_dataset_label()")

    def get_dataset_for_harbor(self) -> Tuple[Optional[str], Optional[str]]:
        raise NotImplementedError("Subclasses must implement get_dataset_for_harbor()")

    def get_experiments_dir(self) -> Path:
        if hasattr(self.args, "experiments_dir") and self.args.experiments_dir:
            return Path(self.args.experiments_dir).expanduser().resolve()
        return self.repo_root / self.DEFAULT_EXPERIMENTS_SUBDIR

    def validate_args(self) -> None:
        pass

    def post_harbor_hook(self) -> None:
        pass

    def print_banner(self) -> None:
        args = self.args
        print(f"=== Local {self.JOB_PREFIX.title()} Runner ===")
        print(f"  Model: {args.model}")
        print(f"  TP/PP/DP: {args.tensor_parallel_size}/{args.pipeline_parallel_size}/{args.data_parallel_size}")
        print(f"  GPUs: {args.gpus}")
        print("=" * 35)

    def setup(self) -> None:
        """Set up the runner — apply defaults, configure environment."""
        args = self.args

        # Initialize eval defaults (no datagen config path in the eval-only runner)
        args._vllm_cli_args: List[str] = []
        args._vllm_env_vars: Dict[str, str] = {}
        args._engine_type = "vllm_local"
        args._needs_local_vllm = True
        args._extra_agent_kwargs: Dict[str, Any] = {}

        # Apply datagen config defaults if provided (eval's --datagen_config)
        datagen_config = getattr(args, "datagen_config", None)
        if datagen_config:
            self._apply_datagen_config(args, datagen_config)

        # Set up Docker runtime if using docker backend
        setup_docker_runtime_if_needed(self.get_env_type())

        # Resolve per-model serve config
        self._apply_model_config(args)

        # Set parallelism defaults
        _iris_serve = os.environ.get("OT_AGENT_IRIS_SERVE") == "1"
        if args.tensor_parallel_size is None:
            if _iris_serve and getattr(args, "gpus", None):
                args.tensor_parallel_size = int(args.gpus)
            else:
                args.tensor_parallel_size = 1
        if args.pipeline_parallel_size is None:
            args.pipeline_parallel_size = 1
        if args.data_parallel_size is None:
            args.data_parallel_size = 1

        needs_local_vllm = getattr(args, "_needs_local_vllm", True)
        if args.model is None and needs_local_vllm:
            raise ValueError("Provide --model or supply a datagen config with vllm_server.model_path.")

        # Generate served model ID (deterministic per job_name)
        if needs_local_vllm:
            served_model_id = generate_served_model_id(job_name=args.job_name)
            args._served_model_id = served_model_id
            args._harbor_model_name = hosted_vllm_alias(served_model_id)
        else:
            args._served_model_id = None
            args._harbor_model_name = args.model

        # Set GPU/CPU defaults
        if args.gpus is None:
            args.gpus = max(1, args.tensor_parallel_size * args.pipeline_parallel_size * args.data_parallel_size)
        if args.cpus is None:
            args.cpus = os.cpu_count() or 16

        # Set port defaults
        if args.ray_port is None:
            args.ray_port = 6379
        if args.api_port is None:
            args.api_port = 8000

        # Resolve paths
        args.harbor_config = str(Path(args.harbor_config).expanduser().resolve())

        harbor_config_data = load_harbor_config(args.harbor_config)
        jobs_dir_value = harbor_config_data.get("jobs_dir") if isinstance(harbor_config_data, dict) else None
        args._jobs_dir_path = resolve_jobs_dir_path(jobs_dir_value, self.repo_root)
        injected_jobs_dir = _extract_injected_jobs_dir(getattr(args, "harbor_extra_arg", None))
        if injected_jobs_dir and "://" not in injected_jobs_dir:
            args._jobs_dir_path = Path(injected_jobs_dir).expanduser()
            print(
                f"[{self.JOB_PREFIX}-local] jobs-dir override: in-pod upload will read "
                f"{args._jobs_dir_path}/<job_name>",
                flush=True,
            )
        args._harbor_config_data = harbor_config_data

        # Load structured JobConfig
        harbor_job = load_job_config(args.harbor_config)
        args._harbor_job_config = harbor_job

        # Apply n_concurrent from harbor config if CLI didn't override
        config_n_concurrent = get_orchestrator_field(harbor_job, "n_concurrent_trials")
        if config_n_concurrent is not None and config_n_concurrent > 0:
            if (
                not _cli_has_option("--n_concurrent", "--n-concurrent")
                and getattr(args, "n_concurrent", None) == self.DEFAULT_N_CONCURRENT
            ):
                args.n_concurrent = int(config_n_concurrent)

        # Apply n_attempts from harbor config if CLI didn't override
        config_n_attempts = getattr(harbor_job, "n_attempts", None)
        if config_n_attempts is not None and config_n_attempts > 0:
            if (
                not _cli_has_option("--n_attempts", "--n-attempts")
                and getattr(args, "n_attempts", 1) == 1
            ):
                args.n_attempts = int(config_n_attempts)

        self.validate_args()

    def _apply_datagen_config(self, args: argparse.Namespace, datagen_config: str) -> None:
        """Load a datagen YAML and seed vllm_server / model defaults."""
        from ..harness.config import load_harbor_config as _load_yaml
        import dataclasses

        parsed = _load_yaml(datagen_config)
        if not isinstance(parsed, dict):
            return

        vllm_server = parsed.get("vllm_server") or {}
        if isinstance(vllm_server, dict):
            cli_args, env_vars = _build_vllm_cli_args(vllm_server)
            args._vllm_cli_args = cli_args
            args._vllm_env_vars = env_vars

        engine = parsed.get("engine") or {}
        if isinstance(engine, dict):
            engine_type = engine.get("type")
            if engine_type and engine_type != "vllm_local":
                args._engine_type = engine_type
                args._needs_local_vllm = False
            model_path = engine.get("model_path")
            if model_path and getattr(args, "model", None) is None:
                args.model = model_path

    def _apply_model_config(self, args: argparse.Namespace) -> None:
        """Resolve per-model serve config from the model_config registry."""
        if not getattr(args, "model", None):
            return
        from ..serve.model_config import resolve_model_config

        resolved = resolve_model_config(args.model, subsystem="eval")
        if not resolved:
            return

        # Apply resolved vllm_server fields to _vllm_cli_args
        merged_server_config = dict(resolved)
        cli_args, env_vars = _build_vllm_cli_args(merged_server_config)
        # Merge (don't overwrite datagen-config values)
        for arg in cli_args:
            if arg not in args._vllm_cli_args:
                args._vllm_cli_args.append(arg)
        for k, v in env_vars.items():
            args._vllm_env_vars.setdefault(k, v)

    def _setup_directories(self) -> Tuple[Path, Path]:
        experiments_dir = self.get_experiments_dir()
        experiments_dir.mkdir(parents=True, exist_ok=True)
        logs_dir = experiments_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        return experiments_dir, logs_dir

    def _setup_signal_handlers(self) -> None:
        import signal as sig

        def _handle_signal(signum, _frame):
            print(f"\nSignal {signum} received; shutting down...", file=sys.stderr)
            self.cleanup()
            sys.exit(1)

        sig.signal(sig.SIGINT, _handle_signal)
        sig.signal(sig.SIGTERM, _handle_signal)

    def cleanup(self) -> None:
        if self._fd_monitor is not None:
            self._fd_monitor.stop()
            self._fd_monitor = None

        terminate_processes(self.processes[::-1])
        needs_local_vllm = getattr(self.args, "_needs_local_vllm", True)
        iris_serve = os.environ.get("OT_AGENT_IRIS_SERVE") == "1"
        if needs_local_vllm and not iris_serve:
            subprocess.run(["ray", "stop", "--force"], check=False)

    def run(self) -> None:
        """Main entry point — start services and run Harbor."""
        args = self.args
        needs_local_vllm = getattr(args, "_needs_local_vllm", True)

        iris_serve = needs_local_vllm and os.environ.get("OT_AGENT_IRIS_SERVE") == "1"
        iris_rank = int(os.environ.get("IRIS_TASK_ID", "0").rsplit("/", 1)[-1].split(":", 1)[0])

        experiments_dir, logs_dir = self._setup_directories()

        self._endpoint_json = Path(args.endpoint_json or (experiments_dir / self.DEFAULT_ENDPOINT_FILENAME))
        if self._endpoint_json.exists():
            self._endpoint_json.unlink()

        os.chdir(self.repo_root)

        ray_log = Path(args.ray_log) if args.ray_log else logs_dir / "ray.log"
        controller_log = Path(args.controller_log) if args.controller_log else logs_dir / "vllm_controller.log"
        harbor_log = Path(args.harbor_log).expanduser().resolve() if args.harbor_log else None

        self._setup_signal_handlers()
        self.print_banner()

        fd_interval = getattr(args, "fd_monitor_interval", DEFAULT_FD_MONITOR_INTERVAL)
        if fd_interval > 0:
            self._fd_monitor = FileDescriptorMonitor(interval_seconds=fd_interval)
            self._fd_monitor.start()

        vllm_proc: Optional[ManagedProcess] = None

        if iris_serve:
            _serve_model = getattr(args, "vllm_model_uri", None) or args.model
            vllm_proc = start_vllm_iris_controller(
                model=_serve_model,
                host=args.host,
                ray_port=args.ray_port,
                api_port=args.api_port,
                tensor_parallel_size=args.tensor_parallel_size,
                pipeline_parallel_size=args.pipeline_parallel_size,
                data_parallel_size=args.data_parallel_size,
                endpoint_path=self._endpoint_json,
                controller_script=self.repo_root / "scripts" / "vllm" / "start_vllm_iris_controller.py",
                log_path=controller_log,
                served_model_name=getattr(args, "_served_model_id", None),
                extra_cli_args=add_tpu_serve_default_flags(
                    drop_tpu_unsupported_serve_flags(getattr(args, "_vllm_cli_args", [])),
                    self.TPU_SERVE_DEFAULT_CLI_ARGS,
                ),
                extra_env_vars=getattr(args, "_vllm_env_vars", {}),
            )
            self.processes.append(vllm_proc)
        elif needs_local_vllm:
            controller_script = self.repo_root / "scripts" / "vllm" / "start_vllm_ray_controller.py"

            ray_memory = None
            if getattr(args, "ray_memory_gb", None) is not None:
                ray_memory = int(args.ray_memory_gb * 1024 * 1024 * 1024)
            ray_object_store = int(getattr(args, "ray_object_store_gb", 40.0) * 1024 * 1024 * 1024)

            ray_proc = start_ray(
                host=args.host,
                ray_port=args.ray_port,
                num_gpus=args.gpus,
                num_cpus=args.cpus,
                log_path=ray_log,
                memory=ray_memory,
                object_store_memory=ray_object_store,
            )
            self.processes.append(ray_proc)

            vllm_proc = start_vllm_controller(
                model=args.model,
                host=args.host,
                ray_port=args.ray_port,
                api_port=args.api_port,
                tensor_parallel_size=args.tensor_parallel_size,
                pipeline_parallel_size=args.pipeline_parallel_size,
                data_parallel_size=args.data_parallel_size,
                endpoint_path=self._endpoint_json,
                controller_script=controller_script,
                log_path=controller_log,
                served_model_name=getattr(args, "_served_model_id", None),
                extra_cli_args=getattr(args, "_vllm_cli_args", []),
                extra_env_vars=getattr(args, "_vllm_env_vars", {}),
            )
            self.processes.append(vllm_proc)
        else:
            print(f"[engine] Using {getattr(args, '_engine_type', 'unknown')} API engine - skipping Ray/vLLM startup")

        try:
            # iris worker ranks: block as Ray worker, skip harbor
            if iris_serve and iris_rank != 0:
                print(
                    f"[iris] Worker rank {iris_rank}: acting as Ray worker node; "
                    "blocking on controller.",
                    flush=True,
                )
                vllm_proc.proc.wait()
                return

            if needs_local_vllm and vllm_proc is not None:
                endpoint_timeout = 1200 if iris_serve else 300
                wait_for_endpoint(self._endpoint_json, vllm_proc, timeout=endpoint_timeout)
                run_endpoint_health_check(
                    self._endpoint_json,
                    args.health_max_attempts,
                    args.health_retry_delay,
                    self.repo_root,
                )
                self._endpoint_meta = load_endpoint_metadata(self._endpoint_json)
            else:
                self._endpoint_meta = None

            # Compute job name
            harbor_model = getattr(args, "_harbor_model_name", args.model)
            job_model_label = args.model or harbor_model or "model"
            dataset_label = self.get_dataset_label()
            job_name = args.job_name or default_job_name(self.JOB_PREFIX, dataset_label, job_model_label)
            self._harbor_job_name = job_name
            args._harbor_job_name = job_name

            dataset_slug, dataset_path = self.get_dataset_for_harbor()

            # Re-fire: prune infra-errored trials before auto-resume
            refire_types = getattr(args, "refire_filter_error_types", None)
            if refire_types and not args.dry_run:
                refire_root = _extract_injected_jobs_dir(
                    getattr(args, "harbor_extra_arg", None)
                ) or (str(getattr(args, "_jobs_dir_path", "") or "") or None)
                if refire_root:
                    refire_run_dir = f"{refire_root.rstrip('/')}/{job_name}"
                    prune_refire_errored_trials(
                        refire_run_dir,
                        list(refire_types),
                        log_prefix=f"[{self.JOB_PREFIX}-local] ",
                    )
            elif refire_types and args.dry_run:
                print(
                    f"[{self.JOB_PREFIX}-local][refire] --dry_run: would prune "
                    f"trials with error types {sorted(set(refire_types))}.",
                    flush=True,
                )

            # Build Harbor command
            harbor_cmd = build_harbor_command(
                harbor_binary=args.harbor_binary,
                harbor_config_path=args.harbor_config,
                harbor_config_data=getattr(args, "_harbor_config_data", {}),
                job_name=job_name,
                agent_name=args.agent,
                model_name=harbor_model,
                env_type=self.get_env_type(),
                n_concurrent=args.n_concurrent,
                n_attempts=args.n_attempts,
                endpoint_meta=self._endpoint_meta,
                agent_kwarg_overrides=list(args.agent_kwarg or []),
                harbor_extra_args=list(args.harbor_extra_arg or []),
                dataset_slug=dataset_slug,
                dataset_path=dataset_path,
                jobs_dir=str(getattr(args, "_jobs_dir_path", None) or ""),
                extra_agent_kwargs=getattr(args, "_extra_agent_kwargs", None),
                export_hf_repo=getattr(args, "upload_hf_repo", None),
            )
            print("Harbor command:", " ".join(harbor_cmd))

            if not args.dry_run:
                run_harbor_cli(harbor_cmd, harbor_log)
                self.post_harbor_hook()
            else:
                print(f"[dry-run] Would run Harbor {self.JOB_PREFIX} job.")

        finally:
            self.cleanup()


__all__ = [
    "ManagedProcess",
    "FileDescriptorMonitor",
    "start_ray",
    "start_vllm_controller",
    "start_vllm_iris_controller",
    "wait_for_endpoint",
    "terminate_processes",
    "LocalHarborRunner",
    "DEFAULT_FD_MONITOR_INTERVAL",
    "generate_served_model_id",
    "hosted_vllm_alias",
    "maybe_int",
    "default_job_name",
    "shorten_model_name",
]
