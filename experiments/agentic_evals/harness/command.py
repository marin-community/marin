"""Harbor command building + execution.

Extracted from OT-Agent ``hpc/harbor_utils.py``. Imports ``merge_agent_kwargs``
from ``.config`` and ``_filter_supported_metrics`` from ``.job_config``.
"""

from __future__ import annotations

import copy
import errno
import json
import os
import pty
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .config import merge_agent_kwargs
from .job_config import _filter_supported_metrics


def build_endpoint_meta(endpoint_url: str, api_key: Optional[str] = None) -> Dict[str, str]:
    """Build endpoint metadata dict from a vLLM endpoint URL.

    Handles both formats:
    - With /v1 suffix: "http://host:port/v1"
    - Without suffix: "http://host:port"
    """
    url = endpoint_url.rstrip("/")

    if url.endswith("/v1"):
        base_url = url[:-3].rstrip("/")
        api_base = url
    else:
        base_url = url
        api_base = f"{url}/v1"

    metrics_endpoint = f"{base_url}/metrics"

    meta: Dict[str, str] = {
        "api_base": api_base,
        "metrics_endpoint": metrics_endpoint,
    }
    if api_key:
        meta["api_key"] = api_key
    return meta


def load_endpoint_metadata(endpoint_json: Path) -> Dict[str, Any]:
    """Load and parse vLLM endpoint metadata from JSON file."""
    data = json.loads(endpoint_json.read_text())
    endpoint_url = data.get("endpoint_url") or ""

    if endpoint_url:
        meta = build_endpoint_meta(endpoint_url)
        data["api_base"] = meta["api_base"]
        data["metrics_endpoint"] = meta["metrics_endpoint"]
    else:
        data["api_base"] = ""
        data["metrics_endpoint"] = ""

    return data


def merge_harbor_config(
    harbor_config_data: dict,
    *,
    agent_name: Optional[str],
    model_name: str,
    n_concurrent: int,
    endpoint_meta: Optional[dict],
    agent_kwarg_overrides: List[str],
    extra_agent_kwargs: Optional[Dict[str, Any]] = None,
) -> dict:
    """Materialize the merged Harbor config dict without writing files."""
    agent_kwargs, _ = merge_agent_kwargs(
        harbor_config_data=harbor_config_data,
        agent_name=agent_name,
        endpoint_meta=endpoint_meta,
        extra_kwargs=extra_agent_kwargs,
        cli_overrides=agent_kwarg_overrides,
    )

    modified_config = copy.deepcopy(harbor_config_data)

    if "orchestrator" not in modified_config:
        modified_config["orchestrator"] = {}
    modified_config["orchestrator"]["n_concurrent_trials"] = n_concurrent

    agents = modified_config.get("agents", [])
    for agent in agents:
        if agent_name:
            agent["name"] = agent_name
        agent["model_name"] = model_name
        existing_kwargs = agent.get("kwargs", {})
        existing_kwargs.update(agent_kwargs)
        agent["kwargs"] = existing_kwargs

    return modified_config


def _sync_runtime_fields_into_config_json(
    *,
    config_json_path: Path,
    modified_config: dict,
    extra_args: List[str],
    n_concurrent: int,
    n_attempts: int,
) -> None:
    """Patch on-disk config.json so it matches the YAML+CLI effective state."""
    cj = json.loads(config_json_path.read_text())

    yaml_agents = modified_config.get("agents") or []
    cj_agents = cj.get("agents") or []
    runtime_kwarg_keys = ("api_base", "api_key", "base_url", "metrics_endpoint")
    for i, yaml_agent in enumerate(yaml_agents):
        if i >= len(cj_agents):
            break
        if not isinstance(yaml_agent, dict) or not isinstance(cj_agents[i], dict):
            continue
        cj_agent = cj_agents[i]
        yaml_kw = yaml_agent.get("kwargs") or {}
        cj_kw = cj_agent.setdefault("kwargs", {})
        for key in runtime_kwarg_keys:
            if key in yaml_kw:
                cj_kw[key] = yaml_kw[key]
        if "model_name" in yaml_agent:
            cj_agent["model_name"] = yaml_agent["model_name"]

    orchestrator = cj.setdefault("orchestrator", {})
    orchestrator["n_concurrent_trials"] = n_concurrent
    cj["n_attempts"] = n_attempts

    yaml_orchestrator = modified_config.get("orchestrator") or {}
    yaml_retry = yaml_orchestrator.get("retry") if isinstance(yaml_orchestrator, dict) else None
    if isinstance(yaml_retry, dict):
        cj_retry = orchestrator.setdefault("retry", {})
        for k in ("include_exceptions", "exclude_exceptions", "mask_exceptions", "passthrough_exceptions"):
            if k in yaml_retry:
                cj_retry[k] = yaml_retry[k]

    config_json_path.write_text(json.dumps(cj, indent=2))


def build_harbor_command(
    harbor_binary: str,
    harbor_config_path: str,
    harbor_config_data: dict,
    job_name: str,
    agent_name: Optional[str],
    model_name: str,
    env_type: str,
    n_concurrent: int,
    n_attempts: int,
    endpoint_meta: Optional[dict],
    agent_kwarg_overrides: List[str],
    harbor_extra_args: List[str],
    dataset_slug: Optional[str] = None,
    dataset_path: Optional[str] = None,
    jobs_dir: Optional[str] = None,
    extra_agent_kwargs: Optional[Dict[str, Any]] = None,
    export_hf_repo: Optional[str] = None,
) -> List[str]:
    """Build the ``harbor jobs start`` command."""
    _, passthrough = merge_agent_kwargs(
        harbor_config_data=harbor_config_data,
        agent_name=agent_name,
        endpoint_meta=endpoint_meta,
        extra_kwargs=extra_agent_kwargs,
        cli_overrides=agent_kwarg_overrides,
    )

    modified_config = merge_harbor_config(
        harbor_config_data,
        agent_name=agent_name,
        model_name=model_name,
        n_concurrent=n_concurrent,
        endpoint_meta=endpoint_meta,
        agent_kwarg_overrides=agent_kwarg_overrides,
        extra_agent_kwargs=extra_agent_kwargs,
    )

    if jobs_dir:
        config_dir = Path(jobs_dir) / job_name
        config_dir.mkdir(parents=True, exist_ok=True)
        merged_config_path = config_dir / "merged_harbor_config.yaml"
    else:
        merged_config_path = Path(f"merged_harbor_config_{job_name}.yaml")

    modified_config = _filter_supported_metrics(modified_config)

    with open(merged_config_path, "w") as f:
        yaml.safe_dump(modified_config, f)
    temp_config_path = str(merged_config_path)

    cmd = [
        harbor_binary,
        "jobs",
        "start",
        "--yes",
        "--config",
        temp_config_path,
        "--job-name",
        job_name,
        "--env",
        env_type,
        "--n-concurrent",
        str(n_concurrent),
        "--n-attempts",
        str(n_attempts),
    ]

    if dataset_slug:
        modified_config.pop("datasets", None)
        modified_config.pop("tasks", None)
        with open(merged_config_path, "w") as f:
            yaml.safe_dump(modified_config, f)
        cmd.extend(["--dataset", dataset_slug])
    elif dataset_path:
        modified_config.pop("datasets", None)
        modified_config.pop("tasks", None)
        with open(merged_config_path, "w") as f:
            yaml.safe_dump(modified_config, f)
        looks_local = not dataset_path.startswith(("gs://", "s3://", "http://", "https://"))
        if looks_local and Path(dataset_path).expanduser().exists():
            cmd.extend(["--path", str(Path(dataset_path).expanduser().resolve())])
        else:
            cmd.extend(["--dataset", dataset_path])
    else:
        _placeholder = "/replace/with/tasks/path"
        yaml_datasets = modified_config.get("datasets") or []
        if yaml_datasets and any(
            d.get("path", "") == _placeholder for d in yaml_datasets if isinstance(d, dict)
        ):
            modified_config.pop("datasets", None)
            modified_config.pop("tasks", None)
            with open(merged_config_path, "w") as f:
                yaml.safe_dump(modified_config, f)

        has_datasets = bool(modified_config.get("datasets"))
        has_tasks = bool(modified_config.get("tasks"))
        has_cli_dataset = "--dataset" in cmd
        if not (has_datasets or has_tasks or has_cli_dataset):
            raise ValueError(
                "[build_harbor_command] BUG: No datasets, tasks, or --dataset flag. "
                f"dataset_slug={dataset_slug!r}, dataset_path={dataset_path!r}."
            )

    if jobs_dir:
        cmd.extend(["--jobs-dir", jobs_dir])

    for passthrough_kw in passthrough:
        cmd.extend(["--agent-kwarg", passthrough_kw])

    extra_args = list(harbor_extra_args or [])

    def _flag_present(flag: str) -> bool:
        return any(arg == flag or arg.startswith(f"{flag}=") for arg in extra_args)

    if not (_flag_present("--export-traces") or _flag_present("--no-export-traces")):
        extra_args.append("--export-traces")
    if not (_flag_present("--export-verifier-metadata") or _flag_present("--no-export-verifier-metadata")):
        extra_args.append("--export-verifier-metadata")
    if not _flag_present("--export-episodes"):
        extra_args.extend(["--export-episodes", "last"])

    if export_hf_repo and not _flag_present("--export-repo"):
        if not _flag_present("--export-push"):
            extra_args.append("--export-push")
        extra_args.extend(["--export-repo", export_hf_repo])

    for extra in extra_args:
        cmd.append(extra)

    if jobs_dir:
        config_json_path = Path(jobs_dir) / job_name / "config.json"
        if config_json_path.exists():
            print(
                f"[build_harbor_command] Syncing runtime fields into {config_json_path}",
                flush=True,
            )
            try:
                _sync_runtime_fields_into_config_json(
                    config_json_path=config_json_path,
                    modified_config=modified_config,
                    extra_args=extra_args,
                    n_concurrent=n_concurrent,
                    n_attempts=n_attempts,
                )
                with open(config_json_path, "rb") as _f:
                    os.fsync(_f.fileno())
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                print(
                    f"[build_harbor_command] WARNING: could not sync runtime "
                    f"fields into {config_json_path}: {exc}",
                    file=sys.stderr, flush=True,
                )

    return cmd


def run_harbor_cli(cmd: List[str], log_path: Optional[Path] = None) -> int:
    """Run Harbor CLI with proper TTY handling."""
    if log_path:
        with open(log_path, "w", encoding="utf-8", buffering=1) as harbor_log_file:
            print(f"Streaming Harbor output to {log_path}")
            result = subprocess.run(cmd, check=False, stdout=harbor_log_file, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, cmd)
        return result.returncode

    master_fd, slave_fd = pty.openpty()
    try:
        proc = subprocess.Popen(cmd, stdin=slave_fd, stdout=slave_fd, stderr=slave_fd, text=False)
        os.close(slave_fd)
        while True:
            try:
                data = os.read(master_fd, 4096)
            except OSError as exc:
                if exc.errno != errno.EIO:
                    raise
                break
            if not data:
                break
            os.write(sys.stdout.fileno(), data)
    finally:
        os.close(master_fd)

    ret = proc.wait()
    if ret != 0:
        raise subprocess.CalledProcessError(ret, cmd)
    return ret
