#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launcher CLI: submit an agentic eval job to a cluster.

Adapted from OT-Agent ``eval/cloud/launch_eval_iris.py``. Parses CLI args
(using ``runtime.args`` helpers), applies presets, resolves model config,
and delegates to ``backends.iris.IrisBackend.submit()``.

Usage:
    python -m agentic_evals.launch \\
        --harbor_config harbor.yaml \\
        --model Qwen/Qwen3-32B \\
        --dataset_path ./tasks \\
        --preset tb2 \\
        --tpu v6e-4 \\
        --cluster-config /path/to/iris.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

from .harness.config import get_harbor_env_from_config, load_harbor_config
from .backends.federated_iris import (
    DEFAULT_PARENT_CLUSTER,
    DEFAULT_PARENT_INGRESS,
    DUMMY_API_KEY,
    build_marin_serve_command,
    durable_harbor_jobs_dir,
    mint_external_api_base,
)
from .results.infra_errors import INFRA_ERROR_TYPES
from .runtime.args import (
    add_database_upload_args,
    add_harbor_args,
    add_harbor_env_arg,
    add_hf_upload_args,
    add_model_compute_args,
)
from .serve.model_config import resolve_model_config

DEFAULT_REFIRE_ERROR_TYPES = sorted(set(INFRA_ERROR_TYPES) | {"DaytonaValidationError", "VerifierTimeoutError"})
DEFAULT_TPU_DISK = "100GB"
DEFAULT_GPU_DISK = "512GB"
DEFAULT_GRUG_SERVE_MODEL = "laion/grug-67b-a2b-sft-s3-agentic-step1903"
DEFAULT_GRUG_S3_OUTPUT_ROOT = "s3://marin-us-east-02a/iris"


def _load_presets() -> Dict[str, dict]:
    """Load preset YAMLs from the presets/ directory."""
    import yaml

    preset_dir = Path(__file__).resolve().parent / "presets"
    presets: Dict[str, dict] = {}
    for path in sorted(preset_dir.glob("*.yaml")):
        with path.open() as f:
            presets[path.stem] = yaml.safe_load(f) or {}
    return presets


def _cli_has(*flags: str) -> bool:
    for arg in sys.argv[1:]:
        token = arg.split("=", 1)[0]
        if token in flags:
            return True
    return False


def _workspace_relative(path_value: str, workspace: Path) -> str:
    """Translate a local launch-host path into the worker-visible `/app` path."""
    path = Path(path_value).expanduser().resolve()
    try:
        return str(path.relative_to(workspace.resolve()))
    except ValueError as error:
        raise ValueError(f"local path {path} is outside the bundled workspace {workspace}") from error


def _prepare_worker_paths(args: argparse.Namespace, workspace: Path | None = None) -> None:
    """Replace host paths with paths visible inside the Iris `/app` workspace.

    Normalization must retain host paths while it reads the Harbor configuration.
    This conversion is deliberately deferred until immediately before submission.
    """
    workspace = workspace or Path.cwd()
    args.harbor_config = _workspace_relative(args.harbor_config, workspace)
    if args.datagen_config:
        args.datagen_config = _workspace_relative(args.datagen_config, workspace)
    if args.dataset_path and not args.dataset_path.startswith(("gs://", "s3://", "http://", "https://")):
        candidate = Path(args.dataset_path).expanduser()
        if candidate.exists():
            args.dataset_path = _workspace_relative(args.dataset_path, workspace)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch an agentic eval job on a cluster (Iris TPU/GPU).")

    # Harbor + model/compute + upload args
    add_harbor_args(parser, config_required=True)
    add_model_compute_args(
        parser,
        model_required=False,
        default_n_concurrent=16,
        default_n_attempts=3,
        n_attempts_help="Times to run each task for standard error calculation (default: 3).",
    )
    add_harbor_env_arg(parser, default="daytona", legacy_names=["--eval-env", "--eval_env"])
    add_hf_upload_args(parser)
    add_database_upload_args(parser)

    # Dataset selection
    parser.add_argument("--dataset", help="Harbor dataset slug.")
    parser.add_argument("--dataset_path", help="Path to tasks directory.")
    parser.add_argument("--dataset-path", dest="dataset_path", help=argparse.SUPPRESS)

    # Preset
    presets = _load_presets()
    parser.add_argument(
        "--preset",
        choices=sorted(presets.keys()),
        default=None,
        help="Eval preset from presets/ (seeds dataset_path, n_concurrent, agent kwargs).",
    )

    # Datagen config (optional, for model inference)
    parser.add_argument("--datagen_config", help="Optional datagen YAML to seed defaults.")
    parser.add_argument("--datagen-config", dest="datagen_config", help=argparse.SUPPRESS)

    # Re-fire filter
    parser.add_argument(
        "--refire_filter_error_type",
        "--refire-filter-error-type",
        dest="refire_filter_error_types",
        action="append",
        default=None,
        help="Infra exception type to delete-and-re-run on a warm-dir re-fire (repeatable).",
    )

    # Iris backend args
    iris = parser.add_argument_group("iris")
    iris.add_argument("--cluster-config", "--cluster_config", default=None, help="Path to the iris cluster YAML.")
    iris.add_argument("--task-image", "--task_image", default=None, help="Container image for the task.")
    iris.add_argument("--tpu", default="v6e-4", help="TPU variant.")
    iris.add_argument("--gpu", default=None, help="GPU variant (e.g. H100x8).")
    iris.add_argument("--replicas", type=int, default=1)
    iris.add_argument("--cpu", type=float, default=8.0)
    iris.add_argument("--memory", default="256GB")
    iris.add_argument("--disk", default=None)
    iris.add_argument("--priority", default="interactive", choices=["production", "interactive", "batch"])
    iris.add_argument("--max-retries", "--max_retries", type=int, default=0)
    iris.add_argument("--timeout", type=int, default=0)
    iris.add_argument("--no-wait", "--no_wait", dest="no_wait", action="store_true", default=False)
    iris.add_argument("--secrets-env", "--secrets_env", default=None, help="Path to a KEY=VALUE env file.")
    iris.add_argument("--target-cluster", default=None, help="Federate the whole eval root job to this Iris peer.")

    external = parser.add_argument_group("separately served endpoint")
    external.add_argument("--external-profile", choices=["grug"], default=None)
    external.add_argument("--external-endpoint", help="Existing parent-controller endpoint name to wait for and mint.")
    external.add_argument("--external-serve-model", help="Start this model with marin-serve before evaluating.")
    external.add_argument("--external-serve-name", help="Serving job name (default: <job-name>-serve).")
    external.add_argument("--external-endpoint-name", help="Endpoint name (default: /serve/<serve-name>).")
    external.add_argument("--external-parent-cluster", default=DEFAULT_PARENT_CLUSTER)
    external.add_argument("--external-parent-ingress-host", default=DEFAULT_PARENT_INGRESS)
    external.add_argument("--iris-bin", default=str(Path(sys.executable).with_name("iris")))
    external.add_argument("--external-ttl-hours", type=float, default=24.0)
    external.add_argument("--external-ready-timeout-seconds", type=float, default=1800.0)
    external.add_argument("--external-serve-idle-timeout-hours", type=float, default=1.0)
    external.add_argument("--external-serve-max-model-len", type=int, default=None)
    external.add_argument("--external-serve-max-num-batched-tokens", type=int, default=None)
    external.add_argument("--external-serve-tensor-parallel-size", type=int, default=None)
    external.add_argument("--external-serve-data-parallel-size", type=int, default=None)
    external.add_argument("--external-serve-max-num-seqs", type=int, default=None)
    external.add_argument("--external-serve-vllm-arg", action="append", default=[])
    external.add_argument("--external-s3-output-root", default=None)

    return parser


def _apply_preset(args: argparse.Namespace) -> None:
    if not args.preset:
        return

    presets = _load_presets()
    preset = presets[args.preset]

    datasets = preset.get("datasets") or []
    if datasets and not args.dataset and not args.dataset_path:
        args.dataset_path = datasets[0]
        print(f"[launch] preset {args.preset}: dataset_path={datasets[0]}")

    if "n_concurrent" in preset and not _cli_has("--n_concurrent", "--n-concurrent"):
        args.n_concurrent = preset["n_concurrent"]

    if "n_attempts" in preset and not _cli_has("--n_attempts", "--n-attempts"):
        args.n_attempts = preset["n_attempts"]

    agent_parser = preset.get("agent_parser")
    if agent_parser:
        existing = {kw.split("=", 1)[0] for kw in args.agent_kwarg or []}
        if "parser" not in existing:
            args.agent_kwarg = args.agent_kwarg or []
            args.agent_kwarg.append(f"parser={agent_parser}")

    for kw in preset.get("agent_kwargs") or []:
        key = kw.split("=", 1)[0]
        existing = {k.split("=", 1)[0] for k in args.agent_kwarg or []}
        if key not in existing:
            args.agent_kwarg = args.agent_kwarg or []
            args.agent_kwarg.append(kw)


def _normalize(args: argparse.Namespace) -> None:
    if args.external_profile == "grug":
        args.external_serve_model = args.external_serve_model or DEFAULT_GRUG_SERVE_MODEL
        args.gpu = args.gpu or "H100x8"
        args.target_cluster = args.target_cluster or "cw-us-east-02a"
        args.external_s3_output_root = args.external_s3_output_root or DEFAULT_GRUG_S3_OUTPUT_ROOT
        if not _cli_has("--n_concurrent", "--n-concurrent"):
            args.n_concurrent = 256
        if not args.model:
            args.model = f"vllm/{args.external_serve_model}"
        args.external_serve_vllm_arg.extend(
            ["--enable-expert-parallel", "--enable-auto-tool-choice", "--tool-call-parser=hermes"]
        )

    if args.external_endpoint and args.external_serve_model:
        raise ValueError("Specify --external-endpoint or --external-serve-model, not both.")
    if (args.external_endpoint or args.external_serve_model) and not args.target_cluster:
        raise ValueError("A separately served endpoint requires --target-cluster.")
    _apply_preset(args)

    if args.dataset and args.dataset_path:
        raise ValueError("Specify either --dataset or --dataset_path (not both).")
    if not args.dataset and not args.dataset_path:
        raise ValueError("Must provide --dataset or --dataset_path.")

    if args.harbor_env is None:
        args.harbor_env = get_harbor_env_from_config(args.harbor_config)

    if not args.agent:
        harbor_cfg = load_harbor_config(args.harbor_config)
        agents = harbor_cfg.get("agents", [])
        if agents and isinstance(agents[0], dict):
            args.agent = agents[0].get("name")
            if args.agent:
                print(f"[launch] Inferred --agent={args.agent} from harbor config")

    if not args.model and args.datagen_config:
        cfg = load_harbor_config(args.datagen_config)
        engine = cfg.get("engine") or {}
        if isinstance(engine, dict):
            args.model = engine.get("model_path")

    if not args.model:
        raise ValueError("Must provide --model or --datagen_config (to infer model).")

    # Resolve re-fire filter
    raw_refire = getattr(args, "refire_filter_error_types", None)
    if raw_refire is None:
        args.refire_filter_error_types = list(DEFAULT_REFIRE_ERROR_TYPES)
    elif any(str(v).strip().lower() in ("", "none", "off") for v in raw_refire):
        args.refire_filter_error_types = []
    else:
        args.refire_filter_error_types = list(raw_refire)

    if args.disk is None:
        args.disk = DEFAULT_GPU_DISK if args.gpu else DEFAULT_TPU_DISK

    if args.external_serve_model:
        if not args.gpu:
            raise ValueError("--external-serve-model requires --gpu (or --external-profile grug).")
        args.external_serve_name = args.external_serve_name or f"{args.job_name or 'eval'}-serve"
        args.external_endpoint_name = args.external_endpoint_name or f"/serve/{args.external_serve_name}"
        defaults = {
            "external_serve_max_model_len": 32768,
            "external_serve_max_num_batched_tokens": 8192,
            "external_serve_tensor_parallel_size": 1,
            "external_serve_data_parallel_size": 1,
            "external_serve_max_num_seqs": 32,
        }
        if args.external_profile == "grug":
            defaults.update(
                external_serve_max_model_len=65536,
                external_serve_max_num_batched_tokens=7168,
                external_serve_data_parallel_size=8,
            )
        for key, value in defaults.items():
            if getattr(args, key) is None:
                setattr(args, key, value)
    elif args.external_endpoint:
        args.external_endpoint_name = args.external_endpoint

    if args.external_s3_output_root:
        args.harbor_extra_arg.append(
            f"--jobs-dir={durable_harbor_jobs_dir(args.external_s3_output_root, args.job_name or 'eval')}"
        )

def build_worker_command(args: argparse.Namespace) -> List[str]:
    """Build the in-pod run_eval.py command."""
    cmd: List[str] = [
        "python",
        "-m",
        "experiments.agentic_evals.run_eval",
        "--harbor_config",
        args.harbor_config,
        "--model",
        args.model,
    ]

    if args.dataset:
        cmd.extend(["--dataset", args.dataset])
    elif args.dataset_path:
        cmd.extend(["--dataset_path", args.dataset_path])

    cmd.extend(
        [
            "--agent",
            args.agent,
            "--n_concurrent",
            str(args.n_concurrent),
            "--n_attempts",
            str(args.n_attempts),
        ]
    )

    if args.harbor_env:
        cmd.extend(["--harbor_env", args.harbor_env])
    if args.job_name:
        cmd.extend(["--job_name", args.job_name])
    if args.dry_run:
        cmd.append("--dry_run")

    for kwarg in args.agent_kwarg or []:
        cmd.extend(["--agent_kwarg", kwarg])
    for extra_arg in args.harbor_extra_arg or []:
        cmd.append(f"--harbor_extra_arg={extra_arg}")

    if args.external_endpoint_name:
        cmd.extend(["--external-agent-api-base-env", "EXTERNAL_AGENT_API_BASE"])

    if args.upload_to_database:
        cmd.append("--upload_to_database")
    if args.upload_hf_repo:
        cmd.extend(["--upload_hf_repo", args.upload_hf_repo])
    if args.upload_hf_private:
        cmd.append("--upload_hf_private")
    if args.upload_hf_episodes:
        cmd.extend(["--upload_hf_episodes", args.upload_hf_episodes])
    if args.upload_username:
        cmd.extend(["--upload_username", args.upload_username])
    if args.upload_error_mode:
        cmd.extend(["--upload_error_mode", args.upload_error_mode])
    if args.upload_forced_update:
        cmd.append("--upload_forced_update")

    for _et in getattr(args, "refire_filter_error_types", None) or []:
        cmd.extend(["--refire_filter_error_type", _et])

    return cmd


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    if not args.job_name:
        args.job_name = f"eval-{int(__import__('time').time())}"
    _normalize(args)
    _prepare_worker_paths(args)

    accelerator = args.gpu or args.tpu

    # Build env vars for the task container
    env_vars: Dict[str, str] = {}
    env_vars["_iris_extras"] = ["datagen"] if args.gpu else ["datagen-tpu"]
    if args.harbor_env:
        env_vars["HARBOR_ENV"] = args.harbor_env

    if args.external_serve_model:
        import subprocess

        result = subprocess.run(build_marin_serve_command(args), cwd=Path.cwd(), check=False)
        if result.returncode:
            raise SystemExit(result.returncode)
    if args.external_endpoint_name:
        env_vars["EXTERNAL_AGENT_API_BASE"] = mint_external_api_base(args)
        env_vars["EXTERNAL_AGENT_API_KEY"] = DUMMY_API_KEY

    # Resolve model config for agent kwargs passthrough
    resolved = resolve_model_config(args.model, subsystem="eval")
    if resolved:
        for k in ("tool_call_parser", "reasoning_parser"):
            v = resolved.get(k)
            if v:
                existing = {kw.split("=", 1)[0] for kw in args.agent_kwarg or []}
                if k not in existing:
                    args.agent_kwarg = args.agent_kwarg or []
                    args.agent_kwarg.append(f"{k}={v}")

    command = build_worker_command(args)

    from .backends.iris import IrisBackend

    backend = IrisBackend(
        workspace=Path.cwd(),
        cluster_config=args.cluster_config,
        task_image=args.task_image,
    )

    exit_code = backend.submit(
        command=command,
        job_name=args.job_name,
        env_vars=env_vars,
        accelerator=accelerator,
        replicas=args.replicas,
        cpu=args.cpu,
        memory=args.memory,
        disk=args.disk,
        task_image=args.task_image,
        priority=args.priority,
        max_retries=args.max_retries,
        timeout=args.timeout,
        secrets_env=args.secrets_env,
        dry_run=args.dry_run,
        no_wait=args.no_wait,
        target_cluster=args.target_cluster,
    )
    sys.exit(exit_code or 0)


if __name__ == "__main__":
    main()
