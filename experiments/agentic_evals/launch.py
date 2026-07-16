#!/usr/bin/env python3
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
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from .runtime.args import (
    add_harbor_args,
    add_harbor_env_arg,
    add_model_compute_args,
    add_hf_upload_args,
    add_database_upload_args,
)
from .harness.config import load_harbor_config, get_harbor_env_from_config
from .results.infra_errors import INFRA_ERROR_TYPES
from .serve.model_config import resolve_model_config


DEFAULT_REFIRE_ERROR_TYPES = sorted(
    set(INFRA_ERROR_TYPES) | {"DaytonaValidationError", "VerifierTimeoutError"}
)


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


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch an agentic eval job on a cluster (Iris TPU/GPU)."
    )

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
    iris.add_argument("--cluster-config", "--cluster_config", default=None,
                      help="Path to the iris cluster YAML.")
    iris.add_argument("--task-image", "--task_image", default=None,
                      help="Container image for the task.")
    iris.add_argument("--tpu", default="v6e-4", help="TPU variant.")
    iris.add_argument("--gpu", default=None, help="GPU variant (e.g. H100x8).")
    iris.add_argument("--replicas", type=int, default=1)
    iris.add_argument("--cpu", type=float, default=8.0)
    iris.add_argument("--memory", default="256GB")
    iris.add_argument("--disk", default="100GB")
    iris.add_argument("--priority", default="interactive",
                      choices=["production", "interactive", "batch"])
    iris.add_argument("--max-retries", "--max_retries", type=int, default=0)
    iris.add_argument("--timeout", type=int, default=0)
    iris.add_argument("--no-wait", "--no_wait", dest="no_wait", action="store_true", default=False)
    iris.add_argument("--secrets-env", "--secrets_env", default=None,
                      help="Path to a KEY=VALUE env file.")

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
        existing = {kw.split("=", 1)[0] for kw in (args.agent_kwarg or [])}
        if "parser" not in existing:
            args.agent_kwarg = args.agent_kwarg or []
            args.agent_kwarg.append(f"parser={agent_parser}")

    for kw in preset.get("agent_kwargs") or []:
        key = kw.split("=", 1)[0]
        existing = {k.split("=", 1)[0] for k in (args.agent_kwarg or [])}
        if key not in existing:
            args.agent_kwarg = args.agent_kwarg or []
            args.agent_kwarg.append(kw)


def _normalize(args: argparse.Namespace) -> None:
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

    accelerator = args.gpu or args.tpu


def build_worker_command(args: argparse.Namespace) -> List[str]:
    """Build the in-pod run_eval.py command."""
    cmd: List[str] = [
        "python", "-m", "agentic_evals.run_eval",
        "--harbor_config", args.harbor_config,
        "--model", args.model,
    ]

    if args.dataset:
        cmd.extend(["--dataset", args.dataset])
    elif args.dataset_path:
        cmd.extend(["--dataset_path", args.dataset_path])

    cmd.extend([
        "--agent", args.agent,
        "--n_concurrent", str(args.n_concurrent),
        "--n_attempts", str(args.n_attempts),
    ])

    if args.harbor_env:
        cmd.extend(["--harbor_env", args.harbor_env])
    if args.job_name:
        cmd.extend(["--job_name", args.job_name])
    if args.dry_run:
        cmd.append("--dry_run")

    for kwarg in args.agent_kwarg or []:
        cmd.extend(["--agent_kwarg", kwarg])

    if args.upload_to_database:
        cmd.append("--upload_to_database")
    if args.upload_hf_repo:
        cmd.extend(["--upload_hf_repo", args.upload_hf_repo])
    if args.upload_hf_private:
        cmd.append("--upload_hf_private")
    if args.upload_hf_episodes:
        cmd.extend(["--upload_hf_episodes", args.upload_hf_episodes])

    for _et in getattr(args, "refire_filter_error_types", None) or []:
        cmd.extend(["--refire_filter_error_type", _et])

    return cmd


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    _normalize(args)

    accelerator = args.gpu or args.tpu

    # Build env vars for the task container
    env_vars: Dict[str, str] = {}
    if args.harbor_env:
        env_vars["HARBOR_ENV"] = args.harbor_env

    # Resolve model config for agent kwargs passthrough
    resolved = resolve_model_config(args.model, subsystem="eval")
    if resolved:
        for k in ("tool_call_parser", "reasoning_parser"):
            v = resolved.get(k)
            if v:
                existing = {kw.split("=", 1)[0] for kw in (args.agent_kwarg or [])}
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
        job_name=args.job_name or f"eval-{int(__import__('time').time())}",
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
    )
    sys.exit(exit_code or 0)


if __name__ == "__main__":
    main()
