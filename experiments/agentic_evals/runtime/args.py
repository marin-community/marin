"""Reusable argument groups for agentic-eval launchers.

Copied verbatim from OT-Agent ``hpc/arg_groups.py``. These are pure argparse
helpers with no ``hpc.*`` dependencies, enabling code reuse across local
runners (run_eval.py) and cloud launchers (launch.py).
"""

from __future__ import annotations

import argparse
from argparse import ArgumentParser, _ArgumentGroup
from typing import List, Optional, Union

ArgTarget = Union[ArgumentParser, _ArgumentGroup]


def _add_arg_with_alias(
    parser: ArgTarget,
    primary: str,
    alias: Optional[str] = None,
    **kwargs,
) -> None:
    parser.add_argument(primary, **kwargs)
    if alias:
        dest = primary.lstrip("-").replace("-", "_")
        parser.add_argument(alias, dest=dest, help=argparse.SUPPRESS)


def add_harbor_args(parser: ArgTarget, *, config_required: bool = True) -> None:
    _add_arg_with_alias(
        parser,
        "--harbor_config",
        "--harbor-config",
        required=config_required,
        help="Path to Harbor job config YAML.",
    )
    parser.add_argument(
        "--agent",
        default=None,
        help="Harbor agent name. If not specified, uses the agent from --harbor_config.",
    )
    _add_arg_with_alias(
        parser,
        "--job_name",
        "--job-name",
        help="Optional override for Harbor job name.",
    )
    _add_arg_with_alias(
        parser,
        "--agent_kwarg",
        "--agent-kwarg",
        action="append",
        default=[],
        help="Additional --agent-kwarg entries (key=value).",
    )
    _add_arg_with_alias(
        parser,
        "--harbor_extra_arg",
        "--harbor-extra-arg",
        action="append",
        default=[],
        help="Extra --harbor jobs start args.",
    )


def add_harbor_env_arg(
    parser: ArgTarget,
    *,
    default: Optional[str] = None,
    legacy_names: Optional[List[str]] = None,
) -> None:
    _add_arg_with_alias(
        parser,
        "--harbor_env",
        "--harbor-env",
        default=default,
        choices=["daytona", "docker", "modal", None],
        help="Harbor environment backend: daytona (cloud), docker (local/podman), modal. "
             "If not specified, inferred from Harbor config YAML.",
    )

    if legacy_names:
        for name in legacy_names:
            parser.add_argument(
                name,
                dest="harbor_env",
                help=argparse.SUPPRESS,
            )


def add_model_compute_args(
    parser: ArgTarget,
    *,
    model_required: bool = False,
    default_n_concurrent: int = 16,
    default_n_attempts: int = 1,
    n_attempts_help: str = "Times to run each task (default: 1).",
) -> None:
    parser.add_argument(
        "--model",
        required=model_required,
        help="Model identifier.",
    )
    _add_arg_with_alias(
        parser,
        "--n_concurrent",
        "--n-concurrent",
        type=int,
        default=default_n_concurrent,
        help=f"Concurrent trials (default: {default_n_concurrent}).",
    )
    _add_arg_with_alias(
        parser,
        "--n_attempts",
        "--n-attempts",
        type=int,
        default=default_n_attempts,
        help=n_attempts_help,
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Number of GPUs to use.",
    )
    _add_arg_with_alias(
        parser,
        "--dry_run",
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )


def add_hf_upload_args(parser: ArgTarget) -> None:
    _add_arg_with_alias(
        parser,
        "--upload_hf_repo",
        "--upload-hf-repo",
        help="HuggingFace repo for traces upload.",
    )
    _add_arg_with_alias(
        parser,
        "--upload_hf_token",
        "--upload-hf-token",
        help="HuggingFace token (defaults to $HF_TOKEN).",
    )
    _add_arg_with_alias(
        parser,
        "--upload_hf_private",
        "--upload-hf-private",
        action="store_true",
        help="Create the HuggingFace repo as private.",
    )
    _add_arg_with_alias(
        parser,
        "--upload_hf_episodes",
        "--upload-hf-episodes",
        choices=["last", "all"],
        default="last",
        help="Which episodes to include in traces upload.",
    )


def add_database_upload_args(parser: ArgTarget) -> None:
    _add_arg_with_alias(
        parser,
        "--upload_to_database",
        "--upload-to-database",
        action="store_true",
        help="Upload result abstracts to Supabase and traces to HuggingFace.",
    )
    _add_arg_with_alias(
        parser,
        "--upload_username",
        "--upload-username",
        help="Username for Supabase result attribution (defaults to $UPLOAD_USERNAME or current user).",
    )
    _add_arg_with_alias(
        parser,
        "--upload_error_mode",
        "--upload-error-mode",
        choices=["skip_on_error", "rollback_on_error"],
        default="skip_on_error",
        help="Supabase upload error handling.",
    )
    _add_arg_with_alias(
        parser,
        "--upload_forced_update",
        "--upload-forced-update",
        action="store_true",
        help="Allow overwriting existing Supabase records.",
    )


def add_ray_vllm_args(parser: ArgTarget) -> None:
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host/IP for Ray and vLLM.",
    )
    _add_arg_with_alias(
        parser,
        "--ray_port",
        "--ray-port",
        type=int,
        default=6379,
        help="Ray head node port.",
    )
    _add_arg_with_alias(
        parser,
        "--api_port",
        "--api-port",
        type=int,
        default=8000,
        help="vLLM OpenAI-compatible API port.",
    )
    _add_arg_with_alias(
        parser,
        "--tensor_parallel_size",
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="Tensor parallel size for vLLM.",
    )
    _add_arg_with_alias(
        parser,
        "--pipeline_parallel_size",
        "--pipeline-parallel-size",
        type=int,
        default=None,
        help="Pipeline parallel size for vLLM.",
    )
    _add_arg_with_alias(
        parser,
        "--data_parallel_size",
        "--data-parallel-size",
        type=int,
        default=None,
        help="Data parallel replicas for vLLM.",
    )
    _add_arg_with_alias(
        parser,
        "--health_max_attempts",
        "--health-max-attempts",
        type=int,
        default=100,
        help="Max health check attempts for vLLM.",
    )
    _add_arg_with_alias(
        parser,
        "--health_retry_delay",
        "--health-retry-delay",
        type=int,
        default=30,
        help="Seconds between health checks.",
    )
    _add_arg_with_alias(
        parser,
        "--ray_memory_gb",
        "--ray-memory-gb",
        type=float,
        default=None,
        help="Total memory (GB) for Ray. Auto-detected if not set.",
    )
    _add_arg_with_alias(
        parser,
        "--ray_object_store_gb",
        "--ray-object-store-gb",
        type=float,
        default=40.0,
        help="Ray object store (plasma) size in GB (default: 40).",
    )


def add_log_path_args(parser: ArgTarget) -> None:
    _add_arg_with_alias(
        parser,
        "--harbor_binary",
        "--harbor-binary",
        default="harbor",
        help="Harbor CLI executable path.",
    )
    _add_arg_with_alias(
        parser,
        "--controller_log",
        "--controller-log",
        default=None,
        help="Path for vLLM controller logs.",
    )
    _add_arg_with_alias(
        parser,
        "--ray_log",
        "--ray-log",
        default=None,
        help="Path for Ray logs.",
    )
    _add_arg_with_alias(
        parser,
        "--harbor_log",
        "--harbor-log",
        default=None,
        help="Path for Harbor CLI logs.",
    )


def add_tasks_input_arg(
    parser: ArgTarget,
    *,
    required: bool = True,
) -> None:
    parser.add_argument(
        "--tasks_input_path",
        required=required,
        help="Path to tasks directory (input for trace generation).",
    )
    parser.add_argument(
        "--tasks-input-path",
        dest="tasks_input_path",
        help=argparse.SUPPRESS,
    )


__all__ = [
    "add_harbor_args",
    "add_harbor_env_arg",
    "add_model_compute_args",
    "add_hf_upload_args",
    "add_database_upload_args",
    "add_ray_vllm_args",
    "add_log_path_args",
    "add_tasks_input_arg",
]
