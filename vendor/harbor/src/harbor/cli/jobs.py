import json
import logging
import signal
from datetime import datetime
from pathlib import Path
from typing import Annotated

import yaml
from dotenv import dotenv_values, load_dotenv
from upath import UPath
from rich.console import Console
from rich.table import Table
from typer import Argument, Option, Typer

from harbor.agents.factory import AgentFactory
from harbor.cli.notifications import show_registry_hint_if_first_run
from harbor.cli.utils import (
    load_mcp_servers,
    parse_env_vars,
    parse_kwargs,
    parse_tpu_spec,
    run_async,
)
from harbor.models.agent.name import AgentName
from harbor.models.bridge import BridgeConfig, BridgeKind
from harbor.models.environment_type import EnvironmentType
from harbor.models.job.config import (
    DatasetConfig,
    JobConfig,
)
from harbor.models.job.result import JobStats
from harbor.models.task.task import Task
from harbor.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    ResourceMode,
    TaskConfig,
    UserAgentConfig,
)
from harbor.models.trial.paths import TrialPaths
from harbor.models.trial.result import TrialResult
from harbor.trial.simulated_user import validate_user_agent_version_pin
from harbor.utils.artifact_authority import validate_artifact_authority

jobs_app = Typer(
    no_args_is_help=True, context_settings={"help_option_names": ["-h", "--help"]}
)
console = Console()
logger = logging.getLogger(__name__)


def _validate_bridge_target_agents(agents: list[AgentConfig], kind: BridgeKind) -> None:
    """Fail before task startup for built-in targets without bridge support."""
    for agent in agents:
        if agent.name is None or agent.name not in AgentName.values():
            continue
        agent_cls = AgentFactory.get_agent_class(AgentName(agent.name))
        if kind not in agent_cls.SUPPORTED_BRIDGES:
            raise ValueError(
                f"Agent '{agent.name}' does not support the '{kind}' bridge."
            )


def _confirm_host_env_access(
    job,
    console: Console,
    *,
    explicit_env_file_keys: set[str] | None = None,
    skip_confirm: bool = False,
) -> None:
    import os
    import tomllib

    from pydantic import ValidationError

    from harbor.models.task.config import TaskConfig as TaskDefinitionConfig
    from harbor.models.task.paths import TaskPaths
    from harbor.utils.env import get_required_host_vars

    is_oracle = any(a.name == AgentName.ORACLE.value for a in job.config.agents)
    explicit_env_file_keys = explicit_env_file_keys or set()
    explicit_job_environment_keys = set(job.config.environment.env)
    explicit_job_verifier_keys = set(job.config.verifier.env)
    sections: dict[str, list[tuple[str, str | None]]] = {}

    for task_config in job._task_configs:
        try:
            local_path = task_config.get_local_path()
        except ValueError:
            continue
        if not local_path.exists():
            continue
        try:
            task_config_model = TaskDefinitionConfig.model_validate_toml(
                TaskPaths(local_path).config_path.read_text()
            )
        except (OSError, tomllib.TOMLDecodeError, ValidationError):
            continue

        env_sections = [
            ("environment", task_config_model.environment.env),
            ("verifier", task_config_model.verifier.env),
        ]
        if is_oracle:
            env_sections.append(("solution", task_config_model.solution.env))

        for section_name, env_dict in env_sections:
            filtered_env_dict = env_dict
            if section_name == "environment" and explicit_job_environment_keys:
                filtered_env_dict = {
                    key: value
                    for key, value in env_dict.items()
                    if key not in explicit_job_environment_keys
                }
            elif section_name == "verifier" and explicit_job_verifier_keys:
                filtered_env_dict = {
                    key: value
                    for key, value in env_dict.items()
                    if key not in explicit_job_verifier_keys
                }

            required = [
                item
                for item in get_required_host_vars(filtered_env_dict)
                if item[0] not in explicit_env_file_keys
            ]
            if required:
                key = f"[{section_name}.env]"
                existing = sections.get(key, [])
                for item in required:
                    if item not in existing:
                        existing.append(item)
                if existing:
                    sections[key] = existing

    if not sections:
        return

    missing = []
    for section, vars_list in sections.items():
        for var_name, default in vars_list:
            if default is None and var_name not in os.environ:
                missing.append((section, var_name))

    if missing:
        table = Table(
            title="Missing Environment Variables",
            title_style="bold red",
            show_header=True,
            header_style="bold",
            padding=(0, 2),
            show_edge=False,
            show_lines=False,
        )
        table.add_column("Variable", style="cyan")
        table.add_column("Phase", style="dim")

        for section, var_name in missing:
            escaped = section.replace("[", "\\[")
            table.add_row(var_name, escaped)

        console.print()
        console.print(table)
        console.print(
            "\n[yellow]Export them in your shell or pass --env-file.[/yellow]"
        )
        raise SystemExit(1)

    if skip_confirm:
        return

    table = Table(
        title="Environment Variables",
        title_style="bold",
        show_header=True,
        header_style="bold",
        padding=(0, 2),
        show_edge=False,
        show_lines=False,
    )
    table.add_column("Variable", style="cyan")
    table.add_column("Phase", style="dim")

    for section, vars_list in sections.items():
        escaped = section.replace("[", "\\[")
        for var_name, default in vars_list:
            table.add_row(var_name, escaped)

    console.print()
    console.print(table)
    console.print()

    response = console.input(
        "Tasks in this run will load these from your environment. [yellow]Proceed? (Y/n):[/yellow] "
    )
    if response.strip().lower() in ("n", "no"):
        raise SystemExit(0)


def _format_duration(started_at: datetime | None, finished_at: datetime | None) -> str:
    if started_at is None or finished_at is None:
        return "unknown"

    total_seconds = max(0, int((finished_at - started_at).total_seconds()))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _format_group_title(evals_key: str, job_result) -> str:
    parts = evals_key.split("__")
    if len(parts) == 3:
        agent_name, model_name, dataset_name = parts
    elif len(parts) == 2:
        agent_name, dataset_name = parts
        model_name = None
    else:
        # A model name can itself contain "__" (e.g.
        # laion/exp-psu-swesmith-1K_glm_4-7_traces_jupiter__0-85__Qwen3-8B), so
        # evals_key = format_agent_evals_key(agent, model, source) =
        # f"{agent}__{model}__{source}" splits into >3 (or, if no model, <2)
        # parts. The bare `agent_name, dataset_name = parts` unpack then raises
        # ValueError -> `harbor jobs` exits 1 -> the eval sbatch skips its upload
        # -> the leg NEVER registers. This silently blocked every
        # exp-psu-swesmith__X__ campaign leg. Reconstruct robustly: agent = first
        # segment, source(dataset) = last segment, model = everything between.
        agent_name = parts[0]
        dataset_name = parts[-1]
        model_name = "__".join(parts[1:-1]) or None

    for trial_result in job_result.trial_results:
        trial_evals_key = JobStats.evals_key_for(trial_result)
        if trial_evals_key != evals_key:
            continue

        agent_name = trial_result.agent_info.name
        model_name = trial_result.config.agent.model_name_without_provider or model_name
        dataset_name = trial_result.source or dataset_name
        break

    title_parts = [dataset_name, agent_name]
    if model_name:
        title_parts.append(model_name)
    return " • ".join(title_parts)


def print_job_results_tables(job_result) -> None:
    for evals_key, dataset_stats in job_result.stats.evals.items():
        console.print(f"[bold]{_format_group_title(evals_key, job_result)}[/bold]")

        summary_table = Table(show_header=True)
        summary_table.add_column("Trials", justify="right")
        summary_table.add_column("Exceptions", justify="right")

        summary_row = [str(dataset_stats.n_trials), str(dataset_stats.n_errors)]
        if dataset_stats.metrics:
            for i, metric in enumerate(dataset_stats.metrics):
                for key, value in metric.items():
                    metric_label = (
                        f"Metric {i + 1}: {key}"
                        if len(dataset_stats.metrics) > 1
                        else key
                    )
                    summary_table.add_column(metric_label.title(), justify="right")
                    if isinstance(value, float):
                        summary_row.append(f"{value:.3f}")
                    else:
                        summary_row.append(str(value))

        if dataset_stats.pass_at_k:
            for k, value in sorted(dataset_stats.pass_at_k.items()):
                summary_table.add_column(f"Pass@{k}", justify="right")
                summary_row.append(f"{value:.3f}")

        summary_table.add_row(*summary_row)
        console.print(summary_table)

        if dataset_stats.reward_stats:
            reward_table = Table(show_header=True)
            reward_table.add_column("Reward")
            reward_table.add_column("Count", justify="right")
            for reward_key, reward_values in dataset_stats.reward_stats.items():
                for reward_value, trial_names in reward_values.items():
                    count = len(trial_names)
                    reward_table.add_row(str(reward_value), str(count))
            console.print()
            console.print(reward_table)

        if dataset_stats.exception_stats:
            exception_table = Table(show_header=True)
            exception_table.add_column("Exception")
            exception_table.add_column("Count", justify="right")
            for (
                exception_type,
                trial_names,
            ) in dataset_stats.exception_stats.items():
                count = len(trial_names)
                exception_table.add_row(exception_type, str(count))
            console.print()
            console.print(exception_table)

        console.print()


def _handle_sigterm(signum, frame):
    raise KeyboardInterrupt


def start(
    config_path: Annotated[
        Path | None,
        Option(
            "-c",
            "--config",
            help="A job configuration path in yaml or json format. "
            "Should implement the schema of harbor.models.job.config:JobConfig. "
            "Allows for more granular control over the job configuration.",
            rich_help_panel="Config",
            show_default=False,
        ),
    ] = None,
    job_name: Annotated[
        str | None,
        Option(
            "--job-name",
            help="Name of the job (default: timestamp)",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
    jobs_dir: Annotated[
        # str (not Path) so URIs like ``gs://bucket/prefix`` survive
        # CLI parsing without typer/click collapsing ``//`` to ``/``.
        # JobConfig.jobs_dir's field_validator coerces str -> UPath, which
        # handles both local paths and fsspec-backed remote schemes.
        str | None,
        Option(
            "-o",
            "--jobs-dir",
            help=f"Directory to store job results (default: {
                JobConfig.model_fields['jobs_dir'].default
            })",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
    n_attempts: Annotated[
        int | None,
        Option(
            "-k",
            "--n-attempts",
            help=f"Number of attempts per trial (default: {
                JobConfig.model_fields['n_attempts'].default
            })",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
    timeout_multiplier: Annotated[
        float | None,
        Option(
            "--timeout-multiplier",
            help=f"Multiplier for task timeouts (default: {
                JobConfig.model_fields['timeout_multiplier'].default
            })",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
    agent_timeout_multiplier: Annotated[
        float | None,
        Option(
            "--agent-timeout-multiplier",
            help="Multiplier for agent execution timeout (overrides --timeout-multiplier)",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
    verifier_timeout_multiplier: Annotated[
        float | None,
        Option(
            "--verifier-timeout-multiplier",
            help="Multiplier for verifier timeout (overrides --timeout-multiplier)",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
    agent_setup_timeout_multiplier: Annotated[
        float | None,
        Option(
            "--agent-setup-timeout-multiplier",
            help="Multiplier for agent setup timeout (overrides --timeout-multiplier)",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
    environment_build_timeout_multiplier: Annotated[
        float | None,
        Option(
            "--environment-build-timeout-multiplier",
            help="Multiplier for environment build timeout (overrides --timeout-multiplier)",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
    trial_attempt_timeout_sec: Annotated[
        float | None,
        Option(
            "--trial-attempt-timeout",
            help="Whole admitted trial-attempt timeout in seconds",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
    trial_cleanup_grace_sec: Annotated[
        float | None,
        Option(
            "--trial-cleanup-grace",
            help="Cleanup grace after a trial-attempt timeout in seconds",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
    quiet: Annotated[
        bool,
        Option(
            "-q",
            "--quiet",
            "--silent",
            help="Suppress individual trial progress displays",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = False,
    debug: Annotated[
        bool,
        Option(
            "--debug",
            help="Enable debug logging",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = False,
    n_concurrent_trials: Annotated[
        int | None,
        Option(
            "-n",
            "--n-concurrent",
            help=f"Number of concurrent trials to run (default: {
                JobConfig.model_fields['n_concurrent_trials'].default
            })",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
    max_retries: Annotated[
        int | None,
        Option(
            "-r",
            "--max-retries",
            help="Maximum number of retry attempts (default: 0)",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
    retry_include_exceptions: Annotated[
        list[str] | None,
        Option(
            "--retry-include",
            help="Exception types to retry on. If not specified, all exceptions except "
            "those in --retry-exclude are retried (can be used multiple times)",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
    retry_exclude_exceptions: Annotated[
        list[str] | None,
        Option(
            "--retry-exclude",
            help="Exception types to NOT retry on (can be used multiple times)",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
    agent_name: Annotated[
        str | None,
        Option(
            "-a",
            "--agent",
            help=(
                "Agent name or ACP registry shorthand "
                "(for example: acp:opencode@1.3.9)."
            ),
            rich_help_panel="Agent",
            show_default=False,
        ),
    ] = None,
    agent_import_path: Annotated[
        str | None,
        Option(
            "--agent-import-path",
            help="Import path for custom agent",
            rich_help_panel="Agent",
            show_default=False,
        ),
    ] = None,
    model_names: Annotated[
        list[str] | None,
        Option(
            "-m",
            "--model",
            help="Model name for the agent (can be used multiple times)",
            rich_help_panel="Agent",
            show_default=True,
        ),
    ] = None,
    agent_kwargs: Annotated[
        list[str] | None,
        Option(
            "--ak",
            "--agent-kwarg",
            help="Additional agent kwarg in the format 'key=value'. You can view "
            "available kwargs by looking at the agent's `__init__` method. "
            "Can be set multiple times to set multiple kwargs. Common kwargs "
            "include: version, prompt_template, etc.",
            rich_help_panel="Agent",
            show_default=False,
        ),
    ] = None,
    agent_env: Annotated[
        list[str] | None,
        Option(
            "--ae",
            "--agent-env",
            help="Environment variable to pass to the agent in KEY=VALUE format. "
            "Can be used multiple times. Example: --ae AWS_REGION=us-east-1",
            rich_help_panel="Agent",
            show_default=False,
        ),
    ] = None,
    mcp_config: Annotated[
        list[Path] | None,
        Option(
            "--mcp-config",
            help="Path to a Claude-style .mcp.json or Harbor MCP config file. Can be used multiple times.",
            rich_help_panel="Agent",
            show_default=False,
        ),
    ] = None,
    skills: Annotated[
        list[Path] | None,
        Option(
            "--skill",
            "--skills",
            help="Path to a skill directory, or a root containing skill directories. "
            "Can be used multiple times.",
            rich_help_panel="Agent",
            show_default=False,
        ),
    ] = None,
    user_agent_name: Annotated[
        str | None,
        Option(
            "--user-agent",
            help="Agent that plays the simulated user.",
            rich_help_panel="Simulated User",
            show_default=False,
        ),
    ] = None,
    user_agent_import_path: Annotated[
        str | None,
        Option(
            "--user-agent-import-path",
            help="Import path for a custom simulated-user agent.",
            rich_help_panel="Simulated User",
            show_default=False,
        ),
    ] = None,
    user_model: Annotated[
        str | None,
        Option(
            "--user-model",
            help="Model name for the simulated-user agent.",
            rich_help_panel="Simulated User",
            show_default=False,
        ),
    ] = None,
    user_agent_kwargs: Annotated[
        list[str] | None,
        Option(
            "--uk",
            "--user-agent-kwarg",
            help="Simulated-user agent kwarg in key=value format.",
            rich_help_panel="Simulated User",
            show_default=False,
        ),
    ] = None,
    user_agent_env: Annotated[
        list[str] | None,
        Option(
            "--ue",
            "--user-agent-env",
            help="Environment variable for the simulated-user agent in KEY=VALUE format.",
            rich_help_panel="Simulated User",
            show_default=False,
        ),
    ] = None,
    user_mcp_config: Annotated[
        list[Path] | None,
        Option(
            "--user-mcp-config",
            help="MCP config for the simulated-user agent. Can be repeated.",
            rich_help_panel="Simulated User",
            show_default=False,
        ),
    ] = None,
    user_persona_path: Annotated[
        Path | None,
        Option(
            "--user-persona-path",
            help="Plain-text persona for the simulated user.",
            rich_help_panel="Simulated User",
            show_default=False,
        ),
    ] = None,
    user_prompt_template_path: Annotated[
        Path | None,
        Option(
            "--user-prompt-template-path",
            help="Jinja2 prompt template for the simulated user.",
            rich_help_panel="Simulated User",
            show_default=False,
        ),
    ] = None,
    bridge_kind: Annotated[
        BridgeKind | None,
        Option(
            "--bridge",
            help="Protocol bridge connecting the simulated user to the target.",
            rich_help_panel="Simulated User",
            show_default=False,
        ),
    ] = None,
    bridge_prompt_path: Annotated[
        Path | None,
        Option(
            "--bridge-prompt-path",
            help="Instructions teaching the user agent how to use the bridge.",
            rich_help_panel="Simulated User",
            show_default=False,
        ),
    ] = None,
    bridge_kwargs: Annotated[
        list[str] | None,
        Option(
            "--bk",
            "--bridge-kwarg",
            help="Bridge kwarg in key=value format.",
            rich_help_panel="Simulated User",
            show_default=False,
        ),
    ] = None,
    environment_type: Annotated[
        EnvironmentType | None,
        Option(
            "-e",
            "--env",
            help=f"Environment type (default: {EnvironmentType.DOCKER.value})",
            rich_help_panel="Environment",
            show_default=False,
        ),
    ] = None,
    environment_import_path: Annotated[
        str | None,
        Option(
            "--environment-import-path",
            help="Import path for custom environment (module.path:ClassName).",
            rich_help_panel="Environment",
            show_default=False,
        ),
    ] = None,
    environment_force_build: Annotated[
        bool | None,
        Option(
            "--force-build/--no-force-build",
            help=f"Whether to force rebuild the environment (default: {
                '--force-build'
                if EnvironmentConfig.model_fields['force_build'].default
                else '--no-force-build'
            })",
            rich_help_panel="Environment",
            show_default=False,
        ),
    ] = None,
    environment_delete: Annotated[
        bool | None,
        Option(
            "--delete/--no-delete",
            help=f"Whether to delete the environment after completion (default: {
                '--delete'
                if EnvironmentConfig.model_fields['delete'].default
                else '--no-delete'
            })",
            rich_help_panel="Environment",
            show_default=False,
        ),
    ] = None,
    cpus: Annotated[
        ResourceMode | None,
        Option(
            "--cpus",
            help="How to apply task CPU resources: auto, limit, request, guarantee, or ignore.",
            rich_help_panel="Environment",
            show_default=False,
        ),
    ] = None,
    memory: Annotated[
        ResourceMode | None,
        Option(
            "--memory",
            help="How to apply task memory resources: auto, limit, request, guarantee, or ignore.",
            rich_help_panel="Environment",
            show_default=False,
        ),
    ] = None,
    override_cpus: Annotated[
        int | None,
        Option(
            "--override-cpus",
            help="Override the number of CPUs for the environment",
            rich_help_panel="Environment",
            show_default=False,
        ),
    ] = None,
    override_memory_mb: Annotated[
        int | None,
        Option(
            "--override-memory-mb",
            help="Override the memory (in MB) for the environment",
            rich_help_panel="Environment",
            show_default=False,
        ),
    ] = None,
    override_storage_mb: Annotated[
        int | None,
        Option(
            "--override-storage-mb",
            help="Override the storage (in MB) for the environment",
            rich_help_panel="Environment",
            show_default=False,
        ),
    ] = None,
    override_gpus: Annotated[
        int | None,
        Option(
            "--override-gpus",
            help="Override the number of GPUs for the environment",
            rich_help_panel="Environment",
            show_default=False,
        ),
    ] = None,
    override_tpu: Annotated[
        str | None,
        Option(
            "--override-tpu",
            help=(
                "Override the TPU spec for the environment in TYPE=TOPOLOGY "
                "format (e.g. 'v6e=2x4'). The task allocates one TPU slice "
                "per pod, so only a single spec is accepted."
            ),
            rich_help_panel="Environment",
            show_default=False,
        ),
    ] = None,
    mounts: Annotated[
        str | None,
        Option(
            "--mounts",
            "--mounts-json",
            help="JSON array of volume mounts for the environment container "
            "(Docker Compose service volume format). --mounts-json is a "
            "deprecated alias.",
            rich_help_panel="Environment",
            show_default=False,
        ),
    ] = None,
    extra_docker_compose: Annotated[
        list[Path] | None,
        Option(
            "--extra-docker-compose",
            help="Additional Docker Compose overlay file. Can be used multiple times.",
            rich_help_panel="Environment",
            show_default=False,
        ),
    ] = None,
    environment_kwargs: Annotated[
        list[str] | None,
        Option(
            "--ek",
            "--environment-kwarg",
            help="Environment kwarg in key=value format (can be used multiple times)",
            rich_help_panel="Environment",
            show_default=False,
        ),
    ] = None,
    yes: Annotated[
        bool,
        Option(
            "-y",
            "--yes",
            help=(
                "Auto-confirm prompts, including host environment access and "
                "sharing with organizations you are not a member of."
            ),
            rich_help_panel="Job Settings",
        ),
    ] = False,
    env_file: Annotated[
        Path | None,
        Option(
            "--env-file",
            help="Path to a .env file to load into environment.",
            rich_help_panel="Job Settings",
        ),
    ] = None,
    path: Annotated[
        Path | None,
        Option(
            "-p",
            "--path",
            help="Path to a local task or dataset directory",
            rich_help_panel="Dataset",
            show_default=False,
        ),
    ] = None,
    extra_instruction_paths: Annotated[
        list[Path] | None,
        Option(
            "--extra-instruction-path",
            help="Path to an extra instruction file to append to the task "
            "instruction. Can be used multiple times.",
            rich_help_panel="Dataset",
            show_default=False,
        ),
    ] = None,
    task_git_url: Annotated[
        str | None,
        Option(
            help="Git URL for a task repository",
            rich_help_panel="Dataset",
            show_default=False,
        ),
    ] = None,
    task_git_commit_id: Annotated[
        str | None,
        Option(
            "--task-git-commit",
            help="Git commit ID for the task (requires --task-git-url)",
            rich_help_panel="Dataset",
            show_default=False,
        ),
    ] = None,
    dataset_name_version: Annotated[
        str | None,
        Option(
            "-d",
            "--dataset",
            help="Dataset name@version (e.g., 'dataset@1.0')",
            rich_help_panel="Dataset",
            show_default=False,
        ),
    ] = None,
    registry_url: Annotated[
        str | None,
        Option(
            "--registry-url",
            help="Registry URL for remote dataset",
            rich_help_panel="Dataset",
            show_default="The default harbor registry.",
        ),
    ] = None,
    registry_path: Annotated[
        Path | None,
        Option(
            "--registry-path",
            help="Path to local registry for dataset",
            rich_help_panel="Dataset",
            show_default=False,
        ),
    ] = None,
    task_ref: Annotated[
        str | None,
        Option(
            "-t",
            "--task",
            help="Run a single task from the registry (org/name[@ref])",
            rich_help_panel="Dataset",
            show_default=False,
        ),
    ] = None,
    dataset_task_names: Annotated[
        list[str] | None,
        Option(
            "-i",
            "--include-task-name",
            help="Task name to include from dataset (supports glob patterns)",
            rich_help_panel="Dataset",
            show_default=False,
        ),
    ] = None,
    dataset_exclude_task_names: Annotated[
        list[str] | None,
        Option(
            "-x",
            "--exclude-task-name",
            help="Task name to exclude from dataset (supports glob patterns)",
            rich_help_panel="Dataset",
            show_default=False,
        ),
    ] = None,
    n_tasks: Annotated[
        int | None,
        Option(
            "-l",
            "--n-tasks",
            help="Maximum number of tasks to run (applied after other filters)",
            rich_help_panel="Dataset",
            show_default=False,
        ),
    ] = None,
    export_traces: Annotated[
        bool,
        Option(
            "--export-traces/--no-export-traces",
            help="After job completes, export traces from the job directory",
            hidden=True,
        ),
    ] = False,
    export_sharegpt: Annotated[
        bool,
        Option(
            "--export-sharegpt/--no-export-sharegpt",
            help="Also emit ShareGPT column when exporting traces",
            hidden=True,
        ),
    ] = False,
    export_episodes: Annotated[
        str,
        Option(
            "--export-episodes",
            help="Which episodes to export per trial (all|last)",
            hidden=True,
        ),
    ] = "all",
    export_push: Annotated[
        bool,
        Option(
            "--export-push/--no-export-push",
            help="Push exported dataset to HF Hub",
            hidden=True,
        ),
    ] = False,
    export_repo: Annotated[
        str | None,
        Option(
            "--export-repo",
            help="Target HF repo id (org/name) when pushing traces",
            hidden=True,
        ),
    ] = None,
    export_instruction_metadata: Annotated[
        bool,
        Option(
            "--export-instruction-metadata/--no-export-instruction-metadata",
            help="Include instruction text column when exporting traces",
            hidden=True,
        ),
    ] = False,
    export_verifier_metadata: Annotated[
        bool,
        Option(
            "--export-verifier-metadata/--no-export-verifier-metadata",
            help="Include verifier stdout/stderr column when exporting traces",
            hidden=True,
        ),
    ] = False,
    artifact_paths: Annotated[
        list[str] | None,
        Option(
            "--artifact",
            help="Environment path to download as an artifact after the trial "
            "(can be used multiple times)",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
    verifier_env: Annotated[
        list[str] | None,
        Option(
            "--ve",
            "--verifier-env",
            help="Environment variable to pass to the verifier in KEY=VALUE format. "
            "Can be used multiple times. Example: --ve OPENAI_BASE_URL=http://localhost:8000/v1",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
    verifier_import_path: Annotated[
        str | None,
        Option(
            "--verifier-import-path",
            help="Import path for custom verifier (module.path:ClassName).",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
    verifier_kwargs: Annotated[
        list[str] | None,
        Option(
            "--verifier-kwarg",
            help="Additional verifier kwarg in the format 'key=value'.",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
    disable_verification: Annotated[
        bool,
        Option(
            "--disable-verification/--enable-verification",
            help="Disable task verification (skip running tests)",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = False,
):
    from harbor.job import Job

    if env_file is not None:
        if not env_file.exists():
            console.print(f"[red]❌ Env file not found: {env_file}[/red]")
            raise SystemExit(1)
        load_dotenv(env_file, override=True)

    base_config = None
    if config_path is not None:
        if config_path.suffix == ".yaml":
            base_config = JobConfig.model_validate(
                yaml.safe_load(config_path.read_text())
            )
        elif config_path.suffix == ".json":
            base_config = JobConfig.model_validate_json(config_path.read_text())
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")

    config = base_config if base_config is not None else JobConfig()

    if job_name is not None:
        config.job_name = job_name
    if jobs_dir is not None:
        # Pydantic field_validators don't run on attribute assignment by
        # default (validate_assignment=False), so a raw str slips past
        # the str->UPath coercion. Coerce explicitly so downstream
        # ``self.config.jobs_dir / self.config.job_name`` works.
        config.jobs_dir = UPath(jobs_dir)
    if n_attempts is not None:
        config.n_attempts = n_attempts
    if timeout_multiplier is not None:
        config.timeout_multiplier = timeout_multiplier
    if agent_timeout_multiplier is not None:
        config.agent_timeout_multiplier = agent_timeout_multiplier
    if verifier_timeout_multiplier is not None:
        config.verifier_timeout_multiplier = verifier_timeout_multiplier
    if agent_setup_timeout_multiplier is not None:
        config.agent_setup_timeout_multiplier = agent_setup_timeout_multiplier
    if environment_build_timeout_multiplier is not None:
        config.environment_build_timeout_multiplier = (
            environment_build_timeout_multiplier
        )
    if trial_attempt_timeout_sec is not None:
        config.trial_attempt_timeout_sec = trial_attempt_timeout_sec
    if trial_cleanup_grace_sec is not None:
        config.trial_cleanup_grace_sec = trial_cleanup_grace_sec
    if debug:
        config.debug = debug

    if n_concurrent_trials is not None:
        config.n_concurrent_trials = n_concurrent_trials
    if quiet:
        config.quiet = quiet
    if max_retries is not None:
        config.retry.max_retries = max_retries
    if retry_include_exceptions is not None:
        config.retry.include_exceptions = set(retry_include_exceptions)
    if retry_exclude_exceptions is not None:
        config.retry.exclude_exceptions = set(retry_exclude_exceptions)

    if agent_name is not None or agent_import_path is not None:
        config.agents = []
        parsed_kwargs = parse_kwargs(agent_kwargs)
        parsed_env = parse_env_vars(agent_env)
        parsed_mcp_servers = [
            server
            for mcp_config_path in mcp_config or []
            for server in load_mcp_servers(mcp_config_path)
        ]

        if model_names is not None:
            config.agents = [
                AgentConfig(
                    name=agent_name,
                    import_path=agent_import_path,
                    model_name=model_name,
                    skills=skills or [],
                    kwargs=parsed_kwargs,
                    env=parsed_env,
                    mcp_servers=parsed_mcp_servers,
                )
                for model_name in model_names
            ]
        else:
            config.agents = [
                AgentConfig(
                    name=agent_name,
                    import_path=agent_import_path,
                    skills=skills or [],
                    kwargs=parsed_kwargs,
                    env=parsed_env,
                    mcp_servers=parsed_mcp_servers,
                )
            ]
    else:
        parsed_kwargs = parse_kwargs(agent_kwargs)
        parsed_env = parse_env_vars(agent_env)
        parsed_mcp_servers = [
            server
            for mcp_config_path in mcp_config or []
            for server in load_mcp_servers(mcp_config_path)
        ]
        if parsed_kwargs or parsed_env or parsed_mcp_servers or skills:
            for agent in config.agents:
                if parsed_kwargs:
                    agent.kwargs.update(parsed_kwargs)
                if parsed_env:
                    agent.env.update(parsed_env)
                if parsed_mcp_servers:
                    agent.mcp_servers.extend(parsed_mcp_servers)
                if skills:
                    agent.skills.extend(skills)

    if user_agent_name is not None and user_agent_import_path is not None:
        console.print(
            "[red]Error:[/red] Use either --user-agent or "
            "--user-agent-import-path, not both."
        )
        raise SystemExit(1)

    user_agent_requested = (
        user_agent_name is not None or user_agent_import_path is not None
    )
    if user_agent_requested:
        existing_bridge = (
            config.user_agent.bridge.model_copy(deep=True)
            if config.user_agent is not None
            else None
        )
        resolved_bridge_kind = bridge_kind or (
            existing_bridge.kind if existing_bridge is not None else None
        )
        if resolved_bridge_kind is None:
            console.print("[red]Error:[/red] --user-agent requires --bridge.")
            raise SystemExit(1)
        bridge = existing_bridge or BridgeConfig(kind=resolved_bridge_kind)
        bridge.kind = resolved_bridge_kind
        if bridge_prompt_path is not None:
            bridge.prompt_path = bridge_prompt_path
        if bridge_kwargs is not None:
            bridge.kwargs.update(parse_kwargs(bridge_kwargs))
        user_mcp_servers = [
            server
            for config_path in user_mcp_config or []
            for server in load_mcp_servers(config_path)
        ]
        config.user_agent = UserAgentConfig(
            name=user_agent_name,
            import_path=user_agent_import_path,
            model_name=user_model,
            kwargs=parse_kwargs(user_agent_kwargs),
            env=parse_env_vars(user_agent_env),
            mcp_servers=user_mcp_servers,
            user_persona_path=user_persona_path,
            user_prompt_template_path=user_prompt_template_path,
            bridge=bridge,
        )
    elif config.user_agent is not None:
        if user_model is not None:
            config.user_agent.model_name = user_model
        if user_agent_kwargs is not None:
            config.user_agent.kwargs.update(parse_kwargs(user_agent_kwargs))
        if user_agent_env is not None:
            config.user_agent.env.update(parse_env_vars(user_agent_env))
        if user_mcp_config is not None:
            config.user_agent.mcp_servers.extend(
                server
                for config_path in user_mcp_config
                for server in load_mcp_servers(config_path)
            )
        if user_persona_path is not None:
            config.user_agent.user_persona_path = user_persona_path
        if user_prompt_template_path is not None:
            config.user_agent.user_prompt_template_path = user_prompt_template_path
        if bridge_kind is not None:
            config.user_agent.bridge.kind = bridge_kind
        if bridge_prompt_path is not None:
            config.user_agent.bridge.prompt_path = bridge_prompt_path
        if bridge_kwargs is not None:
            config.user_agent.bridge.kwargs.update(parse_kwargs(bridge_kwargs))

    if config.user_agent is None and (
        user_model is not None
        or user_agent_kwargs
        or user_agent_env
        or user_mcp_config
        or user_persona_path is not None
        or user_prompt_template_path is not None
        or bridge_kind is not None
        or bridge_prompt_path is not None
        or bridge_kwargs
    ):
        console.print(
            "[red]Error:[/red] simulated-user and bridge flags require "
            "--user-agent, --user-agent-import-path, or user_agent in the job config."
        )
        raise SystemExit(1)

    if config.user_agent is not None:
        try:
            _validate_bridge_target_agents(config.agents, config.user_agent.bridge.kind)
            for agent in config.agents:
                validate_user_agent_version_pin(
                    agent.name,
                    agent.kwargs.get("version"),
                    config.user_agent.name,
                    config.user_agent.kwargs.get("version"),
                )
        except ValueError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise SystemExit(1) from exc

    if environment_type is not None:
        config.environment.type = environment_type
    if environment_import_path is not None:
        config.environment.import_path = environment_import_path
        config.environment.type = None  # Clear type so import_path takes precedence
    if environment_force_build is not None:
        config.environment.force_build = environment_force_build
    if environment_delete is not None:
        config.environment.delete = environment_delete
    if cpus is not None:
        config.environment.cpu_enforcement_policy = cpus
    if memory is not None:
        config.environment.memory_enforcement_policy = memory
    if override_cpus is not None:
        config.environment.override_cpus = override_cpus
    if override_memory_mb is not None:
        config.environment.override_memory_mb = override_memory_mb
    if override_storage_mb is not None:
        config.environment.override_storage_mb = override_storage_mb
    if override_gpus is not None:
        config.environment.override_gpus = override_gpus
    if override_tpu is not None:
        config.environment.override_tpu = parse_tpu_spec(override_tpu)
    if mounts is not None:
        config.environment.mounts = json.loads(mounts)
    if extra_docker_compose is not None:
        config.environment.extra_docker_compose.extend(extra_docker_compose)
    if environment_kwargs is not None:
        config.environment.kwargs.update(parse_kwargs(environment_kwargs))

    if verifier_env is not None:
        config.verifier.env.update(parse_env_vars(verifier_env))
    if verifier_import_path is not None:
        config.verifier.import_path = verifier_import_path
    if verifier_kwargs is not None:
        config.verifier.kwargs.update(parse_kwargs(verifier_kwargs))
    if disable_verification:
        config.verifier.disable = disable_verification

    if artifact_paths is not None:
        config.artifacts = list(artifact_paths)
    if extra_instruction_paths is not None:
        config.extra_instruction_paths = list(extra_instruction_paths)

    task_specified = task_git_url is not None or task_git_commit_id is not None

    dataset_specified = (
        dataset_name_version is not None
        or registry_url is not None
        or registry_path is not None
        or dataset_task_names is not None
        or dataset_exclude_task_names is not None
        or n_tasks is not None
        or task_ref is not None
    )

    if task_specified and dataset_specified:
        raise ValueError("Cannot specify both task and dataset parameters")

    if path is not None:
        is_task = Task.is_valid_dir(
            path,
            disable_verification=config.verifier.disable
            or config.verifier.import_path is not None,
        )

        if is_task:
            config.tasks = [
                TaskConfig(
                    path=path,
                    git_url=task_git_url,
                    git_commit_id=task_git_commit_id,
                )
            ]
            config.datasets = []
        else:
            config.tasks = []
            config.datasets = [
                DatasetConfig(
                    path=path,
                    task_names=dataset_task_names,
                    exclude_task_names=dataset_exclude_task_names,
                    n_tasks=n_tasks,
                )
            ]

    elif task_specified:
        raise ValueError("Task configuration with --task-git-url requires --path")

    elif dataset_specified:
        config.tasks = []

        if task_ref is not None:
            from harbor.models.package.reference import PackageReference

            ref = PackageReference.parse(task_ref)
            config.datasets = []
            config.tasks = [
                TaskConfig(
                    name=ref.name,
                    ref=ref.ref,
                )
            ]

        elif dataset_name_version is not None:
            if "@" in dataset_name_version:
                name, version = dataset_name_version.split("@", 1)
            else:
                name, version = dataset_name_version, None

            # Detect package reference (org/name format)
            if "/" in name:
                config.datasets = [
                    DatasetConfig(
                        name=name,
                        ref=version,
                        task_names=dataset_task_names,
                        exclude_task_names=dataset_exclude_task_names,
                        n_tasks=n_tasks,
                    )
                ]
            else:
                config.datasets = [
                    DatasetConfig(
                        registry_url=registry_url,
                        registry_path=registry_path,
                        name=name,
                        version=version,
                        task_names=dataset_task_names,
                        exclude_task_names=dataset_exclude_task_names,
                        n_tasks=n_tasks,
                    )
                ]

        else:
            raise ValueError(
                "Cannot specify --registry-url, --registry-path, --include-task-name, or "
                "--exclude-task-name without also specifying --dataset, --task, or --path."
            )

    async def _run_job():
        job = await Job.create(config)
        _confirm_host_env_access(
            job,
            console,
            explicit_env_file_keys=explicit_env_file_keys,
            skip_confirm=yes,
        )
        job_result = await job.run()
        console.print()
        print_job_results_tables(job_result)
        console.print("[bold]Job Info[/bold]")
        console.print(
            f"Total runtime: {_format_duration(job_result.started_at, job_result.finished_at)}"
        )
        console.print(f"Results written to {job._job_result_path}")
        console.print()
        return job, job_result

    from harbor.environments.factory import EnvironmentFactory

    EnvironmentFactory.run_preflight(
        type=config.environment.type,
        import_path=config.environment.import_path,
    )

    explicit_env_file_keys: set[str] = set()
    if env_file is not None:
        explicit_env_file_keys = {
            key for key in dotenv_values(env_file).keys() if key is not None
        }

    signal.signal(signal.SIGTERM, _handle_sigterm)

    show_registry_hint_if_first_run(console)

    job, job_result = run_async(_run_job())

    if export_traces:
        from harbor.utils.traces_utils import export_traces as _export_traces

        job_dir = job.job_dir
        print(f"[traces] Exporting traces from job dir: {job_dir}")
        try:
            if export_push and not export_repo:
                raise ValueError("--export-push requires --export-repo <org/name>")
            ds = _export_traces(
                root=job_dir,
                recursive=True,
                episodes="last" if export_episodes == "last" else "all",
                to_sharegpt=export_sharegpt,
                repo_id=export_repo,
                push=export_push,
                include_instruction=export_instruction_metadata,
                include_verifier_output=export_verifier_metadata,
            )
            print(f"[traces] Exported {len(ds)} rows from {job_dir}")
        except Exception as e:
            print(f"[traces] Export failed: {e}")


def _prune_unscored_trials(job_dir: Path) -> None:
    """Delete every UNSCORED trial dir under ``job_dir`` so a warm resume RE-RUNS it.

    A trial is UNSCORED iff ``verifier_result is None`` — the verifier never produced a
    result (e.g. an infra failure such as a Daytona 401 left the trial with no reward).
    This is the re-fire prune keyed on the gate's OWN definition of valid ("verifier
    scored it"), NOT on an exception-type allow-list.

    It replaces the old ``exception_type in filter_error_types`` predicate, which MISSED
    unscored trials mislabeled with a benign leaf type not in the list (e.g. an
    ``AgentTimeoutError`` whose post-timeout verifier then hit a Daytona 401 ->
    ``AddTestsDirError`` -> ``verifier_result=None``, but first-wins kept the benign
    ``AgentTimeoutError`` type). Those trials were neither counted toward completeness
    nor re-run, ceiling-ing eval legs ~72-81%.

    A SCORED trial (``verifier_result is not None``, INCLUDING a legitimate reward 0) is
    ALWAYS kept — never delete a reward-bearing trial even if it also recorded a filtered
    infra leaf. This closes the clobber vector where an infra-typed-but-scored trial would
    be wrongly re-run and could lose its reward. Net: a warm re-fire re-runs exactly the
    trials that do not yet count toward valid-complete, so completeness can only rise
    across resumes.

    A trial with a MISSING ``result.json`` is left as-is (``Job.create`` re-runs
    genuinely-missing trials); a truncated/corrupt ``result.json`` is treated as
    incomplete and removed so it re-runs.
    """
    from harbor.utils.path_compat import safe_rmtree

    for trial_dir in job_dir.iterdir():
        if not trial_dir.is_dir():
            continue

        trial_paths = TrialPaths(UPath(trial_dir))
        if not trial_paths.result_path.exists():
            continue

        try:
            trial_result = TrialResult.model_validate_json(
                trial_paths.result_path.read_text()
            )
        except Exception as e:
            # A truncated/empty/corrupt result.json (e.g. a 0-byte file left by a
            # wall-clock kill mid-write) is an INCOMPLETE trial, not a completed one.
            # Remove its dir so resume re-runs it, instead of aborting the ENTIRE resume
            # on this one parse error (observed on wall-terminated re-resumes).
            console.print(
                f"Removing trial with unreadable result ({type(e).__name__}); "
                f"will re-run: {trial_dir.name}"
            )
            safe_rmtree(trial_dir)
            continue

        if trial_result.verifier_result is None:
            console.print(
                f"Removing unscored trial (verifier_result=None); "
                f"will re-run: {trial_dir.name}"
            )
            safe_rmtree(trial_dir)


@jobs_app.command()
def resume(
    job_path: Annotated[
        Path,
        Option(
            "-p",
            "--job-path",
            help="Path to the job directory containing the config.json file",
        ),
    ],
    filter_error_types: Annotated[
        list[str] | None,
        Option(
            "-f",
            "--filter-error-type",
            help="Remove trials with these error types before resuming (can be used "
            "multiple times)",
            show_default=False,
        ),
    ] = ["CancelledError"],
):
    """Resume an existing job from its job directory."""
    from harbor.job import Job

    job_dir = Path(job_path)
    config_path = job_dir / "config.json"

    if not job_dir.exists():
        raise ValueError(f"Job directory does not exist: {job_dir}")

    if not config_path.exists():
        raise ValueError(f"Config file not found: {config_path}")

    config = JobConfig.model_validate_json(config_path.read_text())
    validate_artifact_authority(
        job_dir,
        config.artifact_authority,
        operation="job resume",
    )

    if filter_error_types:
        # Re-fire mode (--filter-error-type was passed): prune every UNSCORED trial so
        # the warm resume RE-RUNS it; scored (reward-bearing) trials are always kept.
        # The specific error-type list is intentionally no longer consulted — the gate's
        # own "valid = verifier scored it" definition is the correct, complete predicate
        # (it catches unscored trials mislabeled with a benign leaf type the list misses,
        # and it never clobbers a scored trial). See _prune_unscored_trials.
        _prune_unscored_trials(job_dir)

    from harbor.environments.factory import EnvironmentFactory

    EnvironmentFactory.run_preflight(
        type=config.environment.type,
        import_path=config.environment.import_path,
    )

    async def _run_job():
        job = await Job.create(config)
        return await job.run()

    job_result = run_async(_run_job())

    # Print results tables
    print_job_results_tables(job_result)


@jobs_app.command()
def summarize(
    job_path: Annotated[Path, Argument(help="Path to job directory.")] = Path("."),
):
    """Summarize trial failures in a job using Claude Agent SDK."""
    console.print(
        "[red]Error: 'harbor jobs summarize' has been removed. "
        "Use 'harbor analyze <job-dir>' instead.[/red]"
    )
    raise SystemExit(1)


jobs_app.command()(start)
