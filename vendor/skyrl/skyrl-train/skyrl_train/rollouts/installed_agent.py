"""Run Harbor installed agents against engine-owned completions and machines."""

import asyncio
import json
import math
import shlex
from collections.abc import Awaitable, Callable
from typing import Any

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.agents.installed.pi import Pi
from harbor.models.agent.context import AgentContext
from rolloutengine.models import ModelTurn, Prompt

from skyrl_train.rollouts.agent_backend import AgentInferenceBackend
from skyrl_train.rollouts.agent_environment import BorrowedMachineEnvironment
from skyrl_train.rollouts.agent_trace import recorded_completion


_SANDBOX_DIAGNOSTIC_SCHEMA: dict[tuple[str, str], dict[str, type | tuple[type, ...]]] = {
    ("budget", "registered"): {"context_window": int},
    ("budget", "request"): {
        "attempt": int,
        "context_window": int,
        "configured_output_tokens": int,
        "message_count": int,
        "tool_count": int,
    },
    ("budget", "tokenize_failed"): {"http_status": int},
    ("budget", "tokenize_invalid"): {},
    ("budget", "tokenize_completed"): {"prompt_tokens": int},
    ("budget", "decision"): {
        "attempt": int,
        "original_prompt_tokens": int,
        "fitted_prompt_tokens": int,
        "prompt_budget": int,
        "available_output_tokens": int,
        "output_tokens": int,
        "tool_outputs_fitted": bool,
    },
    ("recorder", "upstream_status"): {
        "http_status": int,
        "max_completion_tokens": (int, type(None)),
        "streaming": bool,
    },
    ("recorder", "literal_evidence"): {
        "http_status": int,
        "prompt_token_count": (int, type(None)),
        "completion_token_count": (int, type(None)),
        "logprob_count": (int, type(None)),
        "has_assistant": bool,
        "tool_call_count": int,
    },
    ("recorder", "capture_complete"): {"turn_count": int},
    ("recorder", "capture_failed"): {
        "turn_count": int,
        "missing_prompt_ids": bool,
        "missing_completion_ids": bool,
        "missing_logprobs": bool,
        "missing_assistant": bool,
    },
}


def retain_pi_sandbox_diagnostics(source: Any, target: Any) -> None:
    """Publish only the bounded, credential-free sandbox diagnostic schema."""
    if source.stat().st_size > 1024 * 1024:
        raise ValueError("Pi sandbox diagnostic file exceeds the safe upload limit")
    lines = source.read_text().splitlines()
    if len(lines) > 256:
        raise ValueError("Pi sandbox diagnostic file exceeds the safe event limit")
    validated = []
    for line in lines:
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError("Pi sandbox diagnostic event is not an object")
        if type(record.get("source")) is not str or type(record.get("phase")) is not str:
            raise ValueError("Pi sandbox diagnostic event has invalid labels")
        schema = _SANDBOX_DIAGNOSTIC_SCHEMA.get((record["source"], record["phase"]))
        if schema is None or set(record) != {"source", "phase", *schema}:
            raise ValueError("Pi sandbox diagnostic event has unapproved fields")
        for field, expected in schema.items():
            value = record[field]
            allowed_types = expected if isinstance(expected, tuple) else (expected,)
            valid = any(type(value) is allowed for allowed in allowed_types)
            if not valid or (type(value) is int and not 0 <= value <= 10_000_000):
                raise ValueError("Pi sandbox diagnostic event has an invalid scalar")
        validated.append(json.dumps(record))
    target.write_text("\n".join(validated) + ("\n" if validated else ""))


async def run_installed_agent(
    agent: BaseInstalledAgent,
    environment: BorrowedMachineEnvironment,
    backend: AgentInferenceBackend,
    complete: Callable[[Prompt], Awaitable[ModelTurn]],
    *,
    instruction: str,
    setup_timeout: float,
    run_timeout: float,
    cleanup_timeout: float,
) -> dict[str, Any] | None:
    """Register before setup and retain logs before the engine closes the machine.

    Exact requests and turns are retained separately from the training projection,
    which can drop earlier turns when an agent compacts its conversation. The
    caller owns uploading the logs directory to the run's durable artifact store.
    """
    trial_id = agent.rollout_correlation_id
    if trial_id is None:
        raise ValueError("Installed-agent execution requires rollout capture to be enabled")
    agent.logs_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir = environment.trial_paths.trial_dir / "engine"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    context = AgentContext()
    # Container-owned logs must not overwrite model evidence during download.
    trace_path = evidence_dir / "turns.jsonl"

    async def observed_complete(request: Prompt) -> ModelTurn:
        environment.record_diagnostic("model_request:start")
        try:
            turn = await complete(request)
        except BaseException as error:
            environment.record_diagnostic("model_request:failed", error=error)
            raise
        environment.record_diagnostic("model_request:completed")
        return turn

    with backend.session(trial_id, recorded_completion(observed_complete, trace_path)):
        environment.record_diagnostic("agent:prepare_logs")
        prepared = await environment.exec(
            f"mkdir -p {shlex.quote(str(agent.environment_logs_dir))}",
            user="root",
            timeout_sec=math.ceil(setup_timeout),
        )
        if prepared.return_code != 0:
            raise RuntimeError(f"Cannot create installed-agent logs directory: {prepared.stderr}")
        primary_error: BaseException | None = None
        try:
            environment.record_diagnostic("agent:setup_start")
            async with asyncio.timeout(setup_timeout):
                await agent.setup(environment)
            environment.record_diagnostic("agent:setup_complete")
            environment.record_diagnostic("agent:run_start")
            with environment.with_agent_timeout(run_timeout):
                async with asyncio.timeout(run_timeout):
                    await agent.run(instruction=instruction, environment=environment, context=context)
            environment.record_diagnostic("agent:run_complete")
        except BaseException as error:
            primary_error = error
            environment.record_diagnostic("agent:failed", error=error)
            raise
        finally:
            environment.record_diagnostic("agent:download_start")
            try:
                async with asyncio.timeout(cleanup_timeout):
                    await environment.download_dir(str(agent.environment_logs_dir), agent.logs_dir)
            except BaseException as cleanup_error:
                environment.record_diagnostic("agent:download_failed", error=cleanup_error)
                if primary_error is not None:
                    primary_error.add_note(f"Agent log download also failed: {type(cleanup_error).__name__}")
                    raise primary_error from cleanup_error
                raise
            environment.record_diagnostic("agent:download_complete")
            if isinstance(agent, Pi) and agent.verbose_sandbox_diagnostics:
                sandbox_diagnostics = agent.logs_dir / "pi-sandbox-diagnostics.jsonl"
                if sandbox_diagnostics.is_file():
                    retain_pi_sandbox_diagnostics(sandbox_diagnostics, evidence_dir / sandbox_diagnostics.name)
            agent.populate_context_post_run(context)
            (evidence_dir / "agent-context.json").write_text(context.model_dump_json(indent=2))
            (evidence_dir / "agent-info.json").write_text(agent.to_agent_info().model_dump_json(indent=2))
    # Pi owns this trial-local capture. The persisted turn trace is diagnostic,
    # not an alternative source from which to reconstruct missing evidence.
    return (context.metadata or {}).get("pi_hydrated_trajectory")
