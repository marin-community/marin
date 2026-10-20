"""Run Harbor installed agents against engine-owned completions and machines."""

import asyncio
import math
import shlex
from collections.abc import Awaitable, Callable

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.models.agent.context import AgentContext
from rolloutengine.models import ModelTurn, Prompt

from skyrl_train.rollouts.agent_backend import AgentInferenceBackend
from skyrl_train.rollouts.agent_environment import BorrowedMachineEnvironment
from skyrl_train.rollouts.agent_trace import recorded_completion


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
) -> None:
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
    with backend.session(trial_id, recorded_completion(complete, trace_path)):
        prepared = await environment.exec(
            f"mkdir -p {shlex.quote(str(agent.environment_logs_dir))}",
            user="root",
            timeout_sec=math.ceil(setup_timeout),
        )
        if prepared.return_code != 0:
            raise RuntimeError(f"Cannot create installed-agent logs directory: {prepared.stderr}")
        try:
            async with asyncio.timeout(setup_timeout):
                await agent.setup(environment)
            with environment.with_agent_timeout(run_timeout):
                async with asyncio.timeout(run_timeout):
                    await agent.run(instruction=instruction, environment=environment, context=context)
        finally:
            async with asyncio.timeout(cleanup_timeout):
                await environment.download_dir(str(agent.environment_logs_dir), agent.logs_dir)
            agent.populate_context_post_run(context)
            (evidence_dir / "agent-context.json").write_text(context.model_dump_json(indent=2))
            (evidence_dir / "agent-info.json").write_text(agent.to_agent_info().model_dump_json(indent=2))
