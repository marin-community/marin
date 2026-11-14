# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run physical nop controls through the unchanged production trial factory."""

import hashlib
import json
import math
import re
import shlex
import shutil
import time
from pathlib import Path

from rigging.timing import active_timeout
from taskcompendium.harbor.negative_control import (
    NativeInstructionSetup,
    NegativeControlSetupKind,
    NegativeControlSource,
    public_task_preparation,
)
from taskcompendium.identity import canonical_sha256

from harbor.host_capabilities import HostCapabilities
from harbor.models.task.task import Task
from harbor.models.trial.config import TrialConfig
from harbor.resource_usage import resource_counters, write_resource_usage
from harbor.trial.hooks import TrialEvent, TrialHookEvent
from harbor.trial.taskcompendium import TaskSpecTrial
from harbor.trial.trial import Trial


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _require_nop_configuration(config: TrialConfig, source: NegativeControlSource, task: Task) -> None:
    agent = config.agent
    if agent.name != "nop" or any(
        getattr(agent, name) for name in ("model_name", "model_alias", "kwargs", "env", "skills", "mcp_servers")
    ):
        raise ValueError("A physical negative control requires the unmodified nop agent")
    if (
        config.timeout_multiplier != 1
        or config.agent_timeout_multiplier not in (None, 1)
        or config.verifier_timeout_multiplier not in (None, 1)
        or agent.override_timeout_sec is not None
        or agent.max_timeout_sec is not None
        or config.verifier.override_timeout_sec is not None
        or config.verifier.max_timeout_sec is not None
        or config.verifier.disable
        or config.artifacts
        or config.extra_instruction_paths
    ):
        raise ValueError("A physical negative control cannot override its task or phase budgets")
    environment = config.environment
    if (
        environment.force_build
        or environment.mounts
        or environment.env
        or any(
            getattr(environment, name) is not None
            for name in ("override_cpus", "override_memory_mb", "override_storage_mb", "override_gpus", "override_tpu")
        )
    ):
        raise ValueError("A physical negative control cannot override its source environment")
    shape = source.execution_requirements
    if (
        any(
            getattr(task.config.environment, name) != getattr(shape, name)
            for name in ("cpus", "memory_mb", "storage_mb")
        )
        or task.config.agent.timeout_sec != shape.agent_timeout_sec
        or task.config.verifier.timeout_sec != shape.verifier_timeout_sec
    ):
        raise ValueError("Negative control differs from the source's actual final allocation")


async def run_negative_control(
    config: TrialConfig,
    source: NegativeControlSource,
    output_root: Path,
    native_setup: NativeInstructionSetup | None,
    host_capabilities: HostCapabilities,
) -> dict:
    """Run one nop Trial and write its actual preparation and result artifacts.

    Callers own host admission and archive/upload this fresh directory. Native
    setup selects a verbatim public instruction block; TaskSpec preparation is
    performed entirely by the production adapter before the agent-start event.
    """
    if config.task.path is None:
        raise ValueError("Negative controls require the frozen physical task package")
    packet = Task(Path(config.task.path))
    if packet.checksum != source.task_checksum or packet.has_steps:
        raise ValueError("Negative control requires the exact single-phase consumed package")
    _require_nop_configuration(config, source, packet)
    output_root = output_root.resolve()
    if Path(config.trials_dir).resolve() != output_root / "trials":
        raise ValueError("Negative-control Trial artifacts must belong to its fresh output directory")
    converted = packet.paths.task_spec_path.is_file()
    if converted == (native_setup is not None):
        raise ValueError("Native setup selection must match the actual source workload")
    setup_kind = NegativeControlSetupKind.TASK_SPEC_PREPARATION if converted else NegativeControlSetupKind.NATIVE_SCRIPT
    script = None
    if native_setup is not None:
        blocks = list(re.finditer(r"```bash\r?\n(.*?)\r?\n```", packet.instruction, re.DOTALL))
        if native_setup.block_index >= len(blocks):
            raise ValueError("The selected original public bash setup block does not exist")
        script = blocks[native_setup.block_index].group(1)
    output_root.mkdir(parents=True, exist_ok=False)
    trial = await Trial.create(config, host_capabilities=host_capabilities)
    if isinstance(trial, TaskSpecTrial) != converted:
        raise ValueError("Production trial dispatch differs from the original source workload")
    preparation = None
    if isinstance(trial, TaskSpecTrial):
        specification = trial.specification
        if (
            specification.id != source.task_id
            or canonical_sha256(specification.model_dump(mode="json")) != source.source_task_sha256
            or trial.shape != source.execution_requirements
            or trial.verifier_shape != source.verifier_execution_requirements
            or (trial.source_runtime.runtime_sha256 if trial.source_runtime is not None else None)
            != source.runtime_sha256
        ):
            raise ValueError("TaskSpec control changed the original task, runtime or measured role allocations")
        shutil.copy2(packet.paths.task_spec_path, output_root / "source-task-spec.json")
        preparation = {
            "version": 1,
            "setup_kind": setup_kind.value,
            "task_spec_file": "source-task-spec.json",
            "task_spec_file_sha256": hashlib.sha256(packet.paths.task_spec_path.read_bytes()).hexdigest(),
            "task_spec_sha256": source.source_task_sha256,
            "runtime_sha256": source.runtime_sha256,
            "source_runtime": trial.source_runtime.model_dump(mode="json") if trial.source_runtime is not None else None,
            "source_native_checksum": source.task_checksum,
            "actor_required": trial.network_policy.actor_required,
            "execution_requirements": trial.shape.model_dump(mode="json"),
            "verifier_execution_requirements": trial.verifier_shape.model_dump(mode="json"),
            "public_preparation_sha256": canonical_sha256(public_task_preparation(specification)),
        }
        descriptor = output_root / "task-spec-preparation.json"
        _write_json(descriptor, preparation)
    else:
        assert script is not None
        descriptor = output_root / "original-public-setup.sh"
        descriptor.write_text(script)
    setup_sha256 = hashlib.sha256(descriptor.read_bytes()).hexdigest()
    setup_path = output_root / "setup-receipt.json"
    setup = {
        "event": None,
        "trial_id": config.trial_name,
        "source_native_checksum": source.task_checksum,
        "setup_kind": setup_kind.value,
        "setup_sha256": setup_sha256,
        "setup_exit_code": None,
        "source_agent_timeout_sec": source.execution_requirements.agent_timeout_sec,
        "preparation_completed": False,
        "setup_from_original_public_instruction": setup_kind is NegativeControlSetupKind.NATIVE_SCRIPT,
        "source_commands_outside_trial": 0,
    }
    _write_json(setup_path, setup)
    hook_calls = 0

    async def record_preparation(event: TrialHookEvent) -> None:
        nonlocal hook_calls, setup
        hook_calls += 1
        if hook_calls != 1:
            raise ValueError("A negative control cannot replay public setup within its Trial")
        if isinstance(trial, TaskSpecTrial):
            observed = trial.result.environment_setup
            if observed is None or observed.finished_at is None:
                raise ValueError("TaskSpec public preparation has not completed")
            setup = {
                **setup,
                "event": event.event.value,
                "environment_setup": observed.model_dump(mode="json"),
                "preparation_completed": True,
            }
            _write_json(setup_path, setup)
            return
        assert script is not None
        started = time.monotonic()
        counters = await resource_counters(trial.agent_environment)
        deadline = source.execution_requirements.agent_timeout_sec
        async with active_timeout(deadline):
            with trial.agent_environment.with_agent_timeout(deadline):
                outcome = await trial.agent_environment.exec(
                    "bash -c " + shlex.quote(script), timeout_sec=math.ceil(deadline), user=packet.config.agent.user
                )
        (output_root / "setup-stdout.txt").write_text(outcome.stdout or "")
        (output_root / "setup-stderr.txt").write_text(outcome.stderr or "")
        await write_resource_usage(trial.agent_environment, output_root / "setup-resource-usage.json", counters, started)
        setup = {
            **setup,
            "event": event.event.value,
            "setup_exit_code": outcome.return_code,
            "elapsed_seconds": time.monotonic() - started,
            "preparation_completed": outcome.return_code == 0,
        }
        _write_json(setup_path, setup)
        if outcome.return_code != 0:
            raise RuntimeError("Original public setup failed inside the negative-control Trial")

    trial.add_hook(TrialEvent.AGENT_START, record_preparation)
    await trial.run()
    result_path = Path(trial.paths.result_path)
    result = json.loads(result_path.read_text())
    verdict_path = result_path.parent / "verifier/verdict.json"
    verdict = json.loads(verdict_path.read_text()) if verdict_path.exists() else None
    control = {
        "task_id": source.task_id,
        "source_task_sha256": source.source_task_sha256,
        "source_task_checksum": source.source_task_checksum,
        "task_sha256": source.task_sha256,
        "runtime_sha256": source.runtime_sha256,
        "task_checksum": source.task_checksum,
        "control": "original-public-setup-then-nop",
        "setup_kind": setup_kind.value,
        "native_package_unchanged": Task(Path(config.task.path)).checksum == source.task_checksum,
        "setup": setup,
        "preparation": preparation,
        "hook_calls": hook_calls,
        "physical_verdict": verdict,
        "verifier_exit_code": (result.get("verifier_result") or {}).get("exit_code"),
        "trial_result_path": str(result_path),
        "trial_result_file_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "trial_result": result,
        "source_commands_outside_trial": 0,
    }
    _write_json(output_root / "negative-control.json", control)
    return control
