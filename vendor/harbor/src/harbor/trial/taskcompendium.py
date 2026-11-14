# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute original TaskSpecs within Harbor's trial lifecycle and verdict contract."""

import asyncio
import json
import sys
import time
from tempfile import TemporaryDirectory
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import replace
from pathlib import Path

from upath import UPath

from rigging.filesystem.storage_path import StoragePath
from rolloutengine.machines import grading_machine_factory, prepare_machine, task_machine
from rolloutengine.models import MachineRuntimeSpec, RolloutLimits
from shellbox.backends.local.machine import SYSTEM_DIRECTORIES
from shellbox.image import RegistryImage
from shellbox.machine import (
    ExitReason,
    DEFAULT_MACHINE_OUTPUT_LIMIT_BYTES,
    Command,
    HostImage,
    Machine,
    MachineFactory,
    MachineSpec,
    NetworkPolicy,
    ResourceLimit,
    ResourceLimitExceeded,
)
from taskcompendium.grading import parse_grade_result
from taskcompendium.grading_result import GradeResult, GradingFailure, Outcome, verdict_bytes
from taskcompendium.harbor.adapter import (
    TASK_SPEC_FILE,
    actor_machine_required,
    require_trial_budget,
    unsupported_harbor_contract,
)
from taskcompendium.harbor.protected_runtime import (
    RUNTIME_OBSERVATION_TIMEOUT,
    RuntimeRole,
    read_runtime_binding,
)
from taskcompendium.models import (
    ExecutionRequirements,
    TaskSpec,
    VerifyitGrader,
    judge_spec,
)
from taskcompendium.runtime.grading import (
    VERDICT_PATH,
    grade_empty_in_sandbox,
    grade_in_sandbox,
)
from taskcompendium.runtime.output_capture import captured_output_files
from taskcompendium.runtime.oracle import oracle_actor_required, oracle_environment
from verifyit.modes.grade_judge import (
    API_KEY_ENV,
    BASE_URL_ENV,
    MODEL_ENV,
    JudgeConnection,
)

from harbor.agents.taskcompendium import (
    TaskSpecAgent,
    TaskSpecChatAgent,
    TaskSpecEmptyAgent,
    TaskSpecOracleAgent,
    empty_attempt,
)
from harbor.environments.base import SandboxDiskFullError
from harbor.environments.local import local_disk_pool, local_machine_factory
from harbor.environments.resource_policies import validate_local_resources
from harbor.host_capabilities import MODEL_CREDENTIALS, MODEL_RELAY_PORT
from harbor.environments.model_relay import ModelRelay, start_model_relays
from harbor.models.environment_type import EnvironmentType
from harbor.models.trial.result import TimingInfo
from harbor.models.verifier.result import VerifierResult
from harbor.resource_usage import (
    MachineResourceCounters,
    resource_counters,
    write_resource_usage,
)
from harbor.runtime_observation import observe_runtime
from harbor.trial.errors import (
    AgentTimeoutError,
    EnvironmentStartTimeoutError,
    RuntimeServiceError,
    VerifierTimeoutError,
)
from harbor.trial.hooks import TrialEvent
from harbor.trial.trial import Trial
from harbor.utils.artifact_writer import get_artifact_writer
from harbor.verifier.verifier import VerifierRuntimeError, structured_rewards


class _MeasuredMachine:
    """Retain the grader's actual structured verdict as a production trial artifact."""

    def __init__(
        self,
        machine: Machine,
        verdicts: list[bytes],
        usage_path,
        initial: dict[str, int],
    ):
        self.machine, self.verdicts = machine, verdicts
        self.usage_path, self.initial = usage_path, initial
        self.started = time.monotonic()
        self.usage_written = False

    @classmethod
    async def start(cls, machine: Machine, verdicts: list[bytes], usage_path):
        initial = await resource_counters(MachineResourceCounters(machine))
        return cls(machine, verdicts, usage_path, initial)

    async def finish_usage(self) -> None:
        if self.usage_written:
            return
        self.usage_written = True
        await write_resource_usage(
            MachineResourceCounters(self.machine),
            self.usage_path,
            self.initial,
            self.started,
        )

    async def run(self, command: Command):
        try:
            return await self.machine.run(command)
        except ResourceLimitExceeded as error:
            if error.resource == ResourceLimit.DISK:
                raise SandboxDiskFullError(str(error)) from error
            raise

    async def upload(self, source: Path, target: str) -> None:
        await self.machine.upload(source, target)

    async def download(self, source: str, target: Path) -> None:
        await self.machine.download(source, target)
        if source == VERDICT_PATH:
            self.verdicts.append(target.read_bytes())

    async def close(self) -> None:
        await self.finish_usage()
        await self.machine.close()


class _TrialMachineFactory:
    def __init__(
        self,
        factory: MachineFactory,
        trial: "TaskSpecTrial",
        relay: ModelRelay | None,
        stack: AsyncExitStack,
        observation_role: RuntimeRole | None,
        usage_path: UPath | None,
    ):
        self.factory, self.trial, self.relay, self.stack = factory, trial, relay, stack
        self.observation_role = observation_role
        self.usage_path = usage_path

    @property
    def backend(self):
        return self.factory.backend

    def check_environment(self, requirements, spec):
        self.factory.check_environment(requirements, spec)

    async def create(self, spec: MachineSpec) -> Machine:
        machine = await self.factory.create(spec)
        try:
            assert self.trial._service_tasks is not None
            await start_model_relays(
                machine,
                self.stack,
                self.trial._service_tasks,
                judge=self.relay,
                candidate=None,
            )
            if self.observation_role is not None:
                await self.trial._observe_runtime(machine, self.observation_role)
            if self.usage_path is None:
                return machine
            measured = await _MeasuredMachine.start(
                machine,
                self.trial.verdicts,
                self.usage_path,
            )
            self.stack.push_async_callback(measured.finish_usage)
        except BaseException:
            async with asyncio.timeout(self.trial.config.trial_cleanup_grace_sec):
                await machine.close()
            raise
        return measured


async def _persist(path, payload: str | bytes) -> None:
    writer = get_artifact_writer()
    future = await writer.submit_required(path, lambda: payload, description="TaskSpec trial artifact")
    await writer.await_write(future)


class TaskSpecTrial(Trial):
    """Keep TaskSpec execution separate from native Harbor directory lowering."""

    def _init_agent(self) -> None:
        config = self.config.agent
        if config.name not in {"oracle", "nop"}:
            super()._init_agent()
            return
        if config.model_name or config.env or config.skills or config.mcp_servers:
            raise ValueError("TaskSpec controls do not accept candidate settings")
        if config.name == "oracle":
            self.agent = TaskSpecOracleAgent(
                logs_dir=self.paths.agent_dir,
                task_dir=self.task.paths.task_dir,
                protected_values=self.host_capabilities.protected_values,
            )
        else:
            self.agent = TaskSpecEmptyAgent(logs_dir=self.paths.agent_dir)

    def _init_agent_environment(self) -> None:
        self.specification = TaskSpec.model_validate_json((self.task.paths.task_dir / TASK_SPEC_FILE).read_text())
        if isinstance(self.agent, TaskSpecOracleAgent) and self.agent.command is not None:
            self.agent.command.require_task(self.specification)
        self.runtime_binding = read_runtime_binding(self.task.paths.task_dir)
        if self.runtime_binding is not None:
            self.runtime_binding.require_task(self.specification)
        unsupported = unsupported_harbor_contract(self.specification)
        if unsupported is not None:
            raise ValueError(unsupported)
        if not isinstance(self.agent, TaskSpecAgent):
            raise ValueError("Use a harbor.agents.taskcompendium agent for a TaskSpec payload")
        self._owned = AsyncExitStack()
        self.machine: _MeasuredMachine | None = None
        self.verdicts: list[bytes] = []
        self.attempt = empty_attempt(self.specification)
        self.shape = ExecutionRequirements(
            cpus=self.task.config.environment.cpus,
            memory_mb=self.task.config.environment.memory_mb,
            storage_mb=self.task.config.environment.storage_mb,
            agent_timeout_sec=self.task.config.agent.timeout_sec,
            verifier_timeout_sec=self.task.config.verifier.timeout_sec,
        )
        require_trial_budget(self.specification, self.shape)
        if self.config.environment.type != EnvironmentType.LOCAL:
            raise ValueError("TaskSpec trials currently require the LOCAL backend")
        configuration = self.config.environment
        validate_local_resources(
            self.task.config.environment.model_copy(update={"storage_mb": None}),
            cpu_enforcement_policy=configuration.cpu_enforcement_policy,
            memory_enforcement_policy=configuration.memory_enforcement_policy,
        )
        if any(
            value is not None
            for value in (
                configuration.override_cpus,
                configuration.override_memory_mb,
                configuration.override_storage_mb,
                configuration.override_gpus,
                configuration.override_tpu,
            )
        ):
            raise ValueError("TaskSpec trials do not accept resource overrides")
        if configuration.force_build or configuration.mounts or configuration.env:
            raise ValueError("TaskSpec trials do not accept environment or file overrides")
        if (self.config.agent.env.keys() | self.config.verifier.env.keys()) - set(MODEL_CREDENTIALS):
            raise ValueError("TaskSpec transport accepts only host credential declarations")
        if (
            self.config.artifacts
            or self.config.extra_instruction_paths
            or self.config.agent.skills
            or self.config.agent.mcp_servers
        ):
            raise ValueError("TaskSpec trials do not accept additional instructions, tools or artifacts")
        if (
            self._agent_timeout_sec != self.shape.agent_timeout_sec
            or self._verifier_timeout_sec != self.shape.verifier_timeout_sec
        ):
            raise ValueError("TaskSpec trials do not accept timeout overrides or multipliers")
        self.runtime = MachineRuntimeSpec(
            backend="local",
            network=NetworkPolicy.DENY,
            cpus=self.shape.cpus,
            memory_mb=self.shape.memory_mb,
            storage_mb=self.shape.storage_mb,
            gpus=0,
            user=None,
            startup_timeout=self._environment_build_timeout_sec,
            cleanup_timeout=self.config.trial_cleanup_grace_sec,
        )
        self.limits = RolloutLimits(
            max_turns=None,
            max_total_tokens=None,
            model_turn_timeout=None,
            tool_turn_timeout=None,
            total_turn_timeout=self.shape.agent_timeout_sec,
            attempt_timeout=None,
            verifier_timeout=self.shape.verifier_timeout_sec,
            cleanup_timeout=self.config.trial_cleanup_grace_sec,
        )

    async def _prepare(self) -> None:
        task = self.specification
        self.result.environment_setup = TimingInfo(started_at=self._now())
        # A missing original lock is a source repair gate, before candidate generation or allocation.
        for environment in (task.environment_requirements, task.grader.environment):
            if environment is not None and environment.packages_lock is not None:
                if not await asyncio.to_thread(StoragePath(environment.packages_lock).exists):
                    raise FileNotFoundError(f"Original package lock is unavailable: {environment.packages_lock}")
        kwargs = dict(self.config.environment.kwargs)
        self.judge_endpoint = kwargs.pop("model_endpoint", None)
        supplied_model = kwargs.pop("model_name", None)
        judge = judge_spec(task.grader)
        self.judge_model = None
        if judge is not None:
            if judge.model.strip():
                if supplied_model is not None and supplied_model != judge.model:
                    raise ValueError("Judge model conflicts with the original TaskSpec")
                self.judge_model = judge.model
            else:
                if supplied_model is None or not supplied_model.strip():
                    raise ValueError("TaskSpec judge requires an explicit model_name when its declared model is blank")
                self.judge_model = supplied_model
        disk_root = kwargs.pop("disk_root", None)
        if disk_root is None:
            raise ValueError("Declared TaskSpec disk requires an explicit disk_root")
        kwargs["disk_pool"] = await self._await_phase(
            local_disk_pool(
                self.task.config.environment,
                disk_root=Path(disk_root),
                cpu_enforcement_policy=self.config.environment.cpu_enforcement_policy,
                memory_enforcement_policy=self.config.environment.memory_enforcement_policy,
            ),
            timeout_sec=self._environment_build_timeout_sec,
            timeout_error=EnvironmentStartTimeoutError("TaskSpec disk qualification timed out"),
        )
        self.factory = local_machine_factory(**kwargs)
        if self.factory.cgroup_parent is None:
            raise ValueError("Declared TaskSpec CPU and memory require cgroup enforcement")
        required_actor = actor_machine_required(task)
        if isinstance(self.agent, TaskSpecOracleAgent) and self.agent.command is not None:
            required_actor = oracle_actor_required(task)
        if required_actor:
            factory = self.factory
            if (
                isinstance(self.agent, TaskSpecOracleAgent)
                and self.runtime_binding is not None
                and RuntimeRole.ORACLE_CONTROL in self.runtime_binding.roles
            ):
                factory = _TrialMachineFactory(
                    self.factory, self, None, self._owned,
                    observation_role=RuntimeRole.ORACLE_CONTROL, usage_path=None,
                )
            machine = await self._await_phase(
                self._owned.enter_async_context(
                    task_machine(
                        task,
                        factory,
                        self.runtime,
                        cleanup_timeout=self.config.trial_cleanup_grace_sec,
                    )
                ),
                timeout_sec=self._environment_build_timeout_sec,
                timeout_error=EnvironmentStartTimeoutError("TaskSpec actor setup timed out"),
            )
            self.machine = await _MeasuredMachine.start(
                machine, self.verdicts, self.paths.agent_dir / "resource-usage.json"
            )
            self._owned.push_async_callback(self.machine.finish_usage)
        self.result.environment_setup.finished_at = self._now()

    async def _run(self) -> None:
        task = self.specification
        self.result.agent_execution = TimingInfo(started_at=self._now())
        try:
            async with self._oracle_machine() as machine:
                self.attempt = await self._await_phase(
                    self.agent.run_task(task, machine, self.limits),
                    timeout_sec=self.shape.agent_timeout_sec,
                    timeout_error=AgentTimeoutError("TaskSpec agent deadline expired"),
                )
        except AgentTimeoutError as error:
            self.attempt = self.agent.attempt or self.attempt
            await self._record_exception(error)
        finally:
            self.attempt = self.agent.attempt or self.attempt
            self.result.agent_execution.finished_at = self._now()
            if self.machine is not None:
                await self.machine.finish_usage()
            if isinstance(self.agent, TaskSpecOracleAgent):
                if self.agent.exit_code is not None:
                    await _persist(
                        self.paths.agent_dir / "exit-code.txt",
                        str(self.agent.exit_code),
                    )
                await _persist(self.paths.agent_dir / "oracle.txt", self.agent.output)
                if self.agent.answer is not None:
                    await _persist(self.paths.agent_dir / "answer.bin", self.agent.answer)
        if self.machine is not None and task.output_paths:
            files = await captured_output_files(
                self.machine,
                task.output_paths,
                timeout=self.shape.verifier_timeout_sec,
                limit_bytes=DEFAULT_MACHINE_OUTPUT_LIMIT_BYTES,
            )
            self.attempt = replace(self.attempt, files=files)
        await self._recover_outputs()
        if self.config.verifier.disable:
            return
        self._reached_verifier = True
        await self._emit(TrialEvent.VERIFICATION_START)
        self.result.verifier = TimingInfo(started_at=self._now())
        try:
            grade = await self._grade()
            payload = self.verdicts[-1] if self.verdicts else verdict_bytes(grade)
            await _persist(self.paths.verifier_dir / "verdict.json", payload)
            if grade.failure == GradingFailure.TIMEOUT:
                raise VerifierTimeoutError("TaskSpec verifier command deadline expired")
            if grade.status not in {Outcome.GRADED, Outcome.SUBMISSION_FAILURE}:
                raise VerifierRuntimeError(f"TaskSpec grader did not score the candidate: {grade}")
            rewards = structured_rewards(payload)
            self.result.verifier_result = VerifierResult(rewards=rewards)
        finally:
            self.result.verifier.finished_at = self._now()
            await self._emit(TrialEvent.VERIFICATION_END)

    @asynccontextmanager
    async def _oracle_machine(self):
        """Use the declared grader environment for an actor-less source OracleCommand."""
        task = self.specification
        if self.machine is not None or not isinstance(self.agent, TaskSpecOracleAgent) or self.agent.command is None:
            yield self.machine
            return
        environment = oracle_environment(task, self.agent.command.submission)
        async with AsyncExitStack() as stack:
            machine = await self._await_phase(
                stack.enter_async_context(
                    prepare_machine(
                        environment,
                        task.resources.all + task.resources.worker,
                        _TrialMachineFactory(
                            self.factory, self, None, stack,
                            observation_role=RuntimeRole.ORACLE_CONTROL, usage_path=None,
                        ),
                        self.runtime,
                        cleanup_timeout=self.config.trial_cleanup_grace_sec,
                    )
                ),
                timeout_sec=self._environment_build_timeout_sec,
                timeout_error=EnvironmentStartTimeoutError("TaskSpec oracle environment setup timed out"),
            )
            measured = await _MeasuredMachine.start(machine, self.verdicts, self.paths.agent_dir / "resource-usage.json")
            try:
                yield measured
            finally:
                await measured.finish_usage()

    async def _observe_runtime(self, machine: Machine, role: RuntimeRole) -> None:
        if self.runtime_binding is None:
            return
        expected = self.runtime_binding.roles.get(role)
        if expected is None:
            return
        if role is RuntimeRole.ORACLE_CONTROL:
            directory = self.paths.agent_dir
            deadline = self.shape.agent_timeout_sec
        else:
            directory = self.paths.verifier_dir
            deadline = self.shape.verifier_timeout_sec
        timeout = min(RUNTIME_OBSERVATION_TIMEOUT, deadline)
        try:
            await self._await_phase(
                observe_runtime(machine, expected, role, directory / "protected-runtime.json", timeout),
                timeout_sec=timeout,
                timeout_error=RuntimeServiceError(f"Protected {role.value} runtime observation timed out"),
            )
        except RuntimeServiceError as error:
            await _persist(
                directory / "protected-runtime-failure.json",
                json.dumps({"role": role.value, "status": "infrastructure_pending", "timeout_sec": timeout,
                            "detail": str(error), "stdout": None, "stderr": None}),
            )
            raise

    async def _grade(self) -> GradeResult:
        task, grader = self.specification, self.specification.grader
        judge = judge_spec(grader)
        connection = None
        relay = None
        if judge is not None:
            if not self.judge_endpoint:
                raise ValueError("TaskSpec judge requires model_endpoint with its declared model")
            assert self.judge_model is not None
            relay = ModelRelay.from_endpoint(
                self.judge_endpoint,
                self.judge_model,
                authentication=self.host_capabilities.model_authentication,
            )
            connection = JudgeConnection(
                f"http://127.0.0.1:{MODEL_RELAY_PORT}/v1", "EMPTY", default_model=self.judge_model
            )
        if isinstance(grader, VerifyitGrader) and (grader.environment is None or judge is not None):
            return await self._grade_in_process(relay, connection)
        assert grader.environment is not None
        source = grader.environment.package_spec or HostImage()
        if grader.environment.docker_image is not None:
            source = RegistryImage(grader.environment.docker_image)
        spec = MachineSpec(
            source,
            cpus=self.shape.cpus,
            memory_mb=self.shape.memory_mb,
            storage_mb=self.shape.storage_mb,
            network=NetworkPolicy.DENY,
            startup_timeout=self._environment_build_timeout_sec,
        )
        async with AsyncExitStack() as stack:
            factory = await stack.enter_async_context(
                grading_machine_factory(
                    _TrialMachineFactory(
                        self.factory, self, relay, stack,
                        observation_role=RuntimeRole.SOURCE_GRADER,
                        usage_path=self.paths.verifier_dir / "resource-usage.json",
                    ),
                    self.runtime,
                    cleanup_timeout=self.config.trial_cleanup_grace_sec,
                )
            )
            if isinstance(self.agent, TaskSpecEmptyAgent):
                return await grade_empty_in_sandbox(
                    task,
                    factory,
                    spec,
                    timeout=self.shape.verifier_timeout_sec,
                    judge_connection=connection,
                )
            return await grade_in_sandbox(
                task,
                self.attempt,
                factory,
                spec,
                task_machine=self.machine,
                timeout=self.shape.verifier_timeout_sec,
                judge_connection=connection,
            )

    async def _grade_in_process(self, relay: ModelRelay | None, connection: JudgeConnection | None) -> GradeResult:
        # Production Verifyit judges use the host judge service. Keep its interpreter
        # in a bounded framework worker, leaving the task lock and actor environment untouched.
        directories = {
            Path(path).resolve() for path in (sys.prefix, sys.base_prefix, *sys.path) if path and Path(path).is_dir()
        }
        roots = tuple(
            path
            for path in sorted(directories)
            if not any(path.is_relative_to(root) for root in SYSTEM_DIRECTORIES)
            and not any(parent in directories for parent in path.parents)
        )
        image = HostImage(read_only=roots, bin_dirs=(Path(sys.executable).parent,))
        env = (
            {}
            if connection is None
            else {
                BASE_URL_ENV: connection.base_url,
                API_KEY_ENV: connection.api_key,
                MODEL_ENV: connection.default_model,
            }
        )
        spec = MachineSpec(
            image,
            cpus=self.shape.cpus,
            memory_mb=self.shape.memory_mb,
            storage_mb=self.shape.storage_mb,
            network=NetworkPolicy.DENY,
            startup_timeout=self._environment_build_timeout_sec,
            env=env,
        )
        async with AsyncExitStack() as stack:
            factory = await stack.enter_async_context(
                grading_machine_factory(
                    _TrialMachineFactory(
                        self.factory, self, relay, stack, observation_role=None,
                        usage_path=self.paths.verifier_dir / "resource-usage.json",
                    ),
                    self.runtime,
                    cleanup_timeout=self.config.trial_cleanup_grace_sec,
                )
            )
            machine = await factory.create(spec)
            stack.push_async_callback(machine.close)
            with TemporaryDirectory(prefix="task-spec-grader-") as directory:
                local = Path(directory)
                payload = local / "task.json"
                payload.write_text(self.specification.model_dump_json())
                await machine.upload(payload, "/tests/task-spec.json")
                conversation = local / "conversation.json"
                conversation.write_text(self.attempt.conversation.model_dump_json())
                await machine.upload(conversation, "/tests/conversation.json")
                command = [
                    "python3",
                    "-m",
                    "taskcompendium.harbor.in_process",
                    "--task",
                    "/tests/task-spec.json",
                    "--conversation",
                    "/tests/conversation.json",
                    "--output",
                    VERDICT_PATH,
                ]
                if self.attempt.stop_reason is not None:
                    command.extend(("--stop-reason", self.attempt.stop_reason))
                result = await machine.run(Command(tuple(command), timeout=self.shape.verifier_timeout_sec))
                if result.reason == ExitReason.TIMED_OUT:
                    raise VerifierTimeoutError("Trusted TaskSpec grader deadline expired")
                if result.exit_code != 0:
                    raise RuntimeError(f"Trusted TaskSpec grader failed: {result.stderr.decode(errors='replace')}")
                verdict = local / "verdict.json"
                await machine.download(VERDICT_PATH, verdict)
                return parse_grade_result(None, verdict.read_bytes())

    async def _recover_outputs(self) -> None:
        await _persist(
            self.paths.agent_dir / "conversation.json",
            self.attempt.conversation.model_dump_json(),
        )
        if isinstance(self.agent, TaskSpecChatAgent) and self.agent.model_input is not None:
            await _persist(self.paths.agent_dir / "model-input.json", self.agent.model_input.model_dump_json())

    async def _collect_runtime_logs(self) -> None:
        pass

    async def _cleanup_agents(self) -> None:
        pass

    async def _stop_agent_environment(self) -> None:
        await self._owned.aclose()
        self._is_agent_environment_stopped = True
