# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import override

import pytest
from lib.taskforge.tests.resource_doubles import FakeWeb
from lib.taskforge.tests.resources.inference_fakes import ScriptedProvider
from rolloutengine.machines import _machine_spec
from rolloutengine.test_support import arithmetic_task, machine_runtime, rollout_limits
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Backend, ExecutionRequirements, Machine, MachineFactory, MachineSpec
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import (
    CommandSemantics,
    EnvironmentRequirements,
    FinalAction,
    FunctionDefinition,
    NoGrader,
    PlainText,
    ScriptGrader,
    ShellToolBinding,
    StdoutReward,
)
from verifyit.spec import JudgeSpec

from taskforge.effects.context import Context
from taskforge.effects.events import Resolved
from taskforge.effects.log import EventLog, Missing
from taskforge.harness.protocol import ToolResult
from taskforge.harness.tools.submit import SubmitTool
from taskforge.lowering import (
    GRADING_MACHINE,
    InvalidTaskSpec,
    LoweringSettings,
    lower_with_settings,
    lowering_settings,
    requirements,
)
from taskforge.model.controls import CONTROLS, Candidate, Control, ControlKind, RewardBand
from taskforge.policy import GRADER_BASE_IMAGE, Policy, load_policy, resolution_lines
from taskforge.resources.env import ReadFile
from taskforge.resources.grading import GradeCandidate, GradingResource
from taskforge.resources.rollouts import RolloutResource, SolverRollout
from taskforge.resources.sandbox.factories import SANDBOX_SCHEDULING_TIMEOUT, shellsim_factory
from taskforge.wiring import build_resources, submit_gate


@dataclass(frozen=True, slots=True)
class AvailableFactory(MachineFactory):
    backend: Backend

    @override
    def check_environment(self, requirements: ExecutionRequirements, spec: MachineSpec) -> None:
        """The Iris container backend supports this fixture; lowering starts no machine."""

    @override
    async def create(self, spec: MachineSpec) -> Machine:
        raise AssertionError("Lowering must not create a machine")


@pytest.mark.parametrize(
    "image",
    [
        None,
        GRADER_BASE_IMAGE,
        "registry.example.com/python@sha256:" + "a" * 64,
    ],
)
def test_grader_image_precheck_respects_available_host_factories(image: str | None) -> None:
    task = arithmetic_task(
        grader=ScriptGrader(
            argv=("sh", "-c", "echo 1"),
            reward=StdoutReward(),
            environment=requirements(image=image),
            answer_path=None,
            timeout=1,
        )
    )
    settings = LoweringSettings(GRADING_MACHINE, GRADING_MACHINE, rollout_limits())
    factories = {backend.value: AvailableFactory(backend) for backend in (Backend.GVISOR, Backend.SHELLSIM)}
    if image not in (None, GRADER_BASE_IMAGE):
        with pytest.raises(ValueError, match="grader image must be") as raised:
            lower_with_settings(task, settings, factories)
        assert image in str(raised.value) and GRADER_BASE_IMAGE in str(raised.value)
    else:
        lowered = lower_with_settings(task, settings, factories)
        assert lowered.verifier_machine is not None
        assert lowered.verifier_machine.backend == (Backend.SHELLSIM if image is None else Backend.GVISOR)


@pytest.mark.parametrize("directory", [None, "/custom"])
def test_missing_workdir_resolves_before_machine_creation(directory: str | None) -> None:
    declared = EnvironmentRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS, docker_image=GRADER_BASE_IMAGE, working_directory=directory
    )
    policy = load_policy(Path(__file__).parents[1] / "src/taskforge/policy/defaults")
    task = arithmetic_task(environment_requirements=declared.model_copy(update={"capabilities": ("shell",)}))
    spec = lower_with_settings(task, lowering_settings(policy), {"gvisor": AvailableFactory(Backend.GVISOR)})
    assert spec.task.environment_requirements.working_directory == (directory or "/workspace")
    assert spec.task_machine is not None
    assert _machine_spec(spec.task.environment_requirements, spec.task_machine).workdir == (directory or "/workspace")
    assert _machine_spec(declared, machine_runtime(backend="gvisor")).workdir == (directory or "")


def test_policy_memory_and_actual_grader_timeout_reach_both_machine_specs() -> None:
    memory_mb = 16384
    policy = load_policy(Path(__file__).parents[1] / "src/taskforge/policy/defaults")
    policy = policy.overlaid({"sandbox.memory_mb": memory_mb})
    settings = lowering_settings(policy)
    package = ScriptGrader(
        argv=("true",),
        reward=StdoutReward(),
        environment=requirements(image=GRADER_BASE_IMAGE),
        answer_path=None,
        timeout=900,
    )
    task = arithmetic_task(grader=package, environment_requirements=requirements(image=GRADER_BASE_IMAGE))
    spec = lower_with_settings(task, settings, {"gvisor": AvailableFactory(Backend.GVISOR)})
    assert spec.task_machine is not None and spec.verifier_machine is not None
    assert spec.task_machine.memory_mb == spec.verifier_machine.memory_mb == memory_mb
    assert spec.task_machine.cpus == spec.verifier_machine.cpus == 1
    assert spec.limits.verifier_timeout == SANDBOX_SCHEDULING_TIMEOUT + 900


def test_shellsim_binding_omits_host_cpu_allocation() -> None:
    policy = load_policy(Path(__file__).parents[1] / "src/taskforge/policy/defaults")
    task = arithmetic_task(environment_requirements=requirements(image=None))
    spec = lower_with_settings(task, lowering_settings(policy), {"shellsim": ShellSimMachineFactory()})
    assert spec.task_machine is not None and spec.task_machine.cpus is None


# Full S 20261108 d01.discrete.graph-analysis/06 (also cardinality-analysis/04 and optimization.formulation/04):
# a missing judge reference must return author feedback during parsing and programmatic preflight.
@pytest.mark.parametrize("defect", ["shell", "final_action", "reference", "blank_reference", "checklist"])
async def test_authored_interface_errors_return_tool_feedback(defect: str) -> None:
    task = arithmetic_task(
        answer_format=FinalAction(normal_submission=PlainText()), grader=NoGrader(reason="interface preflight")
    )
    message = "final action requires at least one final tool"
    if defect == "shell":
        definition = FunctionDefinition(
            name="Bash",
            description="Run",
            parameters={
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"],
            },
        )
        task = arithmetic_task(
            environment_requirements=requirements(image=None),
            interaction_tools=(definition,),
            tool_bindings={"Bash": ShellToolBinding(command_parameter="cmd")},
        )
        message = "Shell tools require exactly one required string argument and no additional properties"
    if defect in {"reference", "blank_reference", "checklist"}:
        judge = JudgeSpec(
            rubric="checklist" if defect == "checklist" else "reference",
            references=(" ",) if defect == "blank_reference" else (),
        )
        grader = ScriptGrader(argv=("true",), environment=requirements(image=None)).model_copy(update={"judge": judge})
        task = arithmetic_task().model_copy(update={"grader": grader})
        message = "needs non-empty criteria" if defect == "checklist" else "needs non-empty reference answers"
    settings = LoweringSettings(GRADING_MACHINE, GRADING_MACHINE, rollout_limits())
    factories = {"shellsim": AvailableFactory(Backend.SHELLSIM)}
    with pytest.raises(InvalidTaskSpec, match=message):
        lower_with_settings(task, settings, factories)
    controls = (Control(Candidate("12"), RewardBand(Outcome.GRADED, 1), ControlKind.POSITIVE),)
    files = {"/task.json": task.model_dump_json().encode(), "/controls.json": CONTROLS.dump_json(controls)}

    async def read(event: ReadFile) -> bytes:
        return files[str(event.path)]

    ctx = Context(log=EventLog())
    ctx.handle(ReadFile, read)
    tool = SubmitTool(partial(submit_gate, settings=settings, factories=factories), "/task.json", "/controls.json")
    result = await tool(ctx, {})
    assert isinstance(result, ToolResult) and message in result.content
    await ctx.drain()


def test_shellsim_output_budget_is_declared_and_resolved_without_changing_shared_default() -> None:
    declared = load_policy(Path(__file__).parents[1] / "src/taskforge/policy/defaults")
    resolved = declared.overlaid({"sandbox.output_budget_mb": 512})
    assert "POLICY sandbox.output_budget_mb declared=1024 resolved=512" in resolution_lines(declared, resolved)
    factory = shellsim_factory(output_budget_mb=resolved.setting("sandbox.output_budget_mb", int))
    assert isinstance(factory, ShellSimMachineFactory)
    assert factory.output == 512 * 1024 * 1024
    assert ShellSimMachineFactory().output == 128 * 1024 * 1024


ROLES = ("author", "solver", "adversary")


@pytest.mark.parametrize(
    "overlay, reruns",
    [
        ({"provider.backoff_initial": 5.0}, False),
        ({"provider.max_connections": 8}, False),
        ({"solver.model": "other-model"}, True),
        ({"sampling.temperature": 0.5}, True),
        ({"sandbox.output_budget_mb": 512}, True),
    ],
)
async def test_a_completed_solver_trial_replays_unless_its_model_sampling_or_output_cap_changes(
    overlay: Mapping[str, object], reruns: bool
) -> None:
    """A tighter ShellSim output budget can exhaust a trial that passed under a generous one."""

    def rollouts(policy: Policy) -> RolloutResource:
        providers = {role: ScriptedProvider((), policy.setting(f"{role}.model", str)) for role in ROLES}
        return build_resources(policy, providers, {}, FakeWeb({}, {})).rollouts

    declared = load_policy(Path(__file__).parents[1] / "src/taskforge/policy/defaults")
    spec = lower_with_settings(arithmetic_task(), lowering_settings(declared), {})
    log = EventLog()
    await log.record(Resolved(SolverRollout(spec, 0, 0), (rollouts(declared).fingerprint(),), "graded"))
    found = log.lookup(SolverRollout(spec, 0, 0), (rollouts(declared.overlaid(overlay)).fingerprint(),))
    assert isinstance(found, Missing) is reruns


@pytest.mark.parametrize(
    "overlay, reruns", [({"provider.backoff_initial": 5.0}, False), ({"sandbox.output_budget_mb": 512}, True)]
)
async def test_a_cached_candidate_grade_replays_unless_the_output_cap_changes(
    overlay: Mapping[str, object], reruns: bool
) -> None:
    """Candidate grading runs on the solver's machines, so a tighter ShellSim cap must regrade a control."""

    def grading(policy: Policy) -> GradingResource:
        providers = {role: ScriptedProvider((), policy.setting(f"{role}.model", str)) for role in ROLES}
        return build_resources(policy, providers, {}, FakeWeb({}, {})).grading

    declared = load_policy(Path(__file__).parents[1] / "src/taskforge/policy/defaults")
    spec = lower_with_settings(arithmetic_task(), lowering_settings(declared), {})
    event = GradeCandidate(spec, Candidate("4"))
    log = EventLog()
    await log.record(Resolved(event, (grading(declared).fingerprint(),), GradeResult(Outcome.GRADED, 1.0)))
    found = log.lookup(event, (grading(declared.overlaid(overlay)).fingerprint(),))
    assert isinstance(found, Missing) is reruns
