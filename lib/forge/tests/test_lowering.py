# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import override

import pytest
from rolloutengine.harness.shell import ShellHarness
from rolloutengine.machines import _machine_spec
from rolloutengine.test_support import arithmetic_task, machine_runtime, rollout_limits
from rolloutengine.validation import InvalidTaskSpec
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Backend, ExecutionRequirements, Machine, MachineFactory, MachineSpec
from taskcompendium.grading_result import Outcome
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

from forge.effects.context import Context
from forge.effects.log import EventLog
from forge.harness.protocol import ToolResult
from forge.harness.tools.submit import SubmitTool
from forge.harness.tools.validate import ValidateTool
from forge.lowering import (
    GRADING_MACHINE,
    LoweringSettings,
    lower_with_settings,
    lowering_settings,
    requirements,
)
from forge.model.controls import CONTROLS, Candidate, Control, ControlKind, RewardBand
from forge.model.proposal import TASK_PROPOSAL, TaskProposal
from forge.policy import GRADER_BASE_IMAGE, load_policy, resolution_lines
from forge.resources.env import ReadFile
from forge.resources.sandbox.factories import SANDBOX_SCHEDULING_TIMEOUT, shellsim_factory
from forge.wiring import submit_gate

pytest_plugins = ("lib.forge.tests.test_proposal",)


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
    policy = load_policy(Path(__file__).parents[1] / "src/forge/policy/defaults")
    task = arithmetic_task(environment_requirements=declared.model_copy(update={"capabilities": ("shell",)}))
    spec = lower_with_settings(task, lowering_settings(policy), {"gvisor": AvailableFactory(Backend.GVISOR)})
    assert spec.task.environment_requirements.working_directory == (directory or "/workspace")
    assert spec.task_machine is not None
    assert _machine_spec(spec.task.environment_requirements, spec.task_machine).workdir == (directory or "/workspace")
    assert _machine_spec(declared, machine_runtime(backend="gvisor")).workdir == (directory or "")


def test_policy_memory_and_actual_grader_timeout_reach_both_machine_specs() -> None:
    memory_mb = 16384
    policy = load_policy(Path(__file__).parents[1] / "src/forge/policy/defaults")
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
    policy = load_policy(Path(__file__).parents[1] / "src/forge/policy/defaults")
    task = arithmetic_task(environment_requirements=requirements(image=None))
    spec = lower_with_settings(task, lowering_settings(policy), {"shellsim": ShellSimMachineFactory()})
    assert spec.task_machine is not None and spec.task_machine.cpus is None


@pytest.mark.parametrize("tool_type", [SubmitTool, ValidateTool])
# Full S 20261108 d01.discrete.graph-analysis/06 (also cardinality-analysis/04 and optimization.formulation/04):
# a missing judge reference must return author feedback during parsing and programmatic preflight.
@pytest.mark.parametrize("defect", ["shell", "final_action", "reference", "blank_reference", "checklist"])
async def test_authored_interface_errors_return_tool_feedback(
    tool_type: type[SubmitTool] | type[ValidateTool], defect: str, proposal: TaskProposal
) -> None:
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
        ShellHarness(command_timeout=None).check(task, settings.limits)
    with pytest.raises(InvalidTaskSpec, match=message):
        lower_with_settings(task, settings, factories)
    controls = (Control(Candidate("12"), RewardBand(Outcome.GRADED, 1), ControlKind.POSITIVE),)
    files = {
        "/task.json": task.model_dump_json().encode(),
        "/controls.json": CONTROLS.dump_json(controls),
        "/proposal.json": TASK_PROPOSAL.dump_json(proposal),
    }

    async def read(event: ReadFile) -> bytes:
        return files[str(event.path)]

    ctx = Context(log=EventLog())
    ctx.handle(ReadFile, read)
    tool = tool_type(partial(submit_gate, settings=settings, factories=factories), "/task.json", "/controls.json")
    result = await tool(ctx, {})
    assert isinstance(result, ToolResult) and message in result.content
    await ctx.drain()


def test_shellsim_output_budget_is_declared_and_resolved_without_changing_shared_default() -> None:
    declared = load_policy(Path(__file__).parents[1] / "src/forge/policy/defaults")
    resolved = declared.overlaid({"sandbox.output_budget_mb": 512})
    assert "POLICY sandbox.output_budget_mb declared=1024 resolved=512" in resolution_lines(declared, resolved)
    factory = shellsim_factory(output_budget_mb=resolved.setting("sandbox.output_budget_mb", int))
    assert isinstance(factory, ShellSimMachineFactory)
    assert factory.output == 512 * 1024 * 1024
    assert ShellSimMachineFactory().output == 128 * 1024 * 1024
