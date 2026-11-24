# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hydrate a fresh environment and continue the author's saved conversation."""

import json
from dataclasses import dataclass
from pathlib import PurePosixPath

from taskcompendium.models import CommandSemantics, EnvironmentRequirements, TaskSpec
from taskcompendium.runtime.resources import resource_bytes

from taskforge.config import TaskforgeConfig
from taskforge.effects.context import Context
from taskforge.effects.events import Event
from taskforge.effects.log import Billing
from taskforge.harness.protocol import Tool
from taskforge.harness.role import Files, Fresh, Visit, sitting
from taskforge.harness.tools.submit import RevisedSubmitTool, SubmitTool
from taskforge.llm.policy import ReasoningEffort
from taskforge.lowering import FIXED_JUDGE_SETTINGS, JUDGE_RUBRICS, lowering_settings, requirements
from taskforge.model.controls import CONTROL, CONTROLS
from taskforge.model.proposal import render
from taskforge.model.review import VALIDATION_RESULT, Modify, Revise
from taskforge.model.stage import Exit
from taskforge.model.workpiece import Workpiece
from taskforge.policy import GRADER_BASE_IMAGE, Policy
from taskforge.resources.bundle import Resources


@dataclass(frozen=True, slots=True)
class AuthorReference(Event[str]):
    """The TaskSpec and Control JSON schemas the author is shown, offering the decision-model grader kind only with
    ``decision_model``. Every author request embeds the reference, so a root records it once and its later visits and
    launches replay that text: a later wording change in TaskCompendium leaves its recorded author calls valid."""

    kind = "schema.author_reference"
    decision_model: bool


async def author_reference(event: AuthorReference) -> str:
    """The installed schemas, in the compact form the system prompt embeds."""
    task = TaskSpec.model_json_schema()
    if not event.decision_model:
        grader = task["properties"]["grader"]
        del task["$defs"]["DecisionModelGrader"], grader["discriminator"]["mapping"]["decision_model"]
        grader["oneOf"].remove({"$ref": "#/$defs/DecisionModelGrader"})
    judge = task["$defs"]["JudgeSpec"]["properties"]
    for name in (*FIXED_JUDGE_SETTINGS, "request_timeout", "transport_attempts", "transport_budget"):
        del judge[name]
    judge["rubric"]["enum"] = JUDGE_RUBRICS
    judge["reasoning_effort"]["enum"] = ["", *ReasoningEffort]
    return json.dumps({"TaskSpec": task, "Control": CONTROL.json_schema()}, sort_keys=True, separators=(",", ":"))


def system_prompt(policy: Policy, reference: str) -> str:
    """Render the author's system prompt around ``reference``, an ``AuthorReference`` result."""
    decision_model = uses_decision_model(policy)
    environments = tuple(
        environment.model_dump(mode="json", include={"docker_image", "command_semantics"})
        for environment in (
            EnvironmentRequirements(docker_image=GRADER_BASE_IMAGE, command_semantics=CommandSemantics.LINUX_PROCESS),
            EnvironmentRequirements(docker_image=None, command_semantics=CommandSemantics.SHELL_SIMULATOR),
        )
    )
    image = policy.setting("author.image", str)
    environment = (
        "Python 3.12 Debian container" if image == GRADER_BASE_IMAGE else ("container" if image else "ShellSim")
    )
    return policy.prompt("author.system").render(
        reference=reference,
        decision_model=f"\n\n{policy.prompt('author.decision_model').template}" if decision_model else "",
        environment=environment,
        author_image=image or "none (ShellSim)",
        workdir=policy.setting("author.workdir", str),
        output_chars=policy.setting("tools.output_chars", int),
        grader_environments=json.dumps(environments, sort_keys=True),
        sandbox_limits=sandbox_limits(policy),
        task_environments=json.dumps(
            tuple(dict(value, capabilities=["shell"]) for value in environments), sort_keys=True
        ),
    )


def uses_decision_model(policy: Policy) -> bool:
    return bool(policy.setting("decision_model.model", str))


def sandbox_limits(policy: Policy) -> str:
    limits = lowering_settings(policy).task_machine.limits
    return (
        f"Sandbox memory limit: {limits.memory_mb} MiB; CPU request: {limits.cpus}. "
        f"ShellSim cumulative output budget: {policy.setting('sandbox.output_budget_mb', int)} MiB. "
        f"The shell tool defaults to {policy.setting('solver.command_timeout', float):g} seconds per command; "
        f"Its maximum timeout is {policy.setting('solver.tool_timeout', float):g} seconds. "
        f"Solver commands default to {policy.setting('solver.command_timeout', float):g} seconds. "
        "The grader command uses the task's declared grader timeout."
    )


async def author(ctx: Context, workpiece: Workpiece, config: TaskforgeConfig, resources: Resources) -> Exit:
    policy = config.policy
    workdir = PurePosixPath(policy.setting("author.workdir", str))
    machine = requirements(
        image=policy.setting("author.image", str) or None,
        packages=policy.setting("author.packages", tuple),
        workdir=str(workdir),
    )
    proposal = render(workpiece.proposal)

    def workspace() -> Files:
        files = (*config.workspace.static, *config.workspace.rendered(workpiece, policy))
        writes = [(PurePosixPath("/", file.path), resource_bytes(file)) for file in files]
        submission = workpiece.submission
        if submission is not None:
            writes.extend(
                (PurePosixPath("/", file.path), resource_bytes(file))
                for file in (*submission.task.resources.all, *submission.task.resources.worker)
            )
            writes.extend(
                (PurePosixPath("/tests", file.path), resource_bytes(file)) for file in submission.task.resources.verifier
            )
            writes.extend(
                (
                    (workdir / "task.json", submission.task.model_dump_json().encode()),
                    (workdir / "controls.json", CONTROLS.dump_json(submission.controls)),
                )
            )
        for index, evidence in enumerate(workpiece.evidence):
            # Wire JSON writes a nonfinite reward as null; sorted keys keep the bytes independent of decode order.
            wire = VALIDATION_RESULT.dump_json(evidence, exclude={"solver": {"__all__": {"receipt"}}})
            writes.append(
                (workdir / ".validation" / f"{index}.json", json.dumps(json.loads(wire), sort_keys=True).encode())
            )
        writes.append((workdir / "proposal.json", proposal.encode()))
        return tuple(writes)

    # A round has one author visit, which a stage restart opens again on a fresh machine. The caller's files are
    # prepared inside it and the opening after hydration, so either failing is journalled and its machine closed.
    tools: dict[str, Tool[object]] = {}
    last = workpiece.decisions[-1] if workpiece.decisions else None
    sent = (*last.review, *(last.notes if isinstance(last, Revise) else ())) if isinstance(last, Modify | Revise) else ()
    if sent:  # numbered as the revision prompt numbers them: revise findings, then a final review's notes
        submit = config.author.tools["submit"]
        assert isinstance(submit, SubmitTool)
        tools = {"submit": RevisedSubmitTool(submit, sent)}
    visit = Visit("author", machine, workspace, tools, bills=Billing.AUTHOR)
    async with sitting(ctx, config.author, visit, resources) as session:
        if workpiece.session is not None:
            return await session(workpiece.session)
        reference = await ctx.emit(AuthorReference(uses_decision_model(policy)))
        return await session(
            Fresh(system_prompt(policy, reference), policy.prompt("author.user").render(proposal=proposal))
        )
