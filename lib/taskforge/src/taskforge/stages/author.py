# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hydrate a fresh environment and continue the author's saved conversation."""

import json
from pathlib import PurePosixPath

from taskcompendium.models import CommandSemantics, EnvironmentRequirements, TaskSpec
from taskcompendium.runtime.resources import resource_bytes

from taskforge.config import TaskforgeConfig
from taskforge.effects.context import Context
from taskforge.harness.role import Files, Fresh, Visit, sitting
from taskforge.lowering import lowering_settings, requirements
from taskforge.model.controls import CONTROL, CONTROLS
from taskforge.model.proposal import render
from taskforge.model.review import VALIDATION_RESULT
from taskforge.model.stage import Exit
from taskforge.model.workpiece import Workpiece
from taskforge.policy import GRADER_BASE_IMAGE, Policy
from taskforge.resources.bundle import Resources
from taskforge.stages.summary import validation_path


def system_prompt(policy: Policy) -> str:
    """Render the author reference from the current task and control schemas."""
    reference = json.dumps(
        {"TaskSpec": TaskSpec.model_json_schema(), "Control": CONTROL.json_schema()},
        sort_keys=True,
        separators=(",", ":"),
    )
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
        writes.extend(
            (validation_path(str(workdir), index), VALIDATION_RESULT.dump_json(evidence))
            for index, evidence in enumerate(workpiece.evidence)
        )
        writes.append((workdir / "proposal.json", proposal.encode()))
        return tuple(writes)

    # A round has one author visit, which a stage restart reconstructs on its fresh machine. The caller's files are
    # prepared inside it and the opening after hydration, so either failing is journalled and its machine closed.
    visit = Visit("author", f"author/{workpiece.round}", machine, workspace)
    async with sitting(ctx, config.author, visit, resources) as session:
        if workpiece.session is not None:
            return await session(workpiece.session)
        return await session(Fresh(system_prompt(policy), policy.prompt("author.user").render(proposal=proposal)))
