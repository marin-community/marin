# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hydrate a fresh environment and continue the author's saved conversation."""

import json
from functools import partial
from pathlib import PurePosixPath

from taskcompendium.models import CommandSemantics, EnvironmentRequirements, TaskSpec
from taskcompendium.runtime.resources import resource_bytes

from taskforge.config import TaskforgeConfig
from taskforge.effects.context import Context
from taskforge.effects.events import Note
from taskforge.effects.issues import EnvironmentSetupFailed
from taskforge.lowering import host_requirements, lowering_settings, requirements
from taskforge.model.controls import CONTROL, CONTROLS
from taskforge.model.llm import Message, Role
from taskforge.model.proposal import render
from taskforge.model.review import VALIDATION_RESULT, Modify
from taskforge.model.stage import Exit
from taskforge.model.workpiece import Conversation, Workpiece
from taskforge.policy import GRADER_BASE_IMAGE, Policy
from taskforge.resources.bundle import Resources
from taskforge.resources.env import (
    Env,
    ReadFile,
    RunCommand,
    WriteFile,
    WriteFiles,
    read_file,
    run_command,
    write_file,
    write_files,
)
from taskforge.resources.execution import close_resource
from taskforge.stages.summary import validation_path


def system_prompt(policy: Policy, machine: EnvironmentRequirements) -> str:
    """Render the author reference from the current task and control schemas, for the author's ``machine``."""
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
    image = machine.docker_image
    workdir = policy.setting("author.workdir", str)
    sources = policy.setting("sandbox.source_images", tuple)
    rule = (
        policy.prompt("author.source_image_rule").render(images=", ".join(f"`{source}`" for source in sources))
        if image in sources
        else policy.prompt("author.image_rule").template
    )
    environment = (
        "Python 3.12 Debian container" if image == GRADER_BASE_IMAGE else ("container" if image else "ShellSim")
    )
    return policy.prompt("author.system").render(
        reference=reference,
        environment=environment,
        author_image=image or "none (ShellSim)",
        workdir=workdir,
        shell_directory=machine.working_directory or workdir,
        image_rule=rule,
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


def install_env(ctx: Context, env: Env) -> None:
    ctx.resource(env, env.fingerprint)
    ctx.handle(RunCommand, partial(run_command, notify=ctx.emit), uses=(env,))
    ctx.handle(ReadFile, partial(read_file, notify=ctx.emit), uses=(env,))
    ctx.handle(WriteFile, partial(write_file, notify=ctx.emit), uses=(env,))
    ctx.handle(WriteFiles, partial(write_files, notify=ctx.emit), uses=(env,))


def opening_prompt(workpiece: Workpiece, policy: Policy) -> str:
    """The first user message of a new author session: the proposal to build, or review's findings to repair.

    A workpiece without a proposal holds a task from another source and reaches the author only through a
    review ``Modify``.
    """
    if workpiece.proposal is not None:
        return policy.prompt("author.user").render(proposal=render(workpiece.proposal))
    decision = workpiece.decisions[-1] if workpiece.decisions else None
    if not isinstance(decision, Modify):
        raise ValueError(f"Workpiece {workpiece.id} has no proposal, so its author session needs review's findings")
    return policy.prompt("author.repair").render(kind="review", detail="\n".join(decision.findings))


def author_requirements(workpiece: Workpiece, policy: Policy) -> EnvironmentRequirements:
    """The author's machine: the task's own container when it is a source image, else ``author.image``.

    A task from another source names the image its repository and tests live in, so the author works in
    the same environment, setup included, that the solver and grader get. Only a source task's images are
    in ``sandbox.source_images``, so a generated task's author always gets ``author.image``.
    """
    submission = workpiece.submission
    if submission is not None and submission.task.environment_requirements.docker_image in policy.setting(
        "sandbox.source_images", tuple
    ):
        return host_requirements(submission.task.environment_requirements)
    return author_image_requirements(policy)


def author_image_requirements(policy: Policy) -> EnvironmentRequirements:
    return requirements(
        image=policy.setting("author.image", str) or None,
        packages=policy.setting("author.packages", tuple),
        workdir=policy.setting("author.workdir", str),
    )


async def author(ctx: Context, workpiece: Workpiece, config: TaskforgeConfig, resources: Resources) -> Exit:
    policy = config.policy
    workdir = PurePosixPath(policy.setting("author.workdir", str))
    machine = author_requirements(workpiece, policy)
    setup_failure: str | None = None
    try:
        env = await resources.envs.create(host_requirements(machine))
    except EnvironmentSetupFailed as error:
        if machine.docker_image not in policy.setting("sandbox.source_images", tuple):
            raise
        # The source task's environment is what needs repair, so the author works beside it in author.image.
        setup_failure = str(error)
        machine = author_image_requirements(policy)
        env = await resources.envs.create(host_requirements(machine))
    try:
        async with ctx.scope("author"):
            install_env(ctx, env)
            files = (*config.workspace.static, *config.workspace.rendered(workpiece, policy))
            writes = [(PurePosixPath("/", file.path), resource_bytes(file)) for file in files]
            submission = workpiece.submission
            if submission is not None:
                writes.extend(
                    (PurePosixPath("/", file.path), resource_bytes(file))
                    for file in (*submission.task.resources.all, *submission.task.resources.worker)
                )
                writes.extend(
                    (PurePosixPath("/tests", file.path), resource_bytes(file))
                    for file in submission.task.resources.verifier
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
            proposal = None if workpiece.proposal is None else render(workpiece.proposal)
            if proposal is not None:
                writes.append((workdir / "proposal.json", proposal.encode()))
            await ctx.emit(WriteFiles(tuple(writes)))
            tools = config.tools["author"]
            session = workpiece.session or Conversation(
                (
                    Message(Role.SYSTEM, system_prompt(policy, host_requirements(machine))),
                    Message(Role.USER, opening_prompt(workpiece, policy)),
                ),
                tuple(tool.schema for tool in tools.values()),
            )
            if setup_failure is not None:
                await ctx.emit(Note(f"Author falls back to author.image: {setup_failure}"))
                notice = policy.prompt("author.environment_failure").render(
                    error=setup_failure, image=machine.docker_image or "ShellSim"
                )
                session = Conversation((*session.messages, Message(Role.USER, notice)), session.tool_schemas)
            return await config.author.run(ctx, session, resources.inference_for("author"), tools)
    finally:
        await close_resource(env.close, ctx.emit)
