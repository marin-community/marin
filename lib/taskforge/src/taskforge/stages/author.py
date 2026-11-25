# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hydrate a fresh environment and continue the author's saved conversation."""

import json
from dataclasses import dataclass
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
from taskforge.policy import GRADER_BASE_IMAGE, SOURCE_IMAGES, Policy
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
    sources = policy.setting(SOURCE_IMAGES, tuple)
    # Only a source task's run has source images; its author keeps them even on the author.image fallback.
    rule = (
        policy.prompt("author.source_image_rule").render(images=", ".join(f"`{source}`" for source in sources))
        if sources
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


def source_environment(workpiece: Workpiece, policy: Policy) -> EnvironmentRequirements | None:
    """The source task's own container environment for its author, or ``None`` for ``author.image``.

    A task from another source names the image its repository and tests live in, so the author works in
    the same environment, setup included, that the solver and grader get, when that image is one of the
    source's own, ``sandbox.source_images``. A generated task's author always gets ``author.image``.
    """
    submission = workpiece.submission
    if workpiece.source is None or submission is None:
        return None
    environment = submission.task.environment_requirements
    admitted = policy.setting(SOURCE_IMAGES, tuple)
    return host_requirements(environment) if environment.docker_image in admitted else None


def author_image_requirements(policy: Policy) -> EnvironmentRequirements:
    return requirements(
        image=policy.setting("author.image", str) or None,
        packages=policy.setting("author.packages", tuple),
        workdir=policy.setting("author.workdir", str),
    )


@dataclass(frozen=True, slots=True)
class AuthorMachine:
    """The author's started machine, its requirements, and why it is not the task's own when setup failed."""

    env: Env
    requirements: EnvironmentRequirements
    setup_failure: str | None


async def author_machine(workpiece: Workpiece, policy: Policy, resources: Resources) -> AuthorMachine:
    """Start the author's machine, in ``author.image`` when a source task's own environment will not start.

    The source task's environment is then what needs repair, so the author works beside it. Its resources
    are installed before its setup commands run, as RolloutEngine installs them for the solver.
    """
    fallback = host_requirements(author_image_requirements(policy))
    own = source_environment(workpiece, policy)
    if own is None:
        return AuthorMachine(await resources.envs.create(fallback), fallback, None)
    assert workpiece.submission is not None
    files = (*workpiece.submission.task.resources.all, *workpiece.submission.task.resources.worker)
    try:
        return AuthorMachine(await resources.envs.create(own, files), own, None)
    except EnvironmentSetupFailed as error:
        return AuthorMachine(await resources.envs.create(fallback), fallback, str(error))


async def author(ctx: Context, workpiece: Workpiece, config: TaskforgeConfig, resources: Resources) -> Exit:
    policy = config.policy
    workdir = PurePosixPath(policy.setting("author.workdir", str))
    started = await author_machine(workpiece, policy, resources)
    env, machine, setup_failure = started.env, started.requirements, started.setup_failure
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
                    Message(Role.SYSTEM, system_prompt(policy, machine)),
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
