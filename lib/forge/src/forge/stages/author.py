# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hydrate a fresh environment and continue the author's saved conversation."""

import json
from functools import partial
from pathlib import PurePosixPath

from taskcompendium.models import CommandSemantics, EnvironmentRequirements, TaskSpec
from taskcompendium.runtime.resources import resource_bytes

from forge.config import ForgeConfig
from forge.effects.context import Context
from forge.lowering import host_requirements, lowering_settings, requirements
from forge.model.controls import CONTROL, CONTROLS
from forge.model.llm import Message, Role
from forge.model.proposal import render
from forge.model.review import VALIDATION_RESULT
from forge.model.stage import Exit
from forge.model.workpiece import Conversation, Workpiece
from forge.policy import GRADER_BASE_IMAGE, Policy
from forge.resources.bundle import Resources
from forge.resources.env import (
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
from forge.resources.execution import close_resource
from forge.stages.summary import validation_path


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


def install_env(ctx: Context, env: Env) -> None:
    ctx.resource(env, env.fingerprint)
    ctx.handle(RunCommand, partial(run_command, notify=ctx.emit), uses=(env,))
    ctx.handle(ReadFile, partial(read_file, notify=ctx.emit), uses=(env,))
    ctx.handle(WriteFile, partial(write_file, notify=ctx.emit), uses=(env,))
    ctx.handle(WriteFiles, partial(write_files, notify=ctx.emit), uses=(env,))


async def author(ctx: Context, workpiece: Workpiece, config: ForgeConfig, resources: Resources) -> Exit:
    policy = config.policy
    workdir = PurePosixPath(policy.setting("author.workdir", str))
    files = (*config.workspace.static, *config.workspace.rendered(workpiece, policy))
    requested = requirements(
        image=policy.setting("author.image", str) or None,
        packages=policy.setting("author.packages", tuple),
        workdir=str(workdir),
    )
    env = await resources.envs.create(host_requirements(requested))
    try:
        async with ctx.scope("author"):
            install_env(ctx, env)
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
            proposal = render(workpiece.proposal)
            writes.extend(
                (validation_path(str(workdir), index), VALIDATION_RESULT.dump_json(evidence))
                for index, evidence in enumerate(workpiece.evidence)
            )
            writes.append((workdir / "proposal.json", proposal.encode()))
            await ctx.emit(WriteFiles(tuple(writes)))
            tools = config.tools["author"]
            session = workpiece.session or Conversation(
                (
                    Message(Role.SYSTEM, system_prompt(policy)),
                    Message(Role.USER, policy.prompt("author.user").render(proposal=proposal)),
                ),
                tuple(tool.schema for tool in tools.values()),
            )
            return await config.author.run(ctx, session, resources.inference_for("author"), tools)
    finally:
        await close_resource(env.close, ctx.emit, timeout=policy.setting("solver.cleanup_timeout", float))
