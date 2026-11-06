# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hydrate a fresh environment and continue the author's saved conversation."""

import json
from functools import partial
from pathlib import PurePosixPath

from taskcompendium.models import CommandSemantics, EnvironmentRequirements, TaskResource, TaskSpec
from taskcompendium.runtime.resources import resource_bytes

from forge.config import ForgeConfig
from forge.effects.context import Context
from forge.harness.classify import host_failure
from forge.harness.events import ToolReturned
from forge.lowering import host_requirements, requirements
from forge.model import serialization as wire
from forge.model.llm import Message
from forge.model.proposal import render
from forge.model.stage import Exit
from forge.model.workpiece import Conversation, Workpiece
from forge.policy import Policy
from forge.resources.bundle import Resources
from forge.resources.env import Env, ReadFile, RunCommand, WriteFile, read_file, run_command, write_file
from forge.resources.sandbox.images import GRADER_BASE_IMAGE


def system_prompt(policy: Policy) -> str:
    """Render the author reference from the current task and control schemas."""
    reference = json.dumps(
        {"TaskSpec": TaskSpec.model_json_schema(), "Control": wire.CONTROL.json_schema()},
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
        task_environments=json.dumps(
            tuple(dict(value, capabilities=["shell"]) for value in environments), sort_keys=True
        ),
    )


def install_env(ctx: Context, env: Env) -> None:
    ctx.resource(env, env.fingerprint)
    ctx.handle(RunCommand, partial(run_command, notify=ctx.emit), uses=(env,))
    ctx.handle(ReadFile, partial(read_file, notify=ctx.emit), uses=(env,))
    ctx.handle(WriteFile, partial(write_file, notify=ctx.emit), uses=(env,))
    ctx.observe(ToolReturned, host_failure)


async def hydrate(ctx: Context, files: tuple[TaskResource, ...], root: str = "/") -> None:
    for file in files:
        await ctx.emit(WriteFile(PurePosixPath(root, file.path), resource_bytes(file)))


async def author(ctx: Context, workpiece: Workpiece, config: ForgeConfig, resources: Resources) -> Exit:
    policy = config.policy
    workdir = PurePosixPath(policy.setting("author.workdir", str))
    files = (*config.workspace.static, *config.workspace.rendered(workpiece, policy))
    requested = requirements(
        image=policy.setting("author.image", str) or None,
        packages=policy.setting("author.packages", tuple),
        workdir=str(workdir),
    )
    env = await resources.envs.create(
        host_requirements(requested, resources.factories, tuple(f"/{f.path}" for f in files))
    )
    try:
        async with ctx.scope("author"):
            install_env(ctx, env)
            await hydrate(ctx, files)
            submission = workpiece.submission
            if submission is not None:
                await hydrate(ctx, (*submission.task.resources.all, *submission.task.resources.worker))
                await hydrate(ctx, submission.task.resources.verifier, "/tests")
                await ctx.emit(WriteFile(workdir / "task.json", wire.TASK_SPEC.dump_json(submission.task)))
                await ctx.emit(WriteFile(workdir / "controls.json", wire.CONTROLS.dump_json(submission.controls)))
            proposal = render(workpiece.proposal)
            await ctx.emit(WriteFile(workdir / "proposal.json", proposal.encode()))
            tools = config.tools["author"]
            session = workpiece.session or Conversation(
                (
                    Message("system", system_prompt(policy)),
                    Message("user", policy.prompt("author.user").render(proposal=proposal)),
                ),
                tuple(tool.schema for tool in tools.values()),
            )
            return await config.author.run(ctx, session, env, resources.inference, tools)
    finally:
        await env.close()
