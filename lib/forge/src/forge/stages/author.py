# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hydrate a fresh environment and continue the author's saved conversation."""

from functools import partial
from pathlib import PurePosixPath

from taskcompendium.models import TaskResource
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
from forge.resources.bundle import Resources
from forge.resources.env import Env, ReadFile, RunCommand, WriteFile, read_file, run_command, write_file


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
                    Message("system", policy.prompt("author.system").render()),
                    Message("user", policy.prompt("author.user").render(proposal=proposal)),
                ),
                tuple(tool.schema for tool in tools.values()),
            )
            return await config.author.run(ctx, session, env, resources.inference, tools)
    finally:
        await env.close()
