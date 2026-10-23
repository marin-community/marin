# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build fixtures: a small proposal, laptop build services, and a step context over the fake GLM.

``services`` builds ``RLEnvBuildServices`` on ShellSim, with the ShellSim-backed
``ShellGraderImageFactory`` registered as Docker for verifier machines and grader-base task machines,
so a grader program the fake GLM writes is sh.
``context`` is a step context for calling SDK functions outside a program; its model is the
``glm_client`` on ``fake_glm``. ``web_tool`` is a ``web_search`` tool that records its queries.
``declaring`` changes the proposal's solver machine, and ``RecordingImageBuilder`` publishes no image.
"""

import json
from collections.abc import Callable, Mapping
from dataclasses import replace

import pytest
from shellbox.machine import Backend

from taskforge.llm.agent import AgentTool, ReasoningReplay
from taskforge.llm.policy import LLMPolicy
from taskforge.sandbox.factories import MachineHost
from taskforge.sandbox.images import DockerBuild
from taskforge.stage10_proposal.model import Environment, SolverMachine, TaskProposal, parse
from taskforge.stage40_build_with_program.context import RLEnvBuildContext
from taskforge.stage40_build_with_program.services import AgentLimits, RLEnvBuildServices
from tests.sandbox.fixture_images import PortlessShellSimFactory, ShellGraderImageFactory

PROPOSAL = json.dumps(
    {
        "header": {
            "id": "d00.arithmetic.products/1",
            "source": {"kind": "capability", "ref": "d00.arithmetic.products", "hash": "abc123"},
            "grounding": "unverified",
            "null_reason": None,
            "environment": "reasoning",
            "solver_machine": "shellsim",
            "verification": "simple",
            "research_need": "needed",
            "research_reason": "the task depends on an external fact the build must " "confirm",
            "research": [{"kind": "web", "purpose": "confirm the multiplication table"}],
            "build": [{"name": "grader", "description": "checks the final ANSWER line"}],
            "resources": ["grader/grade.py"],
        },
        "body": (
            """## Task
Multiply 6 by 7 and end the reply with `ANSWER = <n>`.

## Realism and workflow
A clerk checks a product.

## Research plan
Confirm the table. Sources: research[0].

## Build plan
Write the question and the grader.

## Grader design and controls
Full credit only for ANSWER = 42.

## Risks and null conditions
None.
"""
        ),
    }
)
WORKPIECE = "d00.arithmetic.products--1"
SAMPLING = LLMPolicy()
FACTORIES = {Backend.SHELLSIM.value: PortlessShellSimFactory(), Backend.DOCKER.value: ShellGraderImageFactory()}
LIMITS = AgentLimits(max_turns=8, max_tokens=4096, max_calls=16)


IMAGE = f"registry.example/taskforge-tasks/d00@sha256:{'2' * 64}"


@pytest.fixture
def proposal() -> TaskProposal:
    return parse(PROPOSAL)


def declaring(proposal: TaskProposal, machine: SolverMachine | None) -> TaskProposal:
    """``proposal`` declaring ``machine``, as a container proposal when ``machine`` is ``None``."""
    environment = Environment.CONTAINER if machine is None else proposal.header.environment
    return replace(proposal, header=replace(proposal.header, environment=environment, solver_machine=machine))


class RecordingImageBuilder:
    """An ``ImageBuilder`` that records each build and returns ``IMAGE`` without building it."""

    def __init__(self) -> None:
        self.published: list[tuple[DockerBuild, str]] = []

    async def publish(self, build: DockerBuild, repository: str) -> str:
        self.published.append((build, repository))
        return IMAGE


@pytest.fixture
def services(ledger) -> RLEnvBuildServices:
    return RLEnvBuildServices(
        host=MachineHost.LAPTOP,
        factories=FACTORIES,
        images=None,
        web_tools=(),
        ledger=ledger,
        reasoning_replay=ReasoningReplay.KEEP,
        agent_limits=LIMITS,
    )


@pytest.fixture
def context(glm_client, services) -> Callable[..., RLEnvBuildContext]:
    """``context(**service_changes)``: a step context for ``WORKPIECE`` with ``services`` changed as given."""

    def make(**changes: object) -> RLEnvBuildContext:
        return RLEnvBuildContext(
            workpiece_id=WORKPIECE,
            round=0,
            model=glm_client,
            sampling=SAMPLING,
            services=replace(services, **changes),
            step="test",
        )

    return make


def web_tool(queries: list[Mapping[str, object]], output: str = "6 x 7 = 42 (https://example.org/table)") -> AgentTool:
    """A ``web_search`` tool that records each query and returns ``output``."""

    async def search(arguments: Mapping[str, object]) -> str:
        queries.append(arguments)
        return output

    return AgentTool(
        name="web_search",
        description="Search the web.",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        handler=search,
    )
