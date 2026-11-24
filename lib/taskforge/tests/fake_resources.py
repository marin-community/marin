# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A resource bundle of fakes, installed into a context, for the stage and judge tests."""

from lib.taskforge.tests.stage_fakes import CandidateEngine, Environments, SolverEngine, solver_model

from taskforge.effects.context import Context
from taskforge.resources.bundle import Resources
from taskforge.resources.clock import Clock
from taskforge.resources.grading import GradingResource
from taskforge.resources.ids import Ids
from taskforge.resources.inference import FakeInference
from taskforge.resources.rollouts import RolloutResource
from taskforge.resources.web import FakeWeb
from taskforge.wiring import install


def resources(ctx: Context, restart: int | None = None) -> Resources:
    bundle = Resources(
        Environments(),
        FakeInference(()),
        RolloutResource(SolverEngine(3, restart), solver_model, "solver"),
        GradingResource(CandidateEngine(), "grader"),
        Clock(),
        Ids(),
        {},
        FakeWeb({}, {}),
    )
    install(ctx, bundle)
    return bundle
