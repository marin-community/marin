# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A resource bundle of fakes, installed into a context, for the stage and judge tests."""

from lib.taskforge.tests.resource_doubles import CleanReviewer, FakeInference, FakeWeb
from lib.taskforge.tests.stage_fakes import CandidateEngine, Environments, SolverEngine, solver_model

from taskforge.effects.context import Context
from taskforge.resources.bundle import Resources
from taskforge.resources.grading import GradingResource
from taskforge.resources.ids import Ids
from taskforge.resources.rollouts import RolloutResource
from taskforge.wiring import install


def resources(ctx: Context, restart: int | None = None) -> Resources:
    bundle = Resources(
        Environments(),
        FakeInference(()),
        RolloutResource(SolverEngine(3, restart), solver_model, "solver"),
        GradingResource(CandidateEngine(), "grader"),
        Ids(),
        {},
        FakeWeb({}, {}),
        {"solve_review": CleanReviewer()},
    )
    install(ctx, bundle)
    return bundle
