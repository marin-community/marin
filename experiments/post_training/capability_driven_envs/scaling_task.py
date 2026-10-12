# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A deterministic recipe-scaling task for ``d43.culinary.scaling``: its builder program and a scripted solver.

``scaling_problem(seed)`` draws one problem from a seed (the proposal id): a recipe for a number of
portions, a new portion count, and the ingredient whose scaled weight is the answer.

``PROGRAM`` builds the task without a model: ``problem`` draws the problem from the proposal id,
``recipe`` writes it as the worker file and ``grader`` grades its scaled weight, and the sink ``task``
assembles them with the proposal's source.

``scripted_solver`` builds each solver trial's rollout model. It reads the recipe with one shell call
and replies with the scaled weight; trials whose index is odd reply with the unscaled weight, so a
round of ``k`` trials solves half of them.
"""

import hashlib
import json
import random
import re
from dataclasses import dataclass
from typing import Any

from rolloutengine.contracts import ModelRequest, ModelTurn
from taskcompendium.grader import GraderPackage
from taskcompendium.models import AnswerType, PlainText, Source, TaskResource, TaskSpec
from taskforge.builder.sdk import BuildContext
from taskforge.builder.step import SDK_VERSION, program, step
from taskforge.llm.recording import CallLedger
from taskforge.proposal.model import TaskProposal
from taskforge.spec.draft import answer_grader, assemble, file, requirements
from taskforge.validate.trials import RolloutModel
from verifyit.spec import NumericSpec

WORKDIR = "/workspace"
"""The task machine's working directory."""
WORKSPACE = "workspace"
"""``WORKDIR`` relative to the machine root, where worker files are placed."""
RECIPE_FILE = "recipe.txt"
ANSWER_CONTRACT = "Reply with the number of grams alone, in digits."
SOURCE_DATASET = "taskforge"
INGREDIENTS = ("flour", "butter", "sugar", "milk", "onions", "carrots", "stock", "rice")
DISHES = ("vegetable soup", "risotto", "shortbread", "pilaf")
YIELDS = (4, 6, 8, 10, 12)
TARGET = re.compile(r"to (\d+) portions")
ASKED = re.compile(r"grams of (\w+)")
LINE = re.compile(r"(\w+): (\d+) g")
YIELD_LINE = re.compile(r"Yield: (\d+) portions")


@dataclass(frozen=True)
class ScalingProblem:
    """A recipe of ``portions`` portions, ``grams`` per ingredient, scaled to ``target`` portions."""

    dish: str
    portions: int
    target: int
    grams: tuple[tuple[str, int], ...]
    asked: str

    @property
    def unscaled(self) -> int:
        return dict(self.grams)[self.asked]

    @property
    def answer(self) -> int:
        return self.unscaled * self.target // self.portions


def scaling_problem(seed: str) -> ScalingProblem:
    """The problem drawn from ``seed``: every weight is a whole multiple of the yield, so the answer is exact."""
    rng = random.Random(int(hashlib.sha256(seed.encode()).hexdigest(), 16))
    portions = rng.choice(YIELDS)
    target = rng.choice([n for n in range(2, 61) if n != portions])
    names = rng.sample(INGREDIENTS, 3)
    grams = tuple((name, portions * rng.randint(5, 60)) for name in names)
    return ScalingProblem(rng.choice(DISHES), portions, target, grams, rng.choice(names))


def recipe_text(problem: ScalingProblem) -> str:
    lines = [f"Recipe: {problem.dish}", f"Yield: {problem.portions} portions"]
    return "\n".join([*lines, *(f"{name}: {grams} g" for name, grams in problem.grams)]) + "\n"


def instruction(problem: ScalingProblem) -> str:
    return (
        f"The recipe in {WORKDIR}/{RECIPE_FILE} states its yield. Scale it to {problem.target} portions, "
        f"keeping every ratio. How many grams of {problem.asked} does the scaled recipe use?"
    )


def scaled_grams(recipe: str, request: str) -> int:
    """The answer computed from the recipe file and the instruction, as a solver would."""
    portions = int(_match(YIELD_LINE, recipe))
    target = int(_match(TARGET, request))
    grams = {name: int(value) for name, value in LINE.findall(recipe)}
    return grams[_match(ASKED, request)] * target // portions


def _match(pattern: re.Pattern[str], text: str) -> str:
    found = pattern.search(text)
    if found is None:
        raise ValueError(f"{pattern.pattern!r} does not occur in {text!r}")
    return found.group(1)


@step(inputs=("proposal",))
async def problem(b: BuildContext, proposal: TaskProposal) -> ScalingProblem:
    return scaling_problem(proposal.header.id)


@step(inputs=("problem",))
async def recipe(b: BuildContext, problem: ScalingProblem) -> TaskResource:
    return file(f"{WORKSPACE}/{RECIPE_FILE}", recipe_text(problem))


@step(inputs=("problem",))
async def grader(b: BuildContext, problem: ScalingProblem) -> GraderPackage:
    return answer_grader(NumericSpec(expected=str(problem.answer), tolerance_abs=0.0, tolerance_rel=0.0))


@step(inputs=("problem", "recipe", "grader", "proposal"))
async def task(
    b: BuildContext, problem: ScalingProblem, recipe: TaskResource, grader: GraderPackage, proposal: TaskProposal
) -> TaskSpec:
    header = proposal.header
    return assemble(
        b.item_id,
        f"{instruction(problem)} {ANSWER_CONTRACT}",
        AnswerType.TEXT,
        PlainText(),
        grader,
        Source(dataset=SOURCE_DATASET, revision=proposal.digest, row=header.id, importer_revision=SDK_VERSION),
        environment=requirements(workdir=WORKDIR),
        files=(recipe,),
        tags=(header.environment, header.verification),
    )


PROGRAM = program(problem, recipe, grader, task)


def _shell_call(command: str) -> dict[str, Any]:
    call = {
        "id": "call-0",
        "type": "function",
        "function": {"name": "shell", "arguments": json.dumps({"command": command})},
    }
    return {"role": "assistant", "content": None, "tool_calls": [call]}


def scripted_solver(record: CallLedger) -> RolloutModel:
    """The rollout model of solver trial ``record.step`` (``solver/<index>``): odd indices skip the scaling."""
    skips_scaling = int(record.step.rsplit("/", 1)[-1]) % 2 == 1

    async def model(request: ModelRequest) -> ModelTurn:
        served = sum(message["role"] == "assistant" for message in request.messages)
        prompt = (*request.prefix_token_ids, 2 * served + 1)
        if served == 0:
            return ModelTurn(_shell_call(f"cat {WORKDIR}/{RECIPE_FILE}"), prompt, (2 * served + 2,), None, "tool_calls")
        recipe = json.loads(request.messages[-1]["content"])["stdout"]
        task = next(str(message["content"]) for message in request.messages if message["role"] == "user")
        grams = scaled_grams(recipe, task)
        if skips_scaling:
            grams = {name: int(value) for name, value in LINE.findall(recipe)}[_match(ASKED, task)]
        return ModelTurn({"role": "assistant", "content": str(grams)}, prompt, (2 * served + 2,), None, "stop")

    return model
