# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A deterministic recipe-scaling task for ``d43.culinary.scaling``, and scripted models that build and solve it.

``scaling_problem(seed)`` draws one problem from a seed (the proposal id): a recipe for a number of
portions, a new portion count, and the ingredient whose scaled weight is the answer. It stands in for
GLM at the two places a run calls a model:

* ``ScriptedBuilder`` answers the standard template's structured calls (fixtures, grader, instructions,
  controls) for the proposal named in the template's system message.
* ``scripted_solver`` builds each solver trial's rollout model. It reads the recipe with one shell call
  and replies with the scaled weight; trials whose index is odd reply with the unscaled weight, so a
  round of ``k`` trials solves half of them.
"""

import hashlib
import json
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel
from rolloutengine.contracts import ModelRequest, ModelTurn
from taskforge.builder.template.standard import WORKDIR, WORKSPACE, ControlDraft
from taskforge.llm.policy import Message
from taskforge.llm.recording import CallLedger
from taskforge.proposal.model import parse
from taskforge.validate.trials import RolloutModel

RECIPE_FILE = "recipe.txt"
PROPOSAL_MARKER = "# Proposal\n\n"
"""Where the template's system message (``standard.TASK_CONTEXT``) puts the proposal text."""
INGREDIENTS = ("flour", "butter", "sugar", "milk", "onions", "carrots", "stock", "rice")
DISHES = ("vegetable soup", "risotto", "shortbread", "pilaf")
YIELDS = (4, 6, 8, 10, 12)
TARGET = re.compile(r"to (\d+) portions")
ASKED = re.compile(r"grams of (\w+)")
LINE = re.compile(r"(\w+): (\d+) g")
YIELD_LINE = re.compile(r"Yield: (\d+) portions")
BUILDER_MODEL = "scripted-scaling-builder"


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


def _file(name: str, content: str) -> dict[str, object]:
    return {"path": f"{WORKSPACE}/{name}", "content": content, "executable": False}


def _control(control_id: str, kind: str, category: str, concern: str, reply: str, *, passes: bool) -> dict[str, object]:
    return ControlDraft(
        id=control_id,
        kind=kind,
        category=category,
        concern=concern,
        final_reply=reply,
        files=[],
        reward_min=0.99 if passes else None,
        reward_max=None if passes else 0.0,
        rationale="fixed by the scaling problem",
    ).model_dump(mode="json")


def builder_answers(problem: ScalingProblem) -> dict[str, dict[str, Any]]:
    """The arguments of each structured call of ``standard.build`` for ``problem``, by tool name."""
    answer = str(problem.answer)
    per_portion = str(problem.unscaled // problem.portions)
    return {
        "submit_fixtures": {
            "agent_files": [_file(RECIPE_FILE, recipe_text(problem))],
            "private_files": [],
            "facts": (
                f"{problem.asked}: {problem.unscaled} g for {problem.portions} portions, {answer} g for "
                f"{problem.target}"
            ),
        },
        "submit_grader": {
            "kind": "numeric",
            "expected": answer,
            "tolerance": 0.0,
            "answer_contract": "Reply with the number of grams alone, in digits.",
            "reference_reply": answer,
            "reference_files": [],
            "secret_values": [answer],
        },
        "submit_instructions": {
            "system": "",
            "instruction": f"{instruction(problem)} Reply with the number of grams alone, in digits.",
        },
        "submit_controls": {
            "controls": [
                _control("reference", "positive", "known_correct", "reference", answer, passes=True),
                _control("decimal", "positive", "known_correct", "acceptance", f"{answer}.0", passes=True),
                _control("unscaled", "negative", "plausible_wrong", "acceptance", str(problem.unscaled), passes=False),
                _control("per-portion", "negative", "task_specific_shortcut", "shortcut", per_portion, passes=False),
                _control("empty", "malformed", "empty_or_malformed", "extraction", "", passes=False),
            ]
        },
    }


@dataclass(frozen=True)
class BuilderEndpoint:
    model: str


@dataclass(frozen=True)
class ScriptedBuilder:
    """A ``builder.sdk.ModelEndpoint`` that answers the standard template's structured calls for a scaling problem."""

    endpoint: BuilderEndpoint = BuilderEndpoint(BUILDER_MODEL)

    async def structured[T: BaseModel](self, messages: Sequence[Message], output_type: type[T], name: str) -> T:
        system = str(messages[0]["content"])
        proposal = parse(system[system.index(PROPOSAL_MARKER) + len(PROPOSAL_MARKER) :])
        return output_type.model_validate(builder_answers(scaling_problem(proposal.header.id))[name])


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
