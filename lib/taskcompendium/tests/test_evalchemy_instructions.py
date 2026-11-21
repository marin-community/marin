# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish source instruction constraints without exposing private grader inputs."""

import json
from functools import partial

import pytest
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from shellbox.machine import PackageSpec
from verifyit.spec import Constraint
from zephyr.context import ZephyrContext

from taskcompendium.evaluation.corpus import evaluation_cases
from taskcompendium.evaluation.spec import EvaluationProtocol, Metrics, OperationKind, Sampling, SummaryKind, TrialPolicy
from taskcompendium.importers.evalchemy.instructions import convert_instructions, instruction_item
from taskcompendium.importers.evalchemy.scorers.instruction_spec import InstructionProfile, InstructionSpec
from taskcompendium.importers.evalchemy.scorers.instructions import grade_source_instructions
from taskcompendium.models import CommandSemantics, EnvironmentRequirements, TextMessage
from taskcompendium.pipeline.evaluation import publish_evaluation_corpus
from taskcompendium.pipeline.inputs import SourceFiles, SourceFormat
from taskcompendium.pipeline.models import IntendedUse, SourceRecipe
from taskcompendium.runtime.resources import resource_bytes

EUROPEAN_CAPITALS_NORTH_OF_45 = (
    "Reykjavik",
    "Helsinki",
    "Oslo",
    "Tallinn",
    "Stockholm",
    "Riga",
    "Moscow",
    "Copenhagen",
    "Vilnius",
    "Minsk",
    "Dublin",
    "Berlin",
    "Amsterdam",
    "Warsaw",
    "London",
    "Brussels",
    "Kyiv",
    "Prague",
    "Luxembourg",
    "Paris",
    "Vienna",
    "Bratislava",
    "Budapest",
    "Vaduz",
    "Chisinau",
    "Bern",
    "Ljubljana",
    "Zagreb",
)


@pytest.mark.parametrize(
    ("capitals", "expected"),
    [
        (EUROPEAN_CAPITALS_NORTH_OF_45, 1),
        (tuple(capital for capital in EUROPEAN_CAPITALS_NORTH_OF_45 if capital != "Kyiv"), 0),
        ((*EUROPEAN_CAPITALS_NORTH_OF_45[:15], "Kyiv", "Brussels", *EUROPEAN_CAPITALS_NORTH_OF_45[17:]), 0),
        ((*EUROPEAN_CAPITALS_NORTH_OF_45[:16], "Rome", *EUROPEAN_CAPITALS_NORTH_OF_45[17:]), 0),
    ],
)
def test_ifbench_european_capitals_requires_complete_latitude_order(capitals, expected):
    pytest.importorskip("nltk")
    pytest.importorskip("emoji")
    pytest.importorskip("syllapy")
    spec = InstructionSpec(
        constraints=(Constraint("custom:european_capitals_sort", {}),),
        profile=InstructionProfile.IFBENCH,
        prompt="List European capitals north of 45 degrees in descending latitude order.",
    )

    result = grade_source_instructions(spec, ", ".join(capitals))

    assert result.reward == expected
    assert result.detail["metrics"]["strict_prompt_accuracy"] == expected
    assert result.detail["metrics"]["loose_prompt_accuracy"] == expected


@pytest.mark.parametrize("profile", list(InstructionProfile))
def test_instruction_publication_preserves_prompt_channel_and_private_constraints(profile, tmp_path):
    prompt = "Respond with two bullet points."
    row = {
        "prompt": prompt,
        "instruction_id_list": ["detectable_format:number_bullet_lists"],
        "kwargs": [{"num_bullets": 2, "unused_checker_parameter": None}],
    }
    root = tmp_path / "source"
    root.mkdir()
    (root / "input.jsonl").write_text(json.dumps(row) + "\n")
    recipe = SourceRecipe(
        name="instruction-publication",
        version="fixture",
        source=SourceFiles("instruction-fixture", "fixture", ("input.jsonl",), SourceFormat.JSONL),
        convert=partial(convert_instructions, profile=profile),
        intended_use=IntendedUse.EVAL,
        rubric=None,
        controls=None,
        grader_environment=EnvironmentRequirements(
            command_semantics=CommandSemantics.LINUX_PROCESS,
            package_spec=PackageSpec(pypi_packages=("absl-py==2.3.1",)),
        ),
    )
    operation = OperationKind.TEXT_GENERATION if profile is InstructionProfile.HARNESS else OperationKind.CHAT_GENERATION
    protocol = EvaluationProtocol(
        benchmark="instructions",
        version="fixture",
        operation=operation,
        sampling=Sampling(temperature=0, top_p=None, top_k=None, max_tokens=128, stop=(), chat_template_kwargs={}),
        trials=TrialPolicy(count=1, seed=None),
        metrics=(Metrics.ACCURACY,),
        primary_metric="accuracy",
        summary=SummaryKind.WEIGHTED_MEAN,
        expected_population=1,
    )
    output = str(tmp_path / "published")
    client = LocalClient()
    try:
        with (
            set_current_client(client),
            ZephyrContext(client=client, max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context,
        ):
            publish_evaluation_corpus(
                recipe, protocol, partial(instruction_item, profile=profile), context, str(root), output
            )
        (case,) = list(evaluation_cases(output))
    finally:
        client.shutdown()
    assert case.task.context.events == (TextMessage(role="user", content=prompt),)
    assert case.item.operation.kind == operation
    assert not case.task.resources.worker
    private = {resource.path: resource_bytes(resource) for resource in case.task.resources.verifier}
    assert json.loads(private["config.json"]) == {
        "constraints": [{"name": "detectable_format:number_bullet_lists", "params": {"num_bullets": 2}}],
        "profile": profile.value,
        "prompt": prompt,
    }
    assert any(
        path.startswith("source_grader/" + ("ifeval" if profile is InstructionProfile.HARNESS else "ifbench") + "/")
        for path in private
    )
