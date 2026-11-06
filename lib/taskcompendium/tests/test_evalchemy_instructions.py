# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish source instruction constraints without exposing private grader inputs, and grade IFBench responses.

The IFBench checkers need nltk, emoji, syllapy and NLTK's punkt_tab data, which IFBench's grading environment
provides, so their cases skip without them.
"""

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


REQUEST = "explain the context, code and theory behind the paper Identifying Candidate Spaces for Advert Implantation."
ALPHABET = (
    '"Ants move." Bees move. Cats move. Dogs move. Eels move. Foxes move. Goats move. Hens move. Ibises move. '
    "Jays move. Kites move. Lions move. Moles move. Newts move. Owls move. Pigs move. Quails move. Rats move. "
    "Seals move. Toads move. Urchins move. Voles move. Wolves move. Xerus move. Yaks move. Zebras move."
)
EMPTY_FIELD_TABLE = (
    '"StudentID"\t"Subject"\t"Grade"\t"Semester"\t"Score"\n"1"\t"Math"\t"A"\t"Fall"\t"95"\n'
    '"2"\t\t"B"\t"Fall"\t"85"\n"3"\t"Art"\t"C"\t"Fall"\t"75"'
)
# Each case id starts with its finding id in the marin-community/marin#10060 audit,
# https://github.com/XiaoyuanWang111/ifbench-audit/blob/a31c6841dae2632dafda740bd51496843d9e5f38/reports/AUDIT.md,
# and each expected verdict follows the instruction text.
IFBENCH_CASES = {
    "F1-word-indices": (
        "repeat:repeat_span",
        {"prompt_to_repeat": REQUEST, "n_start": 3, "n_end": 6},
        "code and theory behind",
        True,
    ),
    "F1-character-slice": (
        "repeat:repeat_span",
        {"prompt_to_repeat": REQUEST, "n_start": 3, "n_end": 6},
        "lain",
        False,
    ),
    "F2-word-trigrams": (
        "ratio:overlap",
        {"reference_text": "the cat sat on the mat", "percentage": 0},
        "mat the on sat cat the",
        True,
    ),
    "F6-repeated-words": ("ratio:sentence_words", {}, "I am here. I am here. I am here.", False),
    "F7-whole-words": (
        "count:keywords_multiple",
        {"keyword1": "door", "keyword2": "bread", "keyword3": "blue", "keyword4": "lamp", "keyword5": "river"},
        "The door opened onto a doorway. bread bread. blue blue blue. lamp lamp lamp lamp lamp. "
        "river river river river river river river.",
        True,
    ),
    "F8-punctuation-tokens": (
        "words:words_position",
        {"keyword": "vibrant"},
        "Hello, vibrant world, vibrant day.",
        True,
    ),
    "F9-opening-quote": ("custom:sentence_alphabet", {}, ALPHABET, True),
    "F15-empty-field": ("custom:csv_quotes", {}, EMPTY_FIELD_TABLE, False),
}


@pytest.fixture
def ifbench_grading_environment():
    pytest.importorskip("emoji", reason="Requires the IFBench grading environment")
    pytest.importorskip("syllapy", reason="Requires the IFBench grading environment")
    nltk = pytest.importorskip("nltk", reason="Requires the IFBench grading environment")
    try:
        nltk.data.find("tokenizers/punkt_tab/english/")
    except LookupError:
        pytest.skip("Requires the IFBench grading environment's punkt_tab data")


@pytest.mark.usefixtures("ifbench_grading_environment")
@pytest.mark.parametrize(
    ("instruction_id", "kwargs", "response", "expected"), IFBENCH_CASES.values(), ids=IFBENCH_CASES.keys()
)
@pytest.mark.parametrize("mode", ["strict", "loose"])
def test_ifbench_checker_follows_instruction_text(instruction_id, kwargs, response, expected, mode):
    spec = InstructionSpec(
        constraints=(Constraint(instruction_id, kwargs),), profile=InstructionProfile.IFBENCH, prompt=""
    )

    assert grade_source_instructions(spec, response).detail["instruction_pass"][mode] == [expected]
