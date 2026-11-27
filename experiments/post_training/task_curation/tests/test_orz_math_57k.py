# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

from taskcompendium.models import CommandSemantics, EnvironmentRequirements, Source
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import ImportRejection, NormalizedTask, RawRow
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.task_curation.datasets import orz_math_57k
from experiments.post_training.task_curation.datasets.orz_math_grade import extract_candidate, score

TRANSCRIPT = [
    {
        "from": "human",
        "value": "Diameter $AB$ of a circle has length a $2$-digit integer. Perpendicular chords of lengths ...",
    },
    {"from": "assistant", "ground_truth": {"value": "65"}},
]

NATIVE = {"problem": TRANSCRIPT[0]["value"], "ground_truth": "65"}


def converted(native):
    row = RawRow(
        id="original-orz-row",
        source=Source(
            dataset=orz_math_57k.REPO, revision=orz_math_57k.REVISION, row="fixture", importer_revision="test"
        ),
        data=native,
    )
    result = orz_math_57k.convert_problem(
        row,
        ConversionContext(
            {},
            EnvironmentRequirements(
                command_semantics=CommandSemantics.LINUX_PROCESS,
                packages_lock="s3://test/orz-requirements.lock",
            ),
        ),
    )
    assert isinstance(result, NormalizedTask)
    return result


def rejected(native):
    row = RawRow(
        id="original-orz-row",
        source=Source(
            dataset=orz_math_57k.REPO, revision=orz_math_57k.REVISION, row="fixture", importer_revision="test"
        ),
        data=native,
    )
    result = orz_math_57k.convert_problem(row, _context())
    assert isinstance(result, ImportRejection)
    return result


def _context():
    return ConversionContext(
        {},
        EnvironmentRequirements(
            command_semantics=CommandSemantics.LINUX_PROCESS, packages_lock="s3://test/orz-requirements.lock"
        ),
    )


def test_the_context_is_the_verbatim_problem_and_the_gold_stays_out():
    result = converted(NATIVE)
    (user,) = result.task.context.events
    assert user.content == NATIVE["problem"]
    assert len(result.changes) == 1 and "assistant turn" in result.changes[0].reason
    config = next(resource for resource in result.task.resources.verifier if resource.path == "config.json")
    payload = json.loads(resource_bytes(config))
    assert payload["ground_truth"] == "65"
    grade = next(resource for resource in result.task.resources.verifier if resource.path == "grade.py")
    assert b"math_verify" in resource_bytes(grade) and b"boxed" in resource_bytes(grade)
    assert not result.task.resources.oracle and not result.task.resources.worker


def test_the_boxed_extractor_and_verifier():
    reply = "We need $x=\\boxed{65}$."
    assert extract_candidate(reply) == "65"
    assert score(reply, "65") == 1
    assert score("The answer is 64.", "65") == 0
    assert score("no boxed answer here", "65") == 0
    # equivalent forms verify as mathematics
    assert score("So $x = \\boxed{13 \\times 5}$.", "65") == 1


def test_contract_defects_fail_closed():
    assert rejected({**NATIVE, "problem": ""}).reason.startswith("incomplete_orz_contract")
    assert rejected({**NATIVE, "ground_truth": ""}).reason.startswith("incomplete_orz_contract")
    assert rejected({"problem": NATIVE["problem"]}).reason.startswith("incomplete_orz_contract")


def test_the_source_module_binds_the_sealed_population():
    source = orz_math_57k.source(("orz_math_57k_collection.json",))
    assert source.info.id == orz_math_57k.SOURCE_ID
    assert source.info.count == 56878
    assert source.info.origin.value == "Open-Reasoner-Zero"
    assert "reference answer" in source.info.notes and "math_verify" in source.info.notes
