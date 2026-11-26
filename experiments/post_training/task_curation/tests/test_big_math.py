# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from taskcompendium.models import Source, judge_spec
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import ImportRejection, NormalizedTask, RawRow

from experiments.post_training.task_curation.datasets import big_math

NATIVE = json.loads((Path(__file__).parent / "fixtures/big_math_problem.json").read_text())


def converted(native):
    row = RawRow(
        id="original-big-math-row",
        source=Source(dataset=big_math.REPO, revision=big_math.REVISION, row="fixture", importer_revision="test"),
        data=native,
    )
    result = big_math.convert_problem(row, ConversionContext({}, None))
    assert isinstance(result, NormalizedTask)
    return result


def rejected(native):
    row = RawRow(
        id="original-big-math-row",
        source=Source(dataset=big_math.REPO, revision=big_math.REVISION, row="fixture", importer_revision="test"),
        data=native,
    )
    result = big_math.convert_problem(row, ConversionContext({}, None))
    assert isinstance(result, ImportRejection)
    return result


def test_context_is_the_problem_and_the_answer_is_the_judge_reference():
    result = converted(NATIVE)
    task = result.task
    (user,) = task.context.events
    assert user.role == "user" and user.content == NATIVE["problem"]
    judge = judge_spec(task.grader)
    assert judge is not None and judge.references == ("3",)
    assert judge.question.startswith("Determine whether the candidate's final answer")
    assert judge.temperature == 0.0 and judge.max_completion_tokens == 64
    assert judge.model == "MiniMaxAI/MiniMax-M3-MXFP8"
    assert task.labels["domain"] == "Mathematics,Number Theory" and task.labels["native_source"] == "number_theory"
    assert not task.resources.worker and not task.resources.oracle
    assert len(result.changes) == 1 and big_math.CARD_URL in result.changes[0].reason
    assert "llama8b_solve_rate" not in user.content


def test_difficulty_metadata_never_enters_the_context_and_the_answer_stays_gold():
    native = {**NATIVE, "answer": "A different synthetic answer, not trusted gold."}
    original = converted(NATIVE).task
    changed = converted(native).task
    (user,) = changed.context.events
    assert user.content == NATIVE["problem"]
    first, second = judge_spec(original.grader), judge_spec(changed.grader)
    assert first is not None and second is not None
    assert first.references != second.references
    assert first.model == second.model == "MiniMaxAI/MiniMax-M3-MXFP8"


def test_contract_defects_fail_closed():
    assert rejected({**NATIVE, "problem": ""}).reason.startswith("incomplete_big_math_contract")
    assert rejected({**NATIVE, "answer": None}).reason.startswith("incomplete_big_math_contract")
    assert rejected({**NATIVE, "domain": "not-a-list"}).reason.startswith("incomplete_big_math_contract")
    assert rejected({**NATIVE, "llama8b_solve_rate": "0.5"}).reason.startswith("incomplete_big_math_contract")
    assert rejected({"problem": NATIVE["problem"]}).reason.startswith("incomplete_big_math_contract")


def test_source_module_binds_the_sealed_population():
    source = big_math.source(("data/train-00000-of-00001.parquet",))
    assert source.info.id == big_math.SOURCE_ID
    assert source.info.count == 251122
    assert source.info.origin.value == "SynthLabs"
    assert source.config.source.revision == big_math.REVISION
    assert "reference judge" in source.info.notes and "llama8b_solve_rate" in source.info.notes
