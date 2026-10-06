# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from verifyit.grade import Status

from taskcompendium.pipeline.datasets.grader_scripts import references as grade_judge


def test_source_reference_gate_leaves_paraphrases_ungraded():
    references = ("Paris",)
    exact = grade_judge.grade_reference_candidate(references, r"\boxed{The Paris.}")
    paraphrase = grade_judge.grade_reference_candidate(references, "France's capital is Paris.")
    blank = grade_judge.grade_reference_candidate(references, "  ")
    assert (exact.status, exact.reward) == (Status.SCORED, 1.0)
    assert paraphrase.status == Status.INFRA_ERROR
    assert (blank.status, blank.reward) == (Status.SCORED, 0.0)


def test_source_abstention_gate_checks_reference_before_refusal():
    references = ("I don't know",)
    exact = grade_judge.grade_abstention_candidate(references, None, r"\boxed{I don't know}")
    rejected = grade_judge.grade_abstention_candidate(("Paris",), None, r"\boxed{[IDK]}")
    unresolved = grade_judge.grade_abstention_candidate(("Paris",), None, "France's capital is Paris.")
    assert (exact.status, exact.reward) == (Status.SCORED, 1.0)
    assert (rejected.status, rejected.reward) == (Status.SCORED, 0.0)
    assert unresolved.status == Status.INFRA_ERROR
