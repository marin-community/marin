# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.post_training.abstention_sft.problems import nemotron_problems, normalized_question


def _row(**overrides):
    row = {
        "id": "law-wiki-00040",
        "question": "Which publisher released R. L. Maddex's 2008 work 'Constitutions of the World'?",
        "answer": "CQ Press",
        "domain": "Law",
        "messages": [{"role": "system", "content": "Put your final answer in \\boxed{[IDK]} if unsure."}],
    }
    return {**row, **overrides}


def test_overlapping_question_is_dropped():
    kept = _row(id="kept", question="Who wrote Hamlet?")
    shared = _row(id="shared", question="What is the capital of France?")
    eval_questions = {normalized_question("what is the CAPITAL of France")}
    assert [problem["request_id"] for problem in nemotron_problems([kept, shared], eval_questions)] == ["kept"]


def test_missing_domain_becomes_hotpotqa():
    [problem] = nemotron_problems([_row(id="5ac38c75554299741d48a28a", domain=None)], set())
    assert problem["domain"] == "hotpotqa"


def test_problem_is_the_bare_question():
    [problem] = nemotron_problems([_row()], set())
    assert problem == {
        "request_id": "law-wiki-00040",
        "problem": "Which publisher released R. L. Maddex's 2008 work 'Constitutions of the World'?",
        "answer": "CQ Press",
        "domain": "Law",
        "accepted": True,
    }
