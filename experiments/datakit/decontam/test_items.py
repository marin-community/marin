# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The item table rule for lm-eval tasks: question, choices and answer from the task's own fields."""

import pytest

from experiments.datakit.decontam.items import lmh_item, shared_values


class _Task:
    def __init__(self, text, target, choices=None):
        self.OUTPUT_TYPE = "multiple_choice" if choices is not None else "generate_until"
        self._text, self._target, self._choices = text, target, choices

    def doc_to_text(self, doc):
        return self._text(doc)

    def doc_to_target(self, doc):
        return self._target(doc)

    def doc_to_choice(self, doc):
        return self._choices(doc)


@pytest.mark.parametrize(
    "doc",
    [
        {"question": "2 x 3?", "choices": ["6", "12"]},
        {"question": "2 x 3?", "A": "6", "B": "12"},
        {"question": "2 x 3?", "option1": "6", "option2": "12"},
    ],
)
def test_lettered_choices_read_the_option_texts_of_the_doc(doc):
    task = _Task(lambda d: f"{d['question']}\nA. 6\nB. 12\nAnswer:", lambda d: 0, lambda d: ["A", "B"])
    item = lmh_item("e", "e-0", task, doc, set())
    assert item["choices"] == ["6", "12"] and item["answer"] == "6"


def test_choices_written_with_their_own_letters_lose_the_letters():
    agieval = _Task(lambda d: d["query"], lambda d: 1, lambda d: d["choices"])
    item = lmh_item("agieval", "agieval-0", agieval, {"query": "How far?", "choices": ["(A)2.5 m", "(B)5 m"]}, set())
    assert item["choices"] == ["2.5 m", "5 m"] and item["answer"] == "5 m"


def test_options_written_in_the_prompt_are_parsed_and_a_letter_target_names_one():
    bbh = _Task(lambda d: "Q: " + d["input"] + "\nA: Let's think step by step.", lambda d: "(B)")
    doc = {"input": "Which is the humorous edit?\nOptions:\n(A) deadpoel\n(B) deadfool"}
    item = lmh_item("bbh", "bbh-0", bbh, doc, set())
    assert item["choices"] == ["deadpoel", "deadfool"] and item["answer"] == "deadfool"
    bullets = _Task(lambda d: d["input"] + "\nA:", lambda d: "Yes")
    assert lmh_item("bbh", "bbh-1", bullets, {"input": "Do you return?\nOptions:\n- Yes\n- No"}, set())["choices"] == [
        "Yes",
        "No",
    ]


def test_item_question_is_the_prompt_without_the_shared_passage():
    docs = [
        {
            "passage": "Safaris are big in Africa; giraffes come close to cars ...",
            "question": f"Which animals {i}?",
            "mc_answer1": "Lions",
            "mc_answer2": "Giraffes",
            "correct_answer_num": "2",
        }
        for i in range(2)
    ]
    task = _Task(
        lambda d: f"P: {d['passage']}\nQ: {d['question']}\nA: {d['mc_answer1']}\nB: {d['mc_answer2']}\nAnswer:",
        lambda d: int(d["correct_answer_num"]) - 1,
        lambda d: ["A", "B"],
    )
    item = lmh_item("belebele", "belebele-test-0", task, docs[0], shared_values(docs))
    assert item["passage"] == docs[0]["passage"]
    assert "Safaris" not in item["question"] and "Which animals 0?" in item["question"]
    assert item["choices"] == ["Lions", "Giraffes"] and item["answer"] == "Giraffes"


def test_a_minimal_pair_item_is_its_correct_sentence():
    pair = _Task(lambda d: "", lambda d: 0, lambda d: [d["good"], d["bad"]])
    item = lmh_item("blimp", "blimp-0", pair, {"good": "The cats sleep.", "bad": "The cats sleeps."}, set())
    assert item["question"] == "The cats sleep."


def test_a_target_template_that_renders_key_names_reads_the_field_values():
    drop = _Task(
        lambda d: f"Passage: {d['passage']}\nQuestion: {d['question']}\nAnswer:", lambda d: ",".join(d["answer"])
    )
    doc = {
        "passage": "Brady threw two touchdowns.",
        "question": "Who threw the touchdowns?",
        "answer": {
            "number": "",
            "date": {"day": "", "month": "", "year": ""},
            "spans": ["Tom Brady"],
            "worker_id": "W",
            "hit_id": "H",
        },
    }
    assert lmh_item("drop", "drop-0", drop, doc, set())["answer"] == "Tom Brady"


def test_a_generic_stem_shared_by_several_items_is_not_their_passage():
    docs = [
        {
            "question": "Which of the following statements is correct?",
            "choices": [f"Claim {n} a", f"Claim {n} b"],
            "answer": 0,
        }
        for n in range(3)
    ]
    task = _Task(
        lambda d: d["question"] + "\nA. " + d["choices"][0] + "\nB. " + d["choices"][1] + "\nAnswer:",
        lambda d: 0,
        lambda d: ["A", "B"],
    )
    item = lmh_item("mmlu_x", "mmlu_x-0", task, docs[0], shared_values(docs))
    assert item["passage"] == ""
    assert item["question"].startswith("Which of the following statements is correct?")
