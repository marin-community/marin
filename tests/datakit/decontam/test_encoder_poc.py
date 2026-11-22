# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.datakit.decontam.encoder import rarest_key
from experiments.datakit.decontam.encoder.items import lmh_item, shared_values
from experiments.datakit.decontam.encoder.synthetic import generic_pairs
from experiments.datakit.decontam.encoder.text_keys import (
    WINDOW,
    hit_indices,
    record_hit_indices,
    text_stream,
    window_hashes,
)
from experiments.datakit.decontam.encoder.train import pair_left


def test_text_stream_ignores_case_spacing_markup_and_latex_commands():
    assert text_stream("**What is $\\frac{1}{3}$ of 12?**") == text_stream("what is 1/3 of 12")
    # Full-width letters and digits fold to their ASCII forms.
    assert text_stream("\uff31\uff11: \uff21") == "q1a"


def test_hit_indices_find_a_key_inside_other_text():
    key = text_stream("the stylomastoid foramen will cause ipsilateral")[:WINDOW]
    keys = np.sort(window_hashes(key))
    assert len(hit_indices(text_stream("Notes. The stylomastoid foramen will cause ipsilateral palsy."), keys)) == 1
    assert len(hit_indices(text_stream("An unrelated sentence about the weather."), keys)) == 0


def test_a_copy_inside_json_once_or_twice_holds_every_window_of_the_item():
    item = (
        "from typing import List\n\n\ndef has_close_elements(numbers: List[float]) -> bool:\n"
        '    """ Check $\\theta \\neq \\nu$'
    )
    keys = np.sort(window_hashes(text_stream(item)))
    once = json.dumps({"prompt": item})
    twice = json.dumps({"row": once}, ensure_ascii=True)
    assert len(hit_indices(text_stream(once), keys)) < len(keys)
    for record in (once, twice):
        assert len(record_hit_indices("Data: " + record, keys)) == len(keys)


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


def test_key_choice_takes_the_rarest_window_and_a_second_one_that_does_not_overlap(tmp_path: Path):
    streams = [
        text_stream("Which of the following best describes the fallacy of equivocation in arguments?"),
        text_stream("short"),
    ]
    hashes, owners, offsets = rarest_key.candidate_windows(streams)
    np.savez(
        tmp_path / "windows.npz", hashes=hashes, owners=owners, offsets=offsets, ids=np.array(["a", "b"], dtype=object)
    )
    (tmp_path / "counts").mkdir()
    # Later windows are rarer in the reference sample.
    counts = np.arange(len(hashes), 0, -1).astype(np.uint32)
    np.savez(tmp_path / "counts" / "part-00000.npz", hashes=hashes, counts=counts)
    rarest_key.choose(str(tmp_path))
    keys = pd.read_parquet(tmp_path / "keys.parquet").set_index("id")
    assert keys.loc["a", "key1_offset"] == offsets[owners == 0].max()
    assert keys.loc["a", "key2_hash"] != 0 and keys.loc["a", "key2_hash"] != keys.loc["a", "key1_hash"]
    assert keys.loc["b", "key1_hash"] == 0


def test_synthetic_labels_separate_the_question_from_the_item():
    items = [
        {
            "id": f"i{n}",
            "eval": "e",
            "question": "Which planet has the most moons by count today?",
            "choices": [["Mars", "Saturn"], ["Venus", "Jupiter"]][n % 2],
            "answer": ["Saturn", "Jupiter"][n % 2],
            "passage": "",
        }
        for n in range(40)
    ]
    rows = pd.DataFrame(list(generic_pairs(items, ["host text " * 80], random.Random(0))))
    labels = rows.groupby("kind")[["label", "label_q", "label_qa"]].max()
    # The same question with other choices holds the question but not this item.
    assert labels.loc["same_question"].tolist() == [0, 1, 0]
    assert labels.loc["question_only"].tolist() == [1, 1, 0]
    assert rows.loc[rows["kind"].str.startswith("variant"), ["label", "label_q", "label_qa"]].eq(0).all().all()


def test_missing_answer_words_stay_out_of_the_question_hint():
    question = "def common(l1: list, l2: list):\n    Return sorted unique common elements."
    right = "```py\ndef common(l1: list, l2: list):\n    Return sorted unique common elements.```"
    left = pair_left(question, [], "def check(candidate): assert candidate", right)
    assert left.endswith("\n[missing] none\n[answer missing] check candidate assert")
    stem_only = pair_left("What is the GDP deflator?", ["A price index", "A tax"], "", "What is the GDP deflator?")
    assert "[choices] A price index | A tax" in stem_only and "[missing] a price index tax" in stem_only
