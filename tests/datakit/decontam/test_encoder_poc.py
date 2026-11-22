# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.datakit.decontam.encoder import rarest_key
from experiments.datakit.decontam.encoder.items import (
    _lmh_item,
    _option_fields,
    _parsed_options,
    _shared_values,
    _target_text,
    _without_labels,
)
from experiments.datakit.decontam.encoder.synthetic import generic_pairs
from experiments.datakit.decontam.encoder.text_keys import (
    WINDOW,
    hit_indices,
    readings,
    record_hit_indices,
    text_stream,
    window_hashes,
)
from experiments.datakit.decontam.encoder.train import missing_hint


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


def test_readings_stop_when_no_escape_is_left():
    assert readings("plain text") == ["plain text"]
    assert len(readings(json.dumps(json.dumps("a\nb")))) == 3


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


def test_option_texts_come_from_list_named_or_numbered_fields():
    labels = ["A", "B"]
    assert _option_fields({"choices": ["x", "y"]}, labels) == ["x", "y"]
    assert _option_fields({"A": "6", "B": "12"}, labels) == ["6", "12"]
    assert _option_fields({"option1": "red", "option2": "blue"}, labels) == ["red", "blue"]


def test_parsed_options_keep_the_first_complete_run_and_read_bullets():
    prompt = "Options:\n(A) deadpoel\n(B) deadfool\nA: Let's think step by step."
    assert _parsed_options(prompt) == (["A", "B"], ["deadpoel", "deadfool"])
    assert _parsed_options("Do you return?\nOptions:\n- Yes\n- No\nA:") == (["A", "B"], ["Yes", "No"])


def test_choices_lose_their_own_letters_and_structured_targets_become_text():
    assert _without_labels(["(A)2.5 m", "(B)5 m"]) == ["2.5 m", "5 m"]
    assert _without_labels(["Apple", "Banana"]) == ["Apple", "Banana"]
    assert _target_text({"number": "", "spans": ["Tom Brady"], "worker_id": "W1"}) == "Tom Brady"


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
    item = _lmh_item("belebele", "belebele-test-0", task, docs[0], _shared_values(docs))
    assert item["passage"] == docs[0]["passage"]
    assert "Safaris" not in item["question"] and "Which animals 0?" in item["question"]
    assert item["choices"] == ["Lions", "Giraffes"] and item["answer"] == "Giraffes"


def test_minimal_pair_and_letter_targets_resolve_to_text():
    pair = _Task(lambda d: "", lambda d: 0, lambda d: [d["good"], d["bad"]])
    item = _lmh_item("blimp", "blimp-0", pair, {"good": "The cats sleep.", "bad": "The cats sleeps."}, set())
    assert item["question"] == "The cats sleep."
    bbh = _Task(lambda d: "Q: " + d["input"] + "\nA:", lambda d: "(B)")
    item = _lmh_item("bbh", "bbh-0", bbh, {"input": "Pick one\nOptions:\n(A) left\n(B) right"}, set())
    assert item["choices"] == ["left", "right"] and item["answer"] == "right"


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
    assert _lmh_item("drop", "drop-0", drop, doc, set())["answer"] == "Tom Brady"


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


def test_missing_hint_lists_the_words_of_left_that_right_lacks():
    assert missing_hint("orbit of radius R takes 15 days", "orbit of radius R takes 17 days") == "\n[missing] 15"
    assert missing_hint("same words", "same words here") == "\n[missing] none"


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
