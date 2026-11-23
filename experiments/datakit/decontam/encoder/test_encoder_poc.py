# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from marin.datakit.decon_keys import (
    WINDOW,
    KeyStream,
    Windows,
    item_stream,
    keys_table,
    text_stream,
    window_hashes,
)

from experiments.datakit.decontam.encoder import rarest_key
from experiments.datakit.decontam.encoder.items import lmh_item, shared_values
from experiments.datakit.decontam.encoder.planted import holds_key, negative_forms, positive_forms
from experiments.datakit.decontam.encoder.synthetic import context_pairs, generic_pairs
from experiments.datakit.decontam.encoder.tables import ItemText
from experiments.datakit.decontam.encoder.train import pair_left


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
        tmp_path / "windows.npz",
        hashes=hashes,
        owners=owners,
        offsets=offsets,
        ids=np.array(["a", "b"], dtype=object),
        kinds=np.array(["question", "question"], dtype=object),
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


def test_synthetic_question_without_its_answer_counts_for_the_question_output_only():
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
    labels = rows.groupby("kind")[["label_q", "label_qa"]].max()
    # The same question with other choices, or the question alone, holds the question but not the answer.
    assert labels.loc["same_question"].tolist() == [1, 0]
    assert labels.loc["question_only"].tolist() == [1, 0]
    assert rows.loc[rows["kind"].str.startswith("variant"), ["label_q", "label_qa"]].eq(0).all().all()


def test_missing_answer_words_stay_out_of_the_question_hint():
    question = "def common(l1: list, l2: list):\n    Return sorted unique common elements."
    right = "```py\ndef common(l1: list, l2: list):\n    Return sorted unique common elements.```"
    left = pair_left(question, [], "def check(candidate): assert candidate", "", right)
    assert left.endswith("\n[missing] none\n[answer missing] check candidate assert")
    stem_only = pair_left("What is the GDP deflator?", ["A price index", "A tax"], "", "", "What is the GDP deflator?")
    assert "[choices] A price index | A tax" in stem_only and "[missing] a price index tax" in stem_only


def _items(count: int) -> list[dict]:
    return [
        {
            "id": f"i{n}",
            "eval": "e",
            "question": f"Which planet number {n} has the most moons orbiting it today?",
            "choices": ["Mars", "Saturn"],
            "answer": "Saturn",
            "passage": "",
        }
        for n in range(count)
    ]


def test_a_copy_that_only_lists_the_choices_does_not_show_the_answer():
    rows = pd.DataFrame(list(generic_pairs(_items(60), ["host text " * 80], random.Random(0))))
    copies = rows[rows["kind"] == "copy"]
    marked = copies["right"].str.contains('Answer:|"answer": "Saturn', case=False)
    assert (copies["label_qa"] == marked.astype(int)).all()
    # Every copy lists "Saturn" among its choices, yet only the marked ones show the answer.
    assert copies["label_qa"].sum() < len(copies)


def test_context_variants_skip_candidates_that_already_hold_the_item():
    items = _items(2)
    candidates = pd.DataFrame(
        [
            {
                **items[0],
                "item": "i0",
                "right": "notes " * 40 + items[0]["question"] + " more " * 40,
                "question_in_record": True,
            },
            {**items[1], "item": "i1", "right": "unrelated words " * 60, "question_in_record": False},
        ]
    )
    texts = {
        f"i{n}": ItemText(item["question"], list(item["choices"]), item["answer"], "") for n, item in enumerate(items)
    }
    rows = pd.DataFrame(list(context_pairs(candidates, texts, random.Random(0))))
    variants = rows[rows["kind"].str.startswith("ctx_variant")]
    assert set(variants["item"]) == {"i1"}
    assert set(rows.loc[rows["kind"] == "ctx_copy", "item"]) == {"i0", "i1"}


def test_every_planted_form_of_an_item_holds_its_key():
    item = {
        "question": "Which gas do plants take in from the air during photosynthesis?",
        "choices": ["Oxygen", "Carbon dioxide", "Nitrogen", "Helium"],
        "answer": "Carbon dioxide",
    }
    stream = item_stream(item["question"], item["choices"])
    key = int(window_hashes(stream)[len(stream) // 2 - WINDOW])
    host = "Some unrelated record text. " * 120
    for form, text in positive_forms(item, "Argon", random.Random(0)).items():
        assert holds_key(f"{host}\n{text}\n{host}", {key}), form
    # Negatives share the key; stage 2, not stage 1, has to reject them.
    assert set(negative_forms(item, "Argon", host)) == {"changed_choice", "choices_far", "missing_choice"}


def test_an_item_with_a_passage_shows_the_passage_and_its_missing_words_apart():
    passage = "The Bears beat the Lions 24-10 in Chicago."
    right = "Question: How many yards was the longest field goal? It was 52 yards in another game."
    left = pair_left("How many yards was the longest field goal?", [], "52", passage, right)
    assert left.startswith("[passage] The Bears beat the Lions")
    assert "\n[question] How many yards" in left
    # "the" and "in" occur in the record, so only the passage's own words are missing.
    assert "\n[passage missing] bears beat lions 24 10 chicago\n[missing] none\n" in left


def test_a_passage_or_its_question_alone_is_not_a_copy_of_an_item_with_a_passage():
    passage = "Ann has three red apples and two green pears in her basket today."
    items = [
        {"id": f"p{n}", "eval": "e", "question": q, "choices": [], "answer": a, "passage": passage}
        for n, (q, a) in enumerate(
            [("How many apples does Ann have?", "three"), ("How many pears does Ann have?", "two")]
        )
    ]
    rows = pd.DataFrame(list(generic_pairs(items, ["host text " * 80], random.Random(0))))
    label = rows.groupby("kind")["label_q"].max()
    assert label["copy"] == 1 and label["question_only"] == 0 and label["passage_only"] == 0
    assert label["passage_other_question"] == 0 and label["item_without_passage"] == 0
    assert rows.loc[rows["kind"] == "copy", "right"].str.contains("Ann has three red apples").all()


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


def test_a_shared_tie_keys_each_item_off_the_question_stem_that_other_items_share():
    stem = "Which of the following propositions is an immediate consequence of the given premises?"
    questions = [
        f"{stem} {premise}" for premise in ("Henry rides a red bicycle daily", "Mara bakes seven loaves weekly")
    ]
    streams = [text_stream(q) for q in questions]
    hashes, owners, offsets = rarest_key.candidate_windows(streams)
    ids = np.array(["formal-0", "formal-1"], dtype=object)
    windows = Windows(hashes, owners, offsets, ids, np.array([KeyStream.QUESTION] * 2, dtype=object))
    # No window occurs in the reference sample, so the tie order decides the keys.
    counts = np.zeros(len(hashes), dtype=np.int64)
    stem_windows = set(window_hashes(text_stream(stem)).tolist())
    earliest = keys_table(windows, counts).column("key1_hash").to_pylist()
    shared = keys_table(windows, counts, tie=rarest_key.tie_values(windows, rarest_key.Tie.SHARED))
    assert all(key in stem_windows for key in earliest)
    assert not any(key in stem_windows for key in shared.column("key1_hash").to_pylist())
