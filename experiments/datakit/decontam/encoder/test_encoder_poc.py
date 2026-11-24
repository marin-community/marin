# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import random

import pandas as pd
from marin.datakit.decon_keys import (
    WINDOW,
    item_stream,
    window_hashes,
)
from marin.datakit.decon_pairs import ItemText

from experiments.datakit.decontam.encoder.planted import holds_key, positive_forms
from experiments.datakit.decontam.encoder.synthetic import context_pairs, generic_pairs


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
