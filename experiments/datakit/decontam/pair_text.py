# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The decontamination model's input: an eval item's texts paired with a record excerpt.

The item table holds one row per eval record (``marin.datakit.decon.ITEM_SCHEMA``). A pair is
``(left, right)``: ``left`` is the item in marked sections followed by the words of each section that the
excerpt lacks (``pair_left``), and ``right`` is the record excerpt around the item's key. The model has two
outputs (``OUTPUTS``): the excerpt holds the item, and it holds the item with its answer.
"""

import re
import unicodedata
from typing import NamedTuple

import pandas as pd
from marin.datakit.decon_tables import parquet_files, read_frame

# The model's outputs, in order: the excerpt holds the item; it holds the item with its answer.
OUTPUTS = ("label_q", "label_qa")
# Most missing words that a hint lists.
MISSING_SHOWN = 32
LEFT_CHARS = 1500
RIGHT_CHARS = 6000
# Characters of a passage that `left` shows; 9 in 10 passages of the required evals are shorter.
PASSAGE_CHARS = 700

_SPACE = re.compile(r"\s+")
_TOKEN = re.compile(r"\w+")


class ItemText(NamedTuple):
    """The texts of one eval item that the model reads; `passage` is "" when it has none."""

    question: str
    choices: list[str]
    answer: str
    passage: str


def item_texts(items_dir: str) -> dict[str, ItemText]:
    """The texts of every item of the item table under *items_dir*, by item id. Pairs and labels store an
    item id and the model reads the item's texts from the table, so a table fix reaches every stored pair."""
    texts: dict[str, ItemText] = {}
    for path in parquet_files(items_dir):
        table = read_frame(path, ["id", "question", "choices", "answer", "passage"])
        for i, q, c, a, p in zip(
            table["id"], table["question"], table["choices"], table["answer"], table["passage"], strict=True
        ):
            texts[i] = ItemText(q, list(c), a or "", p or "")
    return texts


def text_normalize(text: str) -> str:
    """Light cleanup for both sides: Unicode compatibility forms folded, whitespace runs collapsed."""
    return _SPACE.sub(" ", unicodedata.normalize("NFKC", text)).strip()


def _missing(text: str, present: set[str]) -> str:
    missing = list(dict.fromkeys(token for token in _TOKEN.findall(text.lower()) if token not in present))
    return " ".join(missing[:MISSING_SHOWN]) if missing else "none"


def pair_left(question: str, choices: list[str], answer: str, passage: str, right: str) -> str:
    """The `left` text of a pair: the item in marked sections, then the words of each section that
    *right* lacks.

    The sections are ``[passage]`` (only for an item with a passage, cut to ``PASSAGE_CHARS``),
    ``[question]``, ``[choices]`` and ``[answer]``, normalized and cut to ``LEFT_CHARS``. The hints
    list the missing passage words (``[passage missing]``), question and choice words
    (``[missing]``) and answer words (``[answer missing]``) separately: an item with a passage is
    held only by a record with both its passage and its question, and an answer that the record
    does not show does not count against the question.
    """
    answer, passage = answer or "", text_normalize(passage or "")[:PASSAGE_CHARS]
    sections = [f"[passage] {passage}"] if passage else []
    sections.append(f"[question] {text_normalize(question)}")
    if choices:
        sections.append("[choices] " + " | ".join(map(text_normalize, choices)))
    if answer:
        sections.append(f"[answer] {text_normalize(answer)}")
    left = "\n".join(sections)[:LEFT_CHARS]
    present = set(_TOKEN.findall(right.lower()))
    hints = [f"[passage missing] {_missing(passage, present)}"] if passage else []
    hints.append(f"[missing] {_missing(' '.join([question, *choices]), present)}")
    hints.append(f"[answer missing] {_missing(answer, present) if answer else 'none'}")
    return left + "\n" + "\n".join(hints)


def with_pair_text(frame: pd.DataFrame, items: dict[str, ItemText]) -> pd.DataFrame:
    """*frame* (columns ``item`` and ``right``) with `right` normalized and cut to ``RIGHT_CHARS``, and
    `left` built by ``pair_left`` from the texts of each pair's item in *items* (see ``item_texts``)."""
    frame = frame.copy()
    frame["right"] = frame["right"].map(text_normalize).str.slice(0, RIGHT_CHARS)
    frame["left"] = [
        pair_left(items[item].question, items[item].choices, items[item].answer, items[item].passage, right)
        for item, right in zip(frame["item"], frame["right"], strict=True)
    ]
    return frame
