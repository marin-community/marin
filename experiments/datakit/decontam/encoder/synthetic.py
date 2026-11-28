# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Synthetic (left, right) pairs for the classifier, with exact labels for both outputs.

``generic``: eval items written into host records in one of several layouts (lettered,
numbered, Markdown, JSON, upper case, question only, a cut-off question), and negatives: the
host alone, another item, another item with the same question, and variants of the item (a
changed number, word or operator, or an added constraint).

``context``: the item, or a variant of it, written into the text of a real stage-1 candidate of
the same item, so positives and negatives sit in text that shares the item's rare window and topic.

Labels: ``label_q`` is 1 when the right text holds the item's question; ``label_qa`` is 1 when it
also shows the item's answer.
"""

import argparse
import io
import json
import logging
import random
import re
from collections.abc import Iterator
from typing import Any, NamedTuple

import pandas as pd
import pyarrow.parquet as pq
from marin.datakit.decon_keys import text_stream
from marin.datakit.decon_tables import parquet_files, read_frame, write_frame
from rigging.filesystem.storage_path import StoragePath
from rigging.log_setup import configure_logging

from experiments.datakit.decontam.encoder.labels import SILVER_OVERLAP, word_overlap
from experiments.datakit.decontam.pair_text import ItemText, item_texts

logger = logging.getLogger(__name__)
EVALS_PER_FAMILY = 6
MAX_QUESTION = 3000
HOST_FILES = 12
HOST_TEXTS = 400
MIN_HOST = 400
LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 4
NUMBER = re.compile(r"\d+")
WORD = re.compile(r"[A-Za-z]{5,}")
OPERATORS = (
    ("+", "-"),
    ("-", "+"),
    ("<", ">"),
    (">", "<"),
    ("increase", "decrease"),
    ("more", "less"),
    ("maximum", "minimum"),
)


class Layout(NamedTuple):
    text: str
    shows_answer: bool
    """True when the text marks the answer ("Answer: ...", an answer field), not when it only lists the choices."""


def layout(item: dict[str, Any], rng: random.Random) -> Layout:
    """The item as a copy could write it: one of several layouts of question, choices and answer."""
    question, choices, answer = item["question"], list(item["choices"]), item["answer"]
    style = rng.choice(["lettered", "parens", "numbered", "plain", "markdown", "json", "upper", "qa"])
    if style == "json":
        text = json.dumps({"question": question, "choices": choices, "answer": answer}, ensure_ascii=rng.random() < 0.5)
        return Layout(text, bool(answer))
    shows_answer = False
    if style == "qa" or not choices:
        shows_answer = bool(answer) and rng.random() < 0.7
        text = f"Question: {question}\nAnswer: {answer}" if shows_answer else question
    elif style == "lettered":
        text = question + "\n" + "\n".join(f"{LABELS[i]}. {c}" for i, c in enumerate(choices)) + f"\nAnswer: {answer}"
        shows_answer = bool(answer)
    elif style == "parens":
        text = question + " " + " ".join(f"({LABELS[i]}) {c}" for i, c in enumerate(choices))
    elif style == "numbered":
        text = question + "\n" + "\n".join(f"{i + 1}) {c}" for i, c in enumerate(choices))
    elif style == "markdown":
        text = f"**{question}**\n" + "\n".join(f"- {c}" for c in choices)
    else:
        text = question + "\n" + "\n".join(choices)
    if style == "upper":
        text = text.upper()
    if item.get("passage"):
        # An item with a passage is the passage and the question together.
        text = item["passage"] + "\n\n" + text
    return Layout(text, shows_answer)


def question_only(item: dict[str, Any], rng: random.Random) -> str:
    style = rng.choice(["plain", "prefixed", "markdown", "upper", "json"])
    question = item["question"]
    if style == "prefixed":
        return f"{rng.choice(['Q:', 'Question:', 'Problem:', '#'])} {question}"
    if style == "markdown":
        return f"**{question}**"
    if style == "upper":
        return question.upper()
    if style == "json":
        return json.dumps({"prompt": question}, ensure_ascii=rng.random() < 0.5)
    return question


def truncated(item: dict[str, Any], rng: random.Random) -> str:
    """The question with a fifth of its words cut at the start or the end, as an excerpt edge does."""
    words = item["question"].split()
    cut = max(1, len(words) // 5)
    return " ".join(words[cut:]) if rng.random() < 0.5 else " ".join(words[:-cut])


def variant(item: dict[str, Any], others: list[dict[str, Any]], rng: random.Random) -> tuple[str, str] | None:
    """Another problem made from this one: a changed number, word or operator, or an added constraint."""
    question = item["question"]
    kinds = ["number", "word", "operator", "constraint"]
    rng.shuffle(kinds)
    for kind in kinds:
        if kind == "number" and (numbers := list(NUMBER.finditer(question))):
            match = rng.choice(numbers)
            changed = str(int(match.group()) + rng.choice([1, 2, 3, 7, 10]))
            return kind, question[: match.start()] + changed + question[match.end() :]
        if kind == "word":
            words = list(WORD.finditer(question))
            pool = [word for other in others for word in WORD.findall(other["question"])]
            if words and pool:
                match = rng.choice(words)
                return kind, question[: match.start()] + rng.choice(pool) + question[match.end() :]
        if kind == "operator" and (found := [(a, b) for a, b in OPERATORS if a in question]):
            a, b = rng.choice(found)
            index = rng.choice([m.start() for m in re.finditer(re.escape(a), question)])
            return kind, question[:index] + b + question[index + len(a) :]
        if kind == "constraint" and others:
            sentence = re.split(r"(?<=[.?!])\s+", rng.choice(others)["question"])[0][:200]
            return kind, question.rstrip() + " Additionally, " + sentence[:1].lower() + sentence[1:]
    return None


def _row(item: dict[str, Any], right: str, kind: str, label_q: int, label_qa: int) -> dict[str, Any]:
    return {
        "eval": item["eval"],
        "item": item.get("id", ""),
        "question": item["question"],
        "choices": list(item["choices"]),
        "answer": item["answer"] or "",
        "right": right,
        "kind": kind,
        "label_q": label_q,
        "label_qa": label_qa,
    }


# A sibling negative is the other item of the same eval whose question shares at least this share of words with
# the item's question (another instance of the same template), with a different question stream.
SIBLING_OVERLAP = 0.6


def _siblings(items: list[dict[str, Any]]) -> dict[int, int]:
    """For each item, the item of the same eval with the most question words in common, when that share is at
    least ``SIBLING_OVERLAP`` and the two questions differ in their streams."""
    by_eval: dict[str, list[int]] = {}
    for number, item in enumerate(items):
        by_eval.setdefault(item["eval"], []).append(number)
    streams = [text_stream(item["question"]) for item in items]
    found = {}
    for numbers in by_eval.values():
        for number in numbers:
            best = max(
                (
                    (word_overlap(items[number]["question"], items[other]["question"]), other)
                    for other in numbers
                    if streams[other] != streams[number]
                ),
                default=(0.0, -1),
            )
            if best[0] >= SIBLING_OVERLAP:
                found[number] = best[1]
    return found


def generic_pairs(items: list[dict[str, Any]], hosts: list[str], rng: random.Random) -> Iterator[dict[str, Any]]:
    by_question: dict[str, list[int]] = {}
    by_passage: dict[str, list[int]] = {}
    for number, item in enumerate(items):
        by_question.setdefault(item["question"], []).append(number)
        if item.get("passage"):
            by_passage.setdefault(item["passage"], []).append(number)
    sibling_of = _siblings(items)

    def placed(inserted: str) -> str:
        host = rng.choice(hosts)
        start = rng.randrange(0, max(1, len(host) - 1200))
        before = host[start : start + rng.randrange(0, 600)]
        after = host[start + len(before) : start + len(before) + rng.randrange(0, 600)]
        return before + "\n" + inserted + "\n" + after

    for number, item in enumerate(items):
        if number in sibling_of:
            # Another instance of the same template (names, words or table rows changed) is not this item.
            yield _row(item, placed(layout(items[sibling_of[number]], rng).text), "sibling", 0, 0)
        inserted = layout(item, rng)
        yield _row(item, placed(inserted.text), "copy", 1, int(inserted.shows_answer))
        passage = item.get("passage") or ""
        # Without its passage, the question of an item with a passage is not a copy of the item.
        yield _row(item, placed(question_only(item, rng)), "question_only", 0 if passage else 1, 0)
        if len(item["question"].split()) >= 15 and rng.random() < 0.5:
            yield _row(item, placed((passage + "\n\n" if passage else "") + truncated(item, rng)), "truncated", 1, 0)
        if passage:
            yield _row(item, placed(passage), "passage_only", 0, 0)
            # The question and its choices without the passage, laid out as a copy would be.
            yield _row(item, placed(layout({**item, "passage": ""}, rng).text), "item_without_passage", 0, 0)
            siblings = [items[i] for i in by_passage[passage] if items[i]["question"] != item["question"]]
            if siblings:
                # The same passage with another question of the test holds another item, not this one.
                yield _row(item, placed(layout(rng.choice(siblings), rng).text), "passage_other_question", 0, 0)
        others = [items[rng.randrange(len(items))] for _ in range(3)]
        if found := variant(item, others, rng):
            yield _row(item, placed(layout({**item, "question": found[1]}, rng).text), f"variant_{found[0]}", 0, 0)
        kind = rng.choice(["none", "other", "same_question"])
        if kind == "same_question" and len(by_question[item["question"]]) > 1:
            kind = "other"
            same = [items[i] for i in by_question[item["question"]] if items[i]["answer"] != item["answer"]]
            if same:
                # The same question with other choices: the question is there, this item is not.
                yield _row(item, placed(layout(rng.choice(same), rng).text), "same_question", 1, 0)
                continue
        if kind == "other":
            yield _row(item, placed(layout(rng.choice(items), rng).text), "other", 0, 0)
        else:
            yield _row(item, placed(""), "none", 0, 0)


def context_pairs(candidates: pd.DataFrame, texts: dict[str, ItemText], rng: random.Random) -> Iterator[dict[str, Any]]:
    items = [
        {"id": i, "eval": e, **texts[i]._asdict()}
        for i, e in zip(
            candidates["item"],
            candidates["eval"],
            strict=True,
        )
    ]
    holds_item = [
        bool(in_record) or word_overlap(item["question"], right) >= SILVER_OVERLAP
        for item, right, in_record in zip(items, candidates["right"], candidates["question_in_record"], strict=True)
    ]
    for number, (item, right) in enumerate(zip(items, candidates["right"], strict=True)):
        cut = rng.randrange(len(right) // 3, max(len(right) // 3 + 1, 2 * len(right) // 3))
        before, after = right[:cut], right[cut:]
        inserted = layout(item, rng)
        yield _row(item, before + "\n" + inserted.text + "\n" + after, "ctx_copy", 1, int(inserted.shows_answer))
        if holds_item[number]:
            # The candidate already holds the item, so a variant written into it is not a negative.
            continue
        others = [items[rng.randrange(len(items))] for _ in range(3)]
        if found := variant(item, others, rng):
            changed = layout({**item, "question": found[1]}, rng).text
            yield _row(item, before + "\n" + changed + "\n" + after, f"ctx_variant_{found[0]}", 0, 0)


def main() -> None:
    configure_logging(logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["generic", "context"])
    parser.add_argument("--items", required=True, help="item table directory (items/*.parquet)")
    parser.add_argument("--hosts", help="generic: glob of parquet files with a text column; the first 12 are used")
    parser.add_argument("--per-eval", type=int, default=60, help="generic: items sampled per eval")
    parser.add_argument("--candidates", help="context: scored stage-1 pairs (parquet)")
    parser.add_argument(
        "--exclude-items", nargs="*", default=[], help="parquet files with an item column: items kept out"
    )
    parser.add_argument("--size", type=int, default=12000, help="context: candidates used")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    excluded = {item for path in args.exclude_items for item in read_frame(path, ["item"])["item"]}
    if args.mode == "generic":
        paths = parquet_files(args.items)
        families: dict[str, list[str]] = {}
        for path in paths:
            families.setdefault(StoragePath(path).name.split("_")[0], []).append(path)
        chosen = [path for group in families.values() for path in rng.sample(group, min(len(group), EVALS_PER_FAMILY))]
        frames = []
        for path in chosen:
            frame = read_frame(path)
            frame = frame[(frame["reason"] == "") & frame["question"].str.len().between(1, MAX_QUESTION - 1)]
            frames.append(frame.sample(min(len(frame), args.per_eval), random_state=args.seed))
        items = pd.concat(frames, ignore_index=True)
        items = items[~items["id"].isin(excluded)].to_dict("records")
        hosts = []
        for path in sorted(str(path) for path in StoragePath(args.hosts).glob())[:HOST_FILES]:
            texts = pq.ParquetFile(io.BytesIO(StoragePath(path).read_bytes())).read_row_group(0, columns=["text"])
            texts = texts.column("text")
            hosts += [text for text in texts.to_pylist()[:HOST_TEXTS] if text and len(text) > MIN_HOST]
        rows = list(generic_pairs(items, hosts, rng))
    else:
        candidates = read_frame(args.candidates)
        candidates = candidates[~candidates["item"].isin(excluded)]
        sample = candidates.sample(min(args.size, len(candidates)), random_state=args.seed)
        rows = list(context_pairs(sample, item_texts(args.items), rng))
    frame = pd.DataFrame(rows)
    evals = sorted(frame["eval"].unique())
    held_out = set(random.Random(args.seed).sample(evals, len(evals) // 5)) if args.mode == "generic" else set()
    frame["split"] = ["test" if e in held_out else "train" for e in frame["eval"]]
    frame["source"] = "synthetic" if args.mode == "generic" else "synthetic_ctx"
    write_frame(frame, args.output)
    logger.info(
        "%s: %d pairs; %s", args.mode, len(frame), frame.groupby(["kind", "label_q", "label_qa"]).size().to_dict()
    )


if __name__ == "__main__":
    main()
