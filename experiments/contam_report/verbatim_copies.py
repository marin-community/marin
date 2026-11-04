# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check whether training documents hold an eval item's question and answer verbatim.

Input: ITEMS_JSON from `verbatim_items.py`, and REQUESTS_JSON, a list of
``{"path", "id", "eval_id"}`` (from `verbatim_requests.py copies`). For each
request, "verbatim" means the text occurs in the document after collapsing
whitespace, also after undoing JSON string escapes, since many copies sit inside
JSON lines.

* ``free`` items: ``q_found`` is the question verbatim; ``a_found`` is any answer
  variant verbatim anywhere in the document; ``a_ngram_coverage`` is the best
  variant's share of whitespace 13-grams in the document.
* ``choice`` items: ``q_found`` is the stem verbatim; ``choices_found`` counts
  the distinctive choices (at least 3 words) in the window after it;
  ``a_found`` means an answer marker after the stem names the correct choice by
  letter or text (see `mmlu_copies.classify`).
* ``near`` items: ``q_found`` is the question verbatim; ``a_found`` means an
  answer occurs in the ``WINDOW`` characters after it. It must follow an answer
  marker when it is at most ``SHORT_ANSWER_CHARS`` long or when a wrong option
  (``distractors``) is also in the window, as in a copy that lists every option.
* ``question`` items have no reference answer; only ``q_found`` is set.

Writes one Parquet file per input file under OUT_PREFIX.

Usage: verbatim_copies.py ITEMS_JSON REQUESTS_JSON OUT_PREFIX [--max-workers N] [--worker-ram R]
"""

import argparse
import hashlib
import json
import re
from collections import defaultdict

import pyarrow as pa
from fray.types import ResourceConfig
from rigging.filesystem.storage_path import StoragePath
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.input_file import InputFileSpec
from zephyr.readers import load_file
from zephyr.writers import write_parquet_file

from experiments.contam_report.humaneval_copies import collapse, found_ngrams, ngrams, unescape
from experiments.contam_report.mmlu_copies import WINDOW, classify_flat

SHORT_ANSWER_CHARS = 3

SCHEMA = pa.schema(
    [
        ("eval_id", pa.string()),
        ("path", pa.string()),
        ("id", pa.string()),
        ("q_found", pa.bool_()),
        ("q_words", pa.int64()),
        ("choices_total", pa.int64()),
        ("choices_found", pa.int64()),
        ("a_found", pa.bool_()),
        ("a_ngram_coverage", pa.float64()),
    ]
)


class DocViews:
    """A document's whitespace-collapsed views, raw and JSON-unescaped, built once per document."""

    def __init__(self, doc: str):
        self.flats = (collapse(doc), collapse(unescape(doc)))
        self._tokens: list[list[str]] | None = None

    @property
    def tokens(self) -> list[list[str]]:
        if self._tokens is None:
            self._tokens = [flat.split() for flat in self.flats]
        return self._tokens


def check_free(doc: DocViews, question: str, answers: list[str]) -> dict:
    q_found = any(collapse(question) in flat for flat in doc.flats)
    a_found = any(collapse(answer) in flat for answer in answers for flat in doc.flats)
    return {
        "q_found": q_found,
        "choices_total": None,
        "choices_found": None,
        "a_found": a_found,
        "a_ngram_coverage": None,
    }


def add_answer_coverage(doc: DocViews, rows: list[dict], items: dict) -> None:
    """Fill ``a_ngram_coverage`` for free items with a verbatim question but no verbatim answer.

    The n-gram fallback only matters for those rows, and one scan of the document serves all
    of them: a test dump can pair one document with over a thousand items.
    """
    pending = [r for r in rows if items[r["eval_id"]]["kind"] == "free" and r["q_found"] and not r["a_found"]]
    if not pending:
        return
    answer_grams = {r["eval_id"]: [ngrams(answer.split()) for answer in items[r["eval_id"]]["answers"]] for r in pending}
    wanted = set().union(*(g for grams in answer_grams.values() for g in grams))
    found = set().union(*(found_ngrams(wanted, tokens) for tokens in doc.tokens)) if wanted else set()
    for r in pending:
        covered = [len(g & found) / len(g) for g in answer_grams[r["eval_id"]] if g]
        r["a_ngram_coverage"] = max(covered, default=None)


def answer_marked(answer: str) -> re.Pattern:
    return re.compile(rf"(?:answer|correct|ans)\b[^\n]{{0,40}}{re.escape(answer)}", re.IGNORECASE)


def check_near(doc: DocViews, question: str, answers: list[str], distractors: list[str]) -> dict:
    q_flat = collapse(question)
    a_flats = [collapse(a) for a in answers if a.strip()]
    d_flats = [collapse(d) for d in distractors if d.strip()]
    q_found = a_found = False
    for flat in doc.flats:
        pos = flat.find(q_flat)
        if pos < 0:
            continue
        q_found = True
        window = flat[pos + len(q_flat) : pos + len(q_flat) + WINDOW]
        listed = any(d in window for d in d_flats)
        for a in a_flats:
            needs_marker = listed or len(a) <= SHORT_ANSWER_CHARS
            a_found = a_found or bool(answer_marked(a).search(window) if needs_marker else a in window)
    return {
        "q_found": q_found,
        "choices_total": None,
        "choices_found": None,
        "a_found": a_found,
        "a_ngram_coverage": None,
    }


def check(doc: DocViews, item: dict) -> dict:
    words = len(item["question"].split())
    if item["kind"] == "near":
        return {"q_words": words, **check_near(doc, item["question"], item["answers"], item.get("distractors", []))}
    if item["kind"] == "question":
        q_found = any(collapse(item["question"]) in flat for flat in doc.flats)
        return {
            "q_found": q_found,
            "q_words": words,
            "choices_total": None,
            "choices_found": None,
            "a_found": False,
            "a_ngram_coverage": None,
        }
    if item["kind"] == "free":
        return {"q_words": words, **check_free(doc, item["question"], item["answers"])}
    result = classify_flat(doc.flats, item["question"], item["choices"], item["answer"])
    return {
        "q_found": result["stem_found"],
        "q_words": words,
        "choices_total": result["distinctive_choices"],
        "choices_found": result["distinctive_choices_found"],
        "a_found": result["answer_marked"],
        "a_ngram_coverage": None,
    }


def make_checker(items_path: str, out_prefix: str):
    def run(groups, shard):
        with StoragePath(items_path).open("r") as f:
            items = json.load(f)
        for group in groups:
            path, wanted = group["path"], group["wanted"]
            out_path = f"{out_prefix}/{hashlib.sha1(path.encode()).hexdigest()[:16]}.parquet"
            if StoragePath(out_path).exists():
                continue

            def rows(path: str, wanted: dict):
                for record in load_file(InputFileSpec(path=path, columns=["id", "text"])):
                    doc_id = str(record["id"])
                    eval_ids = wanted.get(doc_id)
                    if not eval_ids:
                        continue
                    doc = DocViews(str(record["text"]))
                    doc_rows = [
                        {"eval_id": eval_id, "path": path, "id": doc_id, **check(doc, items[eval_id])}
                        for eval_id in eval_ids
                    ]
                    add_answer_coverage(doc, doc_rows, items)
                    yield from doc_rows

            yield write_parquet_file(rows(path, wanted), output_path=out_path, schema=SCHEMA)

    return run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("items")
    parser.add_argument("requests")
    parser.add_argument("out_prefix")
    parser.add_argument("--max-workers", type=int, default=100)
    parser.add_argument("--worker-ram", default="16g")
    args = parser.parse_args()
    with StoragePath(args.requests).open("r") as f:
        requests = json.load(f)
    by_path: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for r in requests:
        by_path[r["path"]][r["id"]].append(r["eval_id"])
    groups = [{"path": p, "wanted": dict(w)} for p, w in sorted(by_path.items())]
    resources = ResourceConfig(cpu=1, ram=args.worker_ram)
    ctx = ZephyrContext(name="contam-verbatim-copies", resources=resources, max_workers=args.max_workers)
    ctx.execute(Dataset.from_list(groups).map_shard(make_checker(args.items, args.out_prefix)))


if __name__ == "__main__":
    main()
