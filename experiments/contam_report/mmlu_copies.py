# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Classify kept documents that hold at least half of an MMLU item.

Input: a JSON list of ``{"path", "id", "eval_id", "stem", "has_passage", "choices",
"answer"}``. ``stem`` is the question without the quoted source passage that the
history subjects start with, since a document can quote that public passage without
copying the question. For each request it records whether the stem occurs in the
document (whitespace collapsed, also after undoing JSON escapes), how many of the
distinctive choices (at least 3 words) occur, and whether an answer marker after the
stem names the correct choice by letter or text. Writes one Parquet file per input
file under OUT_PREFIX.

Usage: mmlu_copies.py REQUESTS_JSON OUT_PREFIX [--max-workers N] [--worker-ram R]
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

from experiments.contam_report.humaneval_copies import collapse, unescape

LETTERS = "ABCDEFGHIJ"
# Characters after the stem searched for the choices and an answer marker.
WINDOW = 3000
DISTINCTIVE_WORDS = 3
ANSWER_MARKER = re.compile(r"(?:answer|correct|ans)\b[^A-Za-z0-9\n]{0,12}(?:is\s*)?[\(\[]?([A-J])\b", re.IGNORECASE)
SCHEMA = pa.schema(
    [
        ("eval_id", pa.string()),
        ("path", pa.string()),
        ("id", pa.string()),
        ("has_passage", pa.bool_()),
        ("stem_found", pa.bool_()),
        ("distinctive_choices", pa.int64()),
        ("distinctive_choices_found", pa.int64()),
        ("answer_marked", pa.bool_()),
    ]
)


def classify(doc: str, stem: str, choices: list[str], answer: int) -> dict:
    return classify_flat((collapse(doc), collapse(unescape(doc))), stem, choices, answer)


def classify_flat(flats: tuple[str, ...], stem: str, choices: list[str], answer: int) -> dict:
    """Classify a document given as whitespace-collapsed views (raw and JSON-unescaped)."""
    stem_flat = collapse(stem)
    distinctive = [collapse(c).rstrip(".,;:") for c in choices if len(c.split()) >= DISTINCTIVE_WORDS]
    best = {"stem_found": False, "distinctive_choices": len(distinctive), "distinctive_choices_found": 0}
    marked = False
    for flat in flats:
        pos = flat.find(stem_flat)
        if pos < 0:
            continue
        window = flat[pos + len(stem_flat) : pos + len(stem_flat) + WINDOW]
        best["stem_found"] = True
        best["distinctive_choices_found"] = max(best["distinctive_choices_found"], sum(c in window for c in distinctive))
        letter = ANSWER_MARKER.search(window)
        correct = collapse(choices[answer]).rstrip(".,;:")
        text_marked = re.search(rf"(?:answer|correct)\b[^\n]{{0,40}}{re.escape(correct)}", window, re.IGNORECASE)
        marked = marked or (letter is not None and letter.group(1).upper() == LETTERS[answer]) or text_marked is not None
    return {**best, "answer_marked": marked}


def make_checker(out_prefix: str):
    def run(groups, shard):
        for group in groups:
            path, wanted = group["path"], group["wanted"]
            out_path = f"{out_prefix}/{hashlib.sha1(path.encode()).hexdigest()[:16]}.parquet"
            if StoragePath(out_path).exists():
                continue

            def rows(path: str, wanted: dict):
                for record in load_file(InputFileSpec(path=path, columns=["id", "text"])):
                    doc_id = str(record["id"])
                    for r in wanted.get(doc_id, ()):
                        yield {
                            "eval_id": r["eval_id"],
                            "path": path,
                            "id": doc_id,
                            "has_passage": r["has_passage"],
                            **classify(str(record["text"]), r["stem"], r["choices"], r["answer"]),
                        }

            yield write_parquet_file(rows(path, wanted), output_path=out_path, schema=SCHEMA)

    return run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("requests")
    parser.add_argument("out_prefix")
    parser.add_argument("--max-workers", type=int, default=100)
    parser.add_argument("--worker-ram", default="16g")
    args = parser.parse_args()
    with StoragePath(args.requests).open("r") as f:
        requests = json.load(f)
    by_path: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in requests:
        by_path[r["path"]][r["id"]].append(r)
    groups = [{"path": p, "wanted": dict(w)} for p, w in sorted(by_path.items())]
    resources = ResourceConfig(cpu=1, ram=args.worker_ram)
    ctx = ZephyrContext(name="contam-mmlu-copies", resources=resources, max_workers=args.max_workers)
    ctx.execute(Dataset.from_list(groups).map_shard(make_checker(args.out_prefix)))


if __name__ == "__main__":
    main()
