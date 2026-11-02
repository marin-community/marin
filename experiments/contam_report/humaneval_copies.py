# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check whether kept documents hold HumanEval prompts and solutions verbatim (Gate 6).

Input: a JSON list of ``{"path", "id", "eval_id", "prompt", "solution"}``, one per
kept document that holds at least half of a HumanEval item. For each request it
records whether the prompt and the canonical solution occur in the document
exactly, after collapsing whitespace, and after also undoing JSON string escapes
(many copies sit inside JSON lines), plus the share of the prompt's and the
solution's whitespace 13-grams the document holds. Writes one Parquet file per
input file under OUT_PREFIX.

Usage: humaneval_copies.py REQUESTS_JSON OUT_PREFIX [--max-workers N] [--worker-ram R]
"""

import argparse
import hashlib
import json
from collections import defaultdict

import pyarrow as pa
from fray.types import ResourceConfig
from rigging.filesystem.storage_path import StoragePath
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.input_file import InputFileSpec
from zephyr.readers import load_file
from zephyr.writers import write_parquet_file

NGRAM_LENGTH = 13
JSON_ESCAPES = (("\\n", "\n"), ("\\t", "\t"), ('\\"', '"'), ("\\'", "'"), ("\\\\", "\\"))
SCHEMA = pa.schema(
    [
        ("eval_id", pa.string()),
        ("path", pa.string()),
        ("id", pa.string()),
        ("prompt_exact", pa.bool_()),
        ("prompt_normalized", pa.bool_()),
        ("solution_exact", pa.bool_()),
        ("solution_normalized", pa.bool_()),
        ("prompt_ngram_coverage", pa.float64()),
        ("solution_ngram_coverage", pa.float64()),
        ("solution_ngrams", pa.int64()),
    ]
)


def collapse(text: str) -> str:
    return " ".join(text.split())


def unescape(text: str) -> str:
    for escaped, plain in JSON_ESCAPES:
        text = text.replace(escaped, plain)
    return text


def ngrams(tokens: list[str]) -> set[tuple[str, ...]]:
    return {tuple(tokens[i : i + NGRAM_LENGTH]) for i in range(len(tokens) - NGRAM_LENGTH + 1)}


def found_ngrams(wanted: set[tuple[str, ...]], doc_tokens: list[str]) -> set[tuple[str, ...]]:
    """Return the wanted n-grams that occur in ``doc_tokens``, without building the doc's n-gram set."""
    firsts = {g[0] for g in wanted}
    found = set()
    for i in range(len(doc_tokens) - NGRAM_LENGTH + 1):
        if doc_tokens[i] in firsts and (g := tuple(doc_tokens[i : i + NGRAM_LENGTH])) in wanted:
            found.add(g)
    return found


def coverage(part: str, doc_token_lists: list[list[str]]) -> tuple[float | None, int]:
    part_ngrams = ngrams(part.split())
    if not part_ngrams:
        return None, 0
    found = set().union(*(found_ngrams(part_ngrams, tokens) for tokens in doc_token_lists))
    return len(found) / len(part_ngrams), len(part_ngrams)


def check(doc: str, prompt: str, solution: str) -> dict:
    flat = collapse(doc)
    flat_unescaped = collapse(unescape(doc))
    doc_token_lists = [flat.split(), flat_unescaped.split()]
    prompt_cov, _ = coverage(prompt, doc_token_lists)
    solution_cov, solution_ngrams = coverage(solution, doc_token_lists)
    return {
        "prompt_exact": prompt.strip() in doc,
        "prompt_normalized": collapse(prompt) in flat or collapse(prompt) in flat_unescaped,
        "solution_exact": solution.strip() in doc,
        "solution_normalized": collapse(solution) in flat or collapse(solution) in flat_unescaped,
        "prompt_ngram_coverage": prompt_cov,
        "solution_ngram_coverage": solution_cov,
        "solution_ngrams": solution_ngrams,
    }


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
                    for eval_id, prompt, solution in wanted.get(doc_id, ()):
                        yield {
                            "eval_id": eval_id,
                            "path": path,
                            "id": doc_id,
                            **check(str(record["text"]), prompt, solution),
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
        by_path[r["path"]][r["id"]].append((r["eval_id"], r["prompt"], r["solution"]))
    groups = [{"path": p, "wanted": dict(w)} for p, w in sorted(by_path.items())]
    resources = ResourceConfig(cpu=1, ram=args.worker_ram)
    ctx = ZephyrContext(name="contam-humaneval-copies", resources=resources, max_workers=args.max_workers)
    ctx.execute(Dataset.from_list(groups).map_shard(make_checker(args.out_prefix)))


if __name__ == "__main__":
    main()
