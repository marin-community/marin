# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the literal-search patterns for `literal_scan.py` (gate 22).

The 13-gram scan cannot see short strings (a canary line is 5 tokens, so it never
forms a 13-gram) or text that differs from the indexed item in case or line
breaks. This writes, for a few AAII v5 evals, literal phrases that a copy of an
item would contain:

* ``canary``: the canary GUIDs in the Terminal-Bench task files;
* ``task_name``: each Terminal-Bench task's directory name, when distinctive
  (at least ``MIN_NAME_HYPHENS`` hyphens);
* ``instruction``: up to ``PHRASES_PER_ITEM`` non-overlapping
  ``PHRASE_WORDS``-word phrases from each Terminal-Bench instruction;
* ``question``: the first and last ``PHRASE_WORDS`` words of each AA-Omniscience
  question, and its answer when it has at least ``PHRASE_WORDS`` words.

Words match case-insensitively with any whitespace (or a JSON ``\\n``/``\\t``)
between them. Output: ``[{"pattern_id", "eval", "item", "kind", "text", "regex"}]``.

Usage: literal_patterns.py TARGETS_DIR AAII_ITEMS_JSON OUT_JSON
"""

import json
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

PHRASE_WORDS = 8
PHRASES_PER_ITEM = 12
MIN_NAME_HYPHENS = 2
MIN_LETTER_WORDS = 5
TASK_EVALS = ("terminal_bench_4", "terminal_bench_science")
QUESTION_EVAL = "aa_omniscience_v1_1"
CANARY = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
CANARY_LINE = re.compile(r"canary GUID ([0-9a-f-]{36})")
WORD_GAP = r"(?:\s|\\[nt])+"


def phrase_regex(words: list[str]) -> str:
    return WORD_GAP.join(re.escape(w) for w in words)


def phrases(text: str) -> list[list[str]]:
    """Non-overlapping word windows with enough words containing letters, skipping canary lines."""
    words = [w for line in text.splitlines() if "canary" not in line.lower() for w in line.split()]
    windows = [words[i : i + PHRASE_WORDS] for i in range(0, len(words) - PHRASE_WORDS + 1, PHRASE_WORDS)]
    good = [w for w in windows if sum(any(c.isalpha() for c in t) for t in w) >= MIN_LETTER_WORDS]
    step = max(1, len(good) // PHRASES_PER_ITEM)
    return good[::step][:PHRASES_PER_ITEM]


def task_patterns(targets_dir: Path) -> list[dict]:
    out, canaries = [], set()
    for eval_name in TASK_EVALS:
        for row in pq.read_table(targets_dir / eval_name / "eval.parquet").to_pylist():
            canaries |= set(CANARY_LINE.findall(row["text"]))
            if not row["id"].endswith("-instruction"):
                continue
            task = row["id"].removeprefix(f"{eval_name}-").removesuffix("-instruction")
            if task.count("-") >= MIN_NAME_HYPHENS:
                out.append(
                    {"eval": eval_name, "item": task, "kind": "task_name", "text": task, "regex": re.escape(task)}
                )
            for words in phrases(row["text"]):
                text = " ".join(words)
                out.append(
                    {"eval": eval_name, "item": task, "kind": "instruction", "text": text, "regex": phrase_regex(words)}
                )
    for guid in sorted(canaries):
        assert CANARY.fullmatch(guid), guid
        out.append({"eval": "terminal_bench", "item": "", "kind": "canary", "text": guid, "regex": re.escape(guid)})
    return out


def question_patterns(items: dict) -> list[dict]:
    out = []
    for eval_id, item in items.items():
        if item["eval"] != QUESTION_EVAL:
            continue
        words = item["question"].split()
        if len(words) < PHRASE_WORDS:
            continue
        windows = [words[:PHRASE_WORDS]] + ([words[-PHRASE_WORDS:]] if len(words) >= 2 * PHRASE_WORDS else [])
        answer = item["answers"][0].split()
        if len(answer) >= PHRASE_WORDS:
            windows.append(answer[:PHRASE_WORDS])
        for w in windows:
            out.append(
                {
                    "eval": QUESTION_EVAL,
                    "item": eval_id,
                    "kind": "question",
                    "text": " ".join(w),
                    "regex": phrase_regex(w),
                }
            )
    return out


def main() -> None:
    targets_dir, items_path, out_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    patterns = task_patterns(targets_dir) + question_patterns(json.loads(items_path.read_text()))
    patterns = [{"pattern_id": i, **p} for i, p in enumerate(patterns)]
    out_path.write_text(json.dumps(patterns))
    kinds = {}
    for p in patterns:
        kinds[(p["eval"], p["kind"])] = kinds.get((p["eval"], p["kind"]), 0) + 1
    print(f"{len(patterns)} patterns -> {out_path}: {kinds}")


if __name__ == "__main__":
    main()
