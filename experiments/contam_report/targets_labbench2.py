# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build contamination-scan targets for LABBench2.

The task dataset (EdisonScientific/labbench2 on Hugging Face) is gated, but the public
harness repository publishes its paper reports, which hold every task's question and
reference answer. Items are read from those reports at a pinned commit; the figures,
tables, PDFs and sequence files that some tasks attach are not indexed.

Writes ``OUT_DIR/labbench2/eval.parquet`` (columns ``id`` and ``text``; one record per
item, id ``labbench2-<subset>-<task id>``, plus ``-question`` and ``-answer`` views long
enough to stand alone), ``OUT_DIR/.manifest.json`` and ``OUT_DIR/.verbatim_items.json``
for `verbatim_copies.py`. Prompt paragraphs that several items share (answer-format
instructions, the source-quality preamble) are not indexed.

Usage: ``uv run --with tiktoken python experiments/contam_report/targets_labbench2.py OUT_DIR CACHE_DIR``
"""

import json
import sys
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

from experiments.contam_report.prepare_targets import write
from experiments.contam_report.targets_policy_qa import EvalTargets, free, letter_words, near

REPO = "EdisonScientific/labbench2"
REVISION = "0cabab27f6a23a5bc060a4699205709f1be7e93f"
REPORTS_DIR = "assets/reports_paper/"
# The dataset card lists 1,912 tasks over 15 subsets; the reports cover all of them.
EXPECTED_ITEMS = 1912
# A prompt paragraph shared by this many items is harness text, not item content.
SHARED_PARAGRAPH_ITEMS = 3
# An answer with this many words is matched like a free-text solution.
FREE_ANSWER_WORDS = 30
NO_ANSWER = ("", "None")
DOWNLOAD_TIMEOUT = 60


def fetch(path: str, cache_dir: Path) -> bytes:
    local = cache_dir / REVISION / path
    if not local.exists():
        url = f"https://raw.githubusercontent.com/{REPO}/{REVISION}/{urllib.parse.quote(path)}"
        local.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT) as response:
            local.write_bytes(response.read())
    return local.read_bytes()


def report_paths(cache_dir: Path) -> list[str]:
    url = f"https://api.github.com/repos/{REPO}/git/trees/{REVISION}?recursive=1"
    local = cache_dir / REVISION / "tree.json"
    if not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT) as response:
            local.write_bytes(response.read())
    tree = json.loads(local.read_bytes())
    assert not tree["truncated"], "the tree listing is truncated"
    return sorted(e["path"] for e in tree["tree"] if e["path"].startswith(REPORTS_DIR) and e["path"].endswith(".json"))


def tasks(cache_dir: Path) -> dict[str, dict]:
    """Every task in the reports, keyed by id, with a reference answer when any report has one.

    Reports run a task in several modes; the ``inject`` mode pastes the attached files into
    the question, so the shortest question of a task is the one the dataset holds.
    """
    found: dict[str, dict] = {}
    for path in report_paths(cache_dir):
        for case in json.loads(fetch(path, cache_dir))["cases"]:
            answer = str(case["expected_output"] or "")
            answer = "" if answer in NO_ANSWER else answer
            task = found.setdefault(case["id"], {"tag": case["tag"], "question": case["question"], "answer": ""})
            if len(case["question"]) < len(task["question"]):
                task["question"] = case["question"]
            task["answer"] = task["answer"] or answer
    return found


def labbench2(cache_dir: Path) -> EvalTargets:
    found = tasks(cache_dir)
    assert len(found) == EXPECTED_ITEMS, f"found {len(found)} tasks, expected {EXPECTED_ITEMS}"
    shared = Counter(p.strip() for task in found.values() for p in set(task["question"].split("\n\n")) if p.strip())
    out = EvalTargets("labbench2", f"https://github.com/{REPO}", REVISION, "paper reports", len(found))
    for task_id, task in sorted(found.items()):
        question = "\n\n".join(
            p.strip() for p in task["question"].split("\n\n") if p.strip() and shared[p.strip()] < SHARED_PARAGRAPH_ITEMS
        )
        answer = task["answer"]
        if not answer:
            item = {"kind": "question", "question": question}
        elif letter_words(answer) >= FREE_ANSWER_WORDS:
            item = free(question, answer)
        else:
            item = near(question, [answer])
        out.add(f"{task['tag']}-{task_id}", [question, answer], {"question": question, "answer": answer}, item)
    out.notes.append(
        f"{sum(c >= SHARED_PARAGRAPH_ITEMS for c in shared.values())} prompt paragraphs shared by "
        f"{SHARED_PARAGRAPH_ITEMS} or more items are not indexed; "
        f"{sum(not t['answer'] for t in found.values())} tasks have no reference answer."
    )
    return out


def main() -> None:
    out_dir, cache_dir = Path(sys.argv[1]), Path(sys.argv[2])
    targets = labbench2(cache_dir)
    write(out_dir, targets.name, targets.rows)
    (out_dir / ".manifest.json").write_text(json.dumps([targets.manifest()], indent=1))
    (out_dir / ".verbatim_items.json").write_text(json.dumps(targets.verbatim_items))


if __name__ == "__main__":
    main()
