# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Write the question and answer of every core eval item for `verbatim_copies.py`.

Reads the Hugging Face test splits from RAW_DIR, a `huggingface_hub` download of
``openai/gsm8k``, ``cais/mmlu``, ``TIGER-Lab/MMLU-Pro`` and
``openai/openai_humaneval`` (`verify_targets.py` checks the indexed corpus against
the same sources). Each item is keyed by its corpus ``eval_id`` and has a ``kind``:

* ``free``: ``question`` is what the model sees; ``answers`` are the reference
  answer variants. GSM8K's solution is given with and without its ``<<48/2=24>>``
  calculator notes, since copies differ on keeping them.
* ``choice``: ``question`` is the stem without the quoted source passage of the
  history subjects (``has_passage``); ``choices`` and the correct ``answer`` index
  follow.
* ``near``: ``question`` and short reference ``answers`` (HLE-Diamond,
  AA-Omniscience). The answer counts only after the question, since a short answer
  occurs anywhere by chance.

``core`` writes the core evals from RAW_DIR. ``aaii`` writes the public AAII v5 QA
evals, read from Hugging Face at the revisions `targets_aaii_v5.py` indexed; its
task evals are indexed per view (instruction, solution, tests), so their scan
already scores the instruction alone.

Usage: verbatim_items.py core RAW_DIR OUT_JSON | verbatim_items.py aaii OUT_JSON
"""

import csv
import json
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download

from experiments.contam_report.targets_aaii_v5 import (
    HLE_DIAMOND_REPO,
    HLE_DIAMOND_REVISION,
    OMNISCIENCE_REPO,
    OMNISCIENCE_REVISION,
)

PASSAGE_INTRO = "This question refers to the following information."
CALCULATOR_NOTE = re.compile(r"<<[^>]*>>")


def rows(raw_dir: Path, relative: str) -> list[dict]:
    return pq.read_table(raw_dir / relative).to_pylist()


def stem(question: str) -> tuple[str, bool]:
    """Drop the quoted passage of a history question, keeping its last line, the question proper."""
    if not question.strip().startswith(PASSAGE_INTRO):
        return question, False
    return [line for line in question.strip().split("\n") if line.strip()][-1], True


def choice_item(eval_name: str, question: str, choices: list[str], answer: int) -> dict:
    text, has_passage = stem(question)
    return {
        "eval": eval_name,
        "kind": "choice",
        "question": text,
        "has_passage": has_passage,
        "choices": choices,
        "answer": answer,
    }


def items(raw_dir: Path) -> dict[str, dict]:
    out = {}
    for i, r in enumerate(rows(raw_dir, "openai__gsm8k/main/test-00000-of-00001.parquet")):
        answers = [r["answer"], CALCULATOR_NOTE.sub("", r["answer"])]
        out[f"gsm8k-test-{i}"] = {"eval": "gsm8k", "kind": "free", "question": r["question"], "answers": answers}
    for i, r in enumerate(rows(raw_dir, "openai__openai_humaneval/openai_humaneval/test-00000-of-00001.parquet")):
        out[f"humaneval-test-{i}"] = {
            "eval": "humaneval",
            "kind": "free",
            "question": r["prompt"],
            "answers": [r["canonical_solution"]],
        }
    for i, r in enumerate(rows(raw_dir, "cais__mmlu/all/test-00000-of-00001.parquet")):
        out[f"mmlu-test-{r['subject']}-{i}"] = choice_item("mmlu", r["question"], list(r["choices"]), int(r["answer"]))
    for r in rows(raw_dir, "TIGER-Lab__MMLU-Pro/data/test-00000-of-00001.parquet"):
        item = choice_item("mmlu_pro", r["question"], list(r["options"]), int(r["answer_index"]))
        out[f"mmlu_pro-test-{r['question_id']}"] = item
    return out


def near_item(eval_name: str, question: str, answer: str) -> dict:
    return {"eval": eval_name, "kind": "near", "question": question, "answers": [answer], "distractors": []}


def aaii_items() -> dict[str, dict]:
    """HLE-Diamond and AA-Omniscience items, keyed by the ids `targets_aaii_v5.py` gives their records."""
    out = {}
    path = f"datasets/{HLE_DIAMOND_REPO}@{HLE_DIAMOND_REVISION}/data/test-00000-of-00001.parquet"
    with HfFileSystem().open(path) as f:
        for r in pq.read_table(f, columns=["id", "question", "answer"]).to_pylist():
            out[f"hle_diamond-{r['id']}"] = near_item("hle_diamond", r["question"], r["answer"])
    csv_path = hf_hub_download(
        OMNISCIENCE_REPO, "AA-Omniscience_dataset_public.csv", repo_type="dataset", revision=OMNISCIENCE_REVISION
    )
    with open(csv_path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            out[f"aa_omniscience_v1_1-{r['question_id']}"] = near_item("aa_omniscience_v1_1", r["question"], r["answer"])
    return out


def main() -> None:
    command, *args = sys.argv[1:]
    if command == "core":
        result = items(Path(args[0]))
    elif command == "aaii":
        result = aaii_items()
    else:
        raise ValueError(f"unknown command {command!r}")
    out_path = Path(args[-1])
    out_path.write_text(json.dumps(result))
    print(f"{len(result)} items -> {out_path}")


if __name__ == "__main__":
    main()
