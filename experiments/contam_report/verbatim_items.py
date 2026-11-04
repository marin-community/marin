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

Usage: verbatim_items.py RAW_DIR OUT_JSON
"""

import json
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

PASSAGE_INTRO = "This question refers to the following information."
CALCULATOR_NOTE = re.compile(r"<<[^>]*>>")


def rows(raw_dir: Path, relative: str) -> list[dict]:
    return pq.read_table(raw_dir / relative).to_pylist()


def stem(question: str) -> tuple[str, bool]:
    """Drop the quoted passage of a history question, keeping its last line, the question proper."""
    if not question.strip().startswith(PASSAGE_INTRO):
        return question, False
    return [line for line in question.strip().split("\n") if line.strip()][-1], True


def choice_item(question: str, choices: list[str], answer: int) -> dict:
    text, has_passage = stem(question)
    return {"kind": "choice", "question": text, "has_passage": has_passage, "choices": choices, "answer": answer}


def items(raw_dir: Path) -> dict[str, dict]:
    out = {}
    for i, r in enumerate(rows(raw_dir, "openai__gsm8k/main/test-00000-of-00001.parquet")):
        answers = [r["answer"], CALCULATOR_NOTE.sub("", r["answer"])]
        out[f"gsm8k-test-{i}"] = {"kind": "free", "question": r["question"], "answers": answers}
    for i, r in enumerate(rows(raw_dir, "openai__openai_humaneval/openai_humaneval/test-00000-of-00001.parquet")):
        out[f"humaneval-test-{i}"] = {"kind": "free", "question": r["prompt"], "answers": [r["canonical_solution"]]}
    for i, r in enumerate(rows(raw_dir, "cais__mmlu/all/test-00000-of-00001.parquet")):
        out[f"mmlu-test-{r['subject']}-{i}"] = choice_item(r["question"], list(r["choices"]), int(r["answer"]))
    for r in rows(raw_dir, "TIGER-Lab__MMLU-Pro/data/test-00000-of-00001.parquet"):
        item = choice_item(r["question"], list(r["options"]), int(r["answer_index"]))
        out[f"mmlu_pro-test-{r['question_id']}"] = item
    return out


def main() -> None:
    raw_dir, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    result = items(raw_dir)
    out_path.write_text(json.dumps(result))
    print(f"{len(result)} items -> {out_path}")


if __name__ == "__main__":
    main()
