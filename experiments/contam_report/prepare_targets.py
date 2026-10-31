"""Build the target eval corpus for the hero contamination study (Gate 2).

Writes one Parquet file per eval with columns ``id`` and ``text`` to OUT_DIR,
the layout `marin.datakit.decon.build_eval_bloom` reads. Paragraphs are separated
by a blank line, so question and answer are matched as separate paragraphs.
"""

import gzip
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

UNCHEATABLE_SUBSETS = (
    "wikipedia_english",
    "github_python",
    "github_cpp",
    "bbc_news",
    "arxiv_physics",
    "arxiv_computer_science",
    "ao3_english",
)


def write(out_dir: Path, name: str, rows: list[dict]) -> None:
    path = out_dir / name / "eval.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
    print(f"{name}: {len(rows)} records -> {path}")


def gsm8k() -> list[dict]:
    ds = load_dataset("openai/gsm8k", "main", split="test")
    return [{"id": f"gsm8k-test-{i}", "text": f"{r['question']}\n\n{r['answer']}"} for i, r in enumerate(ds)]


def mmlu() -> list[dict]:
    ds = load_dataset("cais/mmlu", "all", split="test")
    rows = []
    for i, r in enumerate(ds):
        choices = "\n".join(r["choices"])
        rows.append({"id": f"mmlu-test-{r['subject']}-{i}", "text": f"{r['question']}\n{choices}"})
    return rows


def mmlu_pro() -> list[dict]:
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    return [
        {"id": f"mmlu_pro-test-{r['question_id']}", "text": r["question"] + "\n" + "\n".join(r["options"])} for r in ds
    ]


def humaneval() -> list[dict]:
    ds = load_dataset("openai/openai_humaneval", split="test")
    return [{"id": f"humaneval-test-{i}", "text": r["prompt"] + r["canonical_solution"]} for i, r in enumerate(ds)]


def uncheatable(raw_dir: Path) -> dict[str, list[dict]]:
    out = {}
    for subset in UNCHEATABLE_SUBSETS:
        (path,) = raw_dir.glob(f"{subset}_*.jsonl.gz")
        rows = []
        with gzip.open(path, "rt") as f:
            for i, line in enumerate(f):
                rec = json.loads(line)
                text = rec["text"] if isinstance(rec, dict) else rec
                rows.append({"id": f"uncheatable-{subset}-{i}", "text": text})
        out[f"uncheatable_{subset}"] = rows
    return out


def main() -> None:
    out_dir, raw_dir = Path(sys.argv[1]), Path(sys.argv[2])
    write(out_dir, "gsm8k", gsm8k())
    write(out_dir, "mmlu", mmlu())
    write(out_dir, "mmlu_pro", mmlu_pro())
    write(out_dir, "humaneval", humaneval())
    for name, rows in uncheatable(raw_dir).items():
        write(out_dir, name, rows)


if __name__ == "__main__":
    main()
