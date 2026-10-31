"""Check the target eval corpus against its sources (gate: verify eval data).

Reads the corpus back from DIR (a copy downloaded from S3) and checks, for each
eval, that every source item is present once with its full text: GSM8K, MMLU,
MMLU-Pro and HumanEval against the Hugging Face datasets lm-eval loads, and the
Uncheatable Eval subsets against the raw files. Also reports which records the
Bloom index cannot match.

Usage: verify_targets.py CORPUS_DIR RAW_UNCHEATABLE_DIR BLOOM_DIR
"""

import gzip
import json
import sys
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import HfApi

HF_SOURCES = {
    "gsm8k": ("openai/gsm8k", "main"),
    "mmlu": ("cais/mmlu", "all"),
    "mmlu_pro": ("TIGER-Lab/MMLU-Pro", None),
    "humaneval": ("openai/openai_humaneval", None),
}


def expected_hf(name: str) -> tuple[str, list[tuple[str, list[str]]]]:
    repo, config = HF_SOURCES[name]
    sha = HfApi().dataset_info(repo).sha
    ds = load_dataset(repo, config, split="test", revision=sha)
    if name == "gsm8k":
        items = [(f"gsm8k-test-{i}", [r["question"], r["answer"]]) for i, r in enumerate(ds)]
    elif name == "mmlu":
        items = [(f"mmlu-test-{r['subject']}-{i}", [r["question"], *r["choices"]]) for i, r in enumerate(ds)]
    elif name == "mmlu_pro":
        items = [(f"mmlu_pro-test-{r['question_id']}", [r["question"], *r["options"]]) for r in ds]
    else:
        items = [(f"humaneval-test-{i}", [r["prompt"], r["canonical_solution"]]) for i, r in enumerate(ds)]
    return f"{repo}@{sha[:10]}", items


def expected_uncheatable(raw_dir: Path, subset: str) -> tuple[str, list[tuple[str, list[str]]]]:
    (path,) = raw_dir.glob(f"{subset}_*.jsonl.gz")
    with gzip.open(path, "rt") as f:
        texts = [json.loads(line)["text"] for line in f]
    return path.name, [(f"uncheatable-{subset}-{i}", [t]) for i, t in enumerate(texts)]


def check(corpus_dir: Path, name: str, source: str, items: list[tuple[str, list[str]]], indexed: set[str]) -> dict:
    table = pq.read_table(corpus_dir / name / "eval.parquet")
    records = dict(zip(table["id"].to_pylist(), table["text"].to_pylist(), strict=True))
    if len(records) != table.num_rows:
        raise ValueError(f"{name}: duplicate ids")
    missing = [i for i, _ in items if i not in records]
    wrong = [i for i, parts in items if i in records and not all(p in records[i] for p in parts)]
    extra = set(records) - {i for i, _ in items}
    empty = [i for i, t in records.items() if not t.strip()]
    unindexed = sorted(set(records) - indexed)
    return {
        "eval": name,
        "source": source,
        "source_items": len(items),
        "corpus_records": len(records),
        "missing": len(missing),
        "text_mismatch": len(wrong),
        "extra": len(extra),
        "empty": len(empty),
        "unindexed": len(unindexed),
        "unindexed_ids": unindexed[:5],
    }


def main() -> None:
    corpus_dir, raw_dir, bloom_dir = (Path(a) for a in sys.argv[1:4])
    index = pq.read_table(bloom_dir / "_bloom" / "eval_hash_index.parquet", columns=["eval_id"])
    indexed = set(pc.unique(index["eval_id"]).to_pylist())
    results = [check(corpus_dir, name, *expected_hf(name), indexed) for name in HF_SOURCES]
    for path in sorted(corpus_dir.glob("uncheatable_*")):
        subset = path.name.removeprefix("uncheatable_")
        results.append(check(corpus_dir, path.name, *expected_uncheatable(raw_dir, subset), indexed))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
