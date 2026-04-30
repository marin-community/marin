# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-off probe: average document length (tokens) per uncheatable_eval dataset."""

from levanter.data.text.cache import load_lm_dataset_cache
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.tokenizers import load_tokenizer

DATASETS = [
    "wikipedia_english-6330df",
    "github_python-baab41",
    "github_cpp-a9de07",
    "bbc_news-4df59f",
    "arxiv_physics-f4ad8c",
    "arxiv_computer_science-2b4f07",
    "ao3_english-bb5666",
]


def main() -> None:
    tokenizer = load_tokenizer("meta-llama/Meta-Llama-3.1-8B")
    fmt = TextLmDatasetFormat()
    print(f"{'dataset':<35s} {'docs':>8s} {'tokens':>14s} {'avg_len':>10s}")
    for name in DATASETS:
        path = f"gs://marin-us-east5/tokenized/uncheatable_eval/{name}/validation"
        cache = load_lm_dataset_cache(path, fmt, tokenizer, enforce_eos=True)
        ids = cache.store.tree["input_ids"]
        n = int(ids.num_rows)
        total = int(ids.data_size)
        avg = total / n if n else 0
        print(f"{name:<35s} {n:>8d} {total:>14d} {avg:>10.1f}")


if __name__ == "__main__":
    main()
