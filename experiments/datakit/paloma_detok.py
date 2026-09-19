# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reconstruct Paloma raw text from an in-region tokenized cache.

The Paloma raw source is out-of-region (gated HF, no in-region download step), so a fresh-tokenizer
build of the eval sets fails. But the ``paloma/{subset}-llama3`` token caches ARE complete in-region,
and the llama3 tokenizer is byte-lossless (``decode(encode(text)) == text``). This script decodes those
caches back to text jsonl.gz in-region, from which any tokenizer can rebuild the eval cache without the
original raw. Run as an Iris CPU job on an in-region cluster.
"""

import gzip
import json

import click
import numpy as np
from levanter.store.tree_store import TreeStore
from rigging.filesystem.storage_path import StoragePath, prefix_join
from transformers import AutoTokenizer

from experiments.datasets.paloma import _PALOMA_SUBSETS
from experiments.llama import llama3_tokenizer

_EXEMPLAR = {"input_ids": np.zeros((0,), dtype=np.int32)}


def detok_subset(subset: str, *, in_prefix: str, src_tokenizer: str, src_tag: str, version_in: str, out: str) -> int:
    """Decode one subset's token cache to text jsonl.gz. Returns the document count written."""
    tok = AutoTokenizer.from_pretrained(src_tokenizer)
    cache_path = prefix_join(in_prefix, f"{subset}-{src_tag}/{version_in}/validation")
    store = TreeStore.open(_EXEMPLAR, cache_path, mode="r")
    out_path = prefix_join(out, f"{_PALOMA_SUBSETS[subset]}/val/val-00000.jsonl.gz")
    n = 0
    with StoragePath(out_path).open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb") as gz:
        for i in range(len(store)):
            ids = np.asarray(store[i]["input_ids"]).tolist()
            text = tok.decode(ids, clean_up_tokenization_spaces=False, skip_special_tokens=True)
            gz.write((json.dumps({"text": text}) + "\n").encode("utf-8"))
            n += 1
    print(f"{subset}: wrote {n} docs -> {out_path}", flush=True)
    return n


@click.command()
@click.option("--subset", default="all", help="Paloma subset to detokenize, or 'all'.")
@click.option("--in-prefix", required=True, help="Prefix holding the token caches, e.g. s3://.../marin/paloma")
@click.option("--src-tokenizer", default=llama3_tokenizer, help="Tokenizer the source cache was built with.")
@click.option("--src-tag", default="llama3", help="Tokenizer tag in the source cache path.")
@click.option("--version-in", default="2026.06.28", help="Version of the source token cache.")
@click.option("--out", required=True, help="In-region raw output dir for the decoded text jsonl.gz.")
def main(subset: str, in_prefix: str, src_tokenizer: str, src_tag: str, version_in: str, out: str) -> None:
    subsets = list(_PALOMA_SUBSETS) if subset == "all" else [subset]
    total = 0
    for s in subsets:
        total += detok_subset(
            s, in_prefix=in_prefix, src_tokenizer=src_tokenizer, src_tag=src_tag, version_in=version_in, out=out
        )
    print(f"DONE: {len(subsets)} subsets, {total} docs total.", flush=True)


if __name__ == "__main__":
    main()
