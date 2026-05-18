# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sanity-check sampler for the v0.1 datakit-clustered store.

For each of the 40 clusters, opens the in-flight cache for each quality
bucket and detokenizes a couple of documents. Use against in-progress or
completed runs:

    uv run python scripts/sample_v0_1_store.py [--output PATH] [--per-cluster N]
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import textwrap

import numpy as np
from levanter.store.tree_store import TreeStore
from marin.utils import fsspec_glob
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


_DEFAULT_OUTPUT = "gs://marin-eu-west4/datakit/store/v0.1_20260518"
_TOKENIZER = "marin-community/marin-tokenizer"
_EXEMPLAR = {"input_ids": np.array([0], dtype=np.int32)}
_PREVIEW_TOKENS = 200


def _list_part_dirs(cluster_quality_root: str) -> list[str]:
    """List ``part-NNNNN-of-NNNNN/`` subdirs under a cluster=K/quality=Q/ dir."""
    return sorted(p.rstrip("/") for p in fsspec_glob(f"{cluster_quality_root}/part-*-of-*"))


def _open_store(part_dir: str) -> TreeStore | None:
    """Open the Levanter shard cache at part_dir, or return None on failure."""
    try:
        return TreeStore.open(_EXEMPLAR, part_dir, mode="r", cache_metadata=True)
    except Exception as e:
        logger.warning("could not open %s: %s", part_dir, e)
        return None


def _sample_docs(store: TreeStore, n: int) -> list[list[int]]:
    """Return up to ``n`` documents from store as Python lists of token ids."""
    total = len(store)
    if total == 0:
        return []
    indices = random.sample(range(total), min(n, total))
    docs: list[list[int]] = []
    for i in sorted(indices):
        rec = store[i]
        toks = rec["input_ids"]
        # toks could be numpy or jagged-list type; normalize to list[int].
        docs.append(list(np.asarray(toks).tolist()))
    return docs


def _print_doc_preview(tokenizer, cluster_id: int, quality: int, source_part: str, tokens: list[int]) -> None:
    preview = tokenizer.decode(tokens[:_PREVIEW_TOKENS], skip_special_tokens=True)
    preview = textwrap.shorten(preview.replace("\n", " "), width=320, placeholder=" ...")
    print(f"  [c={cluster_id} q={quality} part={os.path.basename(source_part)} n_tok={len(tokens):,}] {preview}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=_DEFAULT_OUTPUT, help="Store root with cluster=K/quality=Q/ subdirs.")
    parser.add_argument("--per-cluster", type=int, default=10, help="Docs to sample per cluster (split across quality).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    random.seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER)

    n_quality = 5
    per_quality = max(1, args.per_cluster // n_quality)

    for cluster_id in range(40):
        cluster_root = f"{args.output.rstrip('/')}/cluster={cluster_id}"
        printed = 0
        print(f"\n=== cluster={cluster_id} ===")
        for q in range(n_quality):
            cq_root = f"{cluster_root}/quality={q}"
            parts = _list_part_dirs(cq_root)
            if not parts:
                continue
            part_dir = random.choice(parts)
            store = _open_store(part_dir)
            if store is None:
                continue
            docs = _sample_docs(store, per_quality)
            for tokens in docs:
                _print_doc_preview(tokenizer, cluster_id, q, part_dir, tokens)
                printed += 1
        if printed == 0:
            print("  (no docs available yet)")


if __name__ == "__main__":
    main()
