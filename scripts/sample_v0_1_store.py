# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sanity-check sampler for the v0.1 datakit-clustered store.

For each of the 40 clusters, opens the in-flight cache for each quality
bucket and detokenizes a couple of documents. Use against in-progress or
completed runs:

    uv run python scripts/sample_v0_1_store.py [--output PATH] [--per-cluster N]
"""

import argparse
import logging
import os
import random
import sys
import textwrap

import numpy as np
from levanter.store.tree_store import TreeStore
from rigging.filesystem import url_to_fs
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


_DEFAULT_OUTPUT = "gs://marin-eu-west4/datakit/store/v0.1_20260518"
_TOKENIZER = "marin-community/marin-tokenizer"
_EXEMPLAR = {"input_ids": np.array([0], dtype=np.int32)}
_PREVIEW_TOKENS = 200
_PART_PREFIX = "part-"


def _list_part_dirs(cluster_quality_root: str) -> list[str]:
    """Shallowly list completed ``part-NNNNN-of-NNNNN/`` subdirs under cluster_quality_root.

    Uses ``fs.ls(detail=True)`` (one delimited call). ``StoragePath.glob`` against
    gcsfs calls ``_find`` which recursively lists every object under the
    prefix -- against a near-complete cluster=K/quality=Q/ (~98K parts x
    several files each) that's a 500K+ object listing per (cluster, quality),
    which wedges in practice.
    """
    fs, _ = url_to_fs(cluster_quality_root)
    prefix = cluster_quality_root.removeprefix("gs://").rstrip("/")
    parts: list[str] = []
    for info in fs.ls(prefix, detail=True):
        if info.get("type") != "directory":
            continue
        bn = os.path.basename(info["name"].rstrip("/"))
        # skip ``.tmp.<hash>`` directories from in-flight writers
        if bn.startswith(_PART_PREFIX) and ".tmp." not in bn:
            parts.append(f"gs://{info['name'].rstrip('/')}")
    parts.sort()
    return parts


def _open_store(part_dir: str) -> TreeStore | None:
    """Open the Levanter shard cache at part_dir, or return None on failure."""
    try:
        # cache_metadata=False keeps open() from eagerly reading every per-shard
        # ledger (slow when a part has tens of thousands of shards written).
        return TreeStore.open(_EXEMPLAR, part_dir, mode="r", cache_metadata=False)
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
    parser.add_argument("--cluster", type=int, default=None, help="If set, only sample this cluster_id.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    random.seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER)

    n_quality = 5
    per_quality = max(1, args.per_cluster // n_quality)

    cluster_ids = [args.cluster] if args.cluster is not None else range(40)
    for cluster_id in cluster_ids:
        cluster_root = f"{args.output.rstrip('/')}/cluster={cluster_id}"
        printed = 0
        print(f"\n=== cluster={cluster_id} ===", flush=True)
        for q in range(n_quality):
            cq_root = f"{cluster_root}/quality={q}"
            parts = _list_part_dirs(cq_root)
            logger.info("cluster=%d quality=%d: %d completed parts", cluster_id, q, len(parts))
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
            sys.stdout.flush()
        if printed == 0:
            print("  (no docs available yet)", flush=True)


if __name__ == "__main__":
    main()
