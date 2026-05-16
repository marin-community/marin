# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-cluster summaries (top c-TF-IDF terms + reps) at K=k_train and each coarser K.

Co-partitioning across ``NormalizedData -> EmbeddingAttrData ->
AssignmentAttrData`` makes this cheap: we reservoir-sample
``(source, shard_basename, row_idx)`` from the assignment shards (small),
then pluck the corresponding text directly from the source's normalized
parquet at that row index. No id lookup, no full scan.

At K=5000 with n_sample_per_cluster=200 that's 1M text reads. For K=40
with n_sample=2000 it's 80K text reads — trivial.

Two-pass:
  1. Stream every source's AssignmentAttrData shards once, maintaining
     two per-cluster data structures in parallel:
       * a reservoir of ``n_sample_per_cluster`` random
         ``(source, basename, row_idx, dist_<k_train>)`` rows (used for
         c-TF-IDF — needs uniform coverage of the cluster's vocabulary)
       * a top-``n_reps``-by-smallest-dist heap (used for representative
         docs — needs the globally-closest-to-centroid docs, not the
         closest within a random sample).
  2. Fetch the union of the two sets' texts from each source's
     normalized parquet (one read per touched shard), assemble
     cluster_stats_<k_view>.json.
"""

import heapq
import json
import logging
import os
import tempfile
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq
from rigging.filesystem import open_url

from experiments.datakit.cluster.assign import AssignmentAttrData

logger = logging.getLogger(__name__)

DEFAULT_N_REPS = 5
DEFAULT_N_TERMS = 10
DEFAULT_EXCERPT_CHARS = 500
CTFIDF_MAX_FEATURES = 50_000


def _scan_assignments(
    assignments: dict[str, AssignmentAttrData],
    cluster_col: str,
    dist_col: str,
    n_sample_per_cluster: int,
    n_reps: int,
    seed: int = 42,
) -> tuple[
    dict[int, list[tuple[str, str, int, float]]],
    dict[int, list[tuple[str, str, int, float]]],
    dict[str, str],
]:
    """One pass over all assignment shards. Returns ``(samples, reps, source_main_dirs)``.

    * ``samples[c]``: reservoir-sampled ``(source, basename, row_idx, dist)``
      rows in cluster ``c`` (uniform random; for c-TF-IDF).
    * ``reps[c]``: the ``n_reps`` rows with smallest ``dist`` in cluster ``c``,
      sorted ascending by ``dist`` (for representative docs).
    * ``source_main_dirs[source_name]``: normalized main_output_dir.
    """
    rng = np.random.default_rng(seed)
    samples: dict[int, list[tuple[str, str, int, float]]] = defaultdict(list)
    counts: dict[int, int] = defaultdict(int)
    # Top-K-by-smallest-dist via a max-heap on (-dist). heappushpop keeps
    # heap size <= n_reps and the items with the n_reps smallest dists.
    reps_heap: dict[int, list[tuple[float, str, str, int]]] = defaultdict(list)
    source_main_dirs: dict[str, str] = {}

    for source_name, attr in sorted(assignments.items()):
        source_main_dirs[source_name] = attr.source_main_dir
        for shard_uri in attr.shard_paths():
            basename = os.path.basename(shard_uri)
            table = pq.read_table(shard_uri, columns=[cluster_col, dist_col])
            cl = table[cluster_col].to_numpy(zero_copy_only=False)
            d_arr = table[dist_col].to_numpy(zero_copy_only=False)
            for row_idx in range(len(cl)):
                c = int(cl[row_idx])
                d = float(d_arr[row_idx])
                entry = (source_name, basename, row_idx, d)

                # Reservoir sample for c-TF-IDF.
                counts[c] += 1
                if len(samples[c]) < n_sample_per_cluster:
                    samples[c].append(entry)
                else:
                    j = int(rng.integers(0, counts[c]))
                    if j < n_sample_per_cluster:
                        samples[c][j] = entry

                # Top-K heap for reps: (-d, ...) so heap[0] has the largest dist.
                rep_item = (-d, source_name, basename, row_idx)
                if len(reps_heap[c]) < n_reps:
                    heapq.heappush(reps_heap[c], rep_item)
                else:
                    heapq.heappushpop(reps_heap[c], rep_item)
        logger.info("Scanned %s assignments; clusters seen=%d", source_name, len(samples))

    reps: dict[int, list[tuple[str, str, int, float]]] = {}
    for c, heap in reps_heap.items():
        reps[c] = sorted(
            ((sn, bn, ri, -neg_d) for neg_d, sn, bn, ri in heap),
            key=lambda r: r[3],
        )

    return samples, reps, source_main_dirs


def _fetch_texts(
    source_main_dirs: dict[str, str],
    needed: set[tuple[str, str, int]],
    excerpt_chars: int,
) -> dict[tuple[str, str, int], str]:
    """For each (source, basename, row_idx) in ``needed``, pluck text from the normalized shard."""
    plan: dict[tuple[str, str], list[int]] = defaultdict(list)
    for source_name, basename, row_idx in needed:
        plan[(source_name, basename)].append(row_idx)

    out: dict[tuple[str, str, int], str] = {}
    for (source_name, basename), row_idxs in plan.items():
        shard_uri = f"{source_main_dirs[source_name].rstrip('/')}/{basename}"
        # Reading only the text column from this one shard is much cheaper
        # than a corpus scan. For 256MB partitions that's ~hundreds of MB
        # of GCS read per source — manageable.
        text_col = pq.read_table(shard_uri, columns=["text"])["text"]
        for row_idx in row_idxs:
            t = text_col[row_idx].as_py() or ""
            out[(source_name, basename, row_idx)] = t[:excerpt_chars]
    logger.info("Fetched %d texts across %d (source, shard) pairs", len(out), len(plan))
    return out


def _ctfidf(texts_by_cluster: dict[int, list[str]], n_terms: int) -> dict[int, list[str]]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    cluster_ids = sorted(texts_by_cluster)
    docs = [" ".join(texts_by_cluster[c]) for c in cluster_ids]
    vec = TfidfVectorizer(max_features=CTFIDF_MAX_FEATURES, ngram_range=(1, 2), stop_words="english")
    tfidf = vec.fit_transform(docs)
    vocab = np.array(vec.get_feature_names_out())
    top: dict[int, list[str]] = {}
    for row, c in enumerate(cluster_ids):
        order = np.argsort(-tfidf[row].toarray().ravel())[:n_terms]
        top[c] = vocab[order].tolist()
    return top


def summarize_at_k(
    output_path: str,
    k_train: int,
    k_view: int,
    assignments: dict[str, AssignmentAttrData],
    n_sample_per_cluster: int,
    n_reps: int = DEFAULT_N_REPS,
    n_terms: int = DEFAULT_N_TERMS,
    excerpt_chars: int = DEFAULT_EXCERPT_CHARS,
) -> None:
    """Write ``cluster_stats_<k_view>.json``: top terms + reps for every cluster at this K."""
    cluster_col = f"cluster_{k_view}"
    dist_col = f"dist_{k_train}"
    samples, reps, source_main_dirs = _scan_assignments(
        assignments,
        cluster_col=cluster_col,
        dist_col=dist_col,
        n_sample_per_cluster=n_sample_per_cluster,
        n_reps=n_reps,
    )

    needed: set[tuple[str, str, int]] = set()
    for entries in samples.values():
        for sn, bn, ri, _ in entries:
            needed.add((sn, bn, ri))
    for entries in reps.values():
        for sn, bn, ri, _ in entries:
            needed.add((sn, bn, ri))
    texts = _fetch_texts(source_main_dirs, needed, excerpt_chars)

    texts_by_cluster: dict[int, list[str]] = {c: [] for c in samples}
    for c, entries in samples.items():
        texts_by_cluster[c] = [t for sn, bn, ri, _ in entries if (t := texts.get((sn, bn, ri), ""))]

    top_terms = _ctfidf(texts_by_cluster, n_terms)

    clusters = []
    for c in sorted(samples):
        clusters.append(
            {
                "cluster_id": int(c),
                "n_sampled": len(samples[c]),
                "top_terms": top_terms.get(c, []),
                "representatives": [
                    {
                        "source": sn,
                        "shard": bn,
                        "row_idx": ri,
                        "dist": dist,
                        "excerpt": texts.get((sn, bn, ri), ""),
                    }
                    for sn, bn, ri, dist in reps.get(c, [])
                ],
            }
        )

    summary = {
        "k_train": int(k_train),
        "k_view": int(k_view),
        "n_clusters_seen": len(clusters),
        "n_sample_per_cluster": int(n_sample_per_cluster),
        "clusters": clusters,
    }

    local = os.path.join(tempfile.gettempdir(), f"cluster_stats_{k_view}.json")
    with open(local, "w") as f:
        json.dump(summary, f, indent=2)
    with open(local, "rb") as src, open_url(os.path.join(output_path, f"cluster_stats_{k_view}.json"), "wb") as dst:
        dst.write(src.read())
    os.remove(local)
    logger.info("Wrote cluster_stats_%d.json (%d clusters)", k_view, len(clusters))
