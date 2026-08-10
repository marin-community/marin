# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage report for decontamination.

Aggregates each source's :class:`DeconAttributes` counters into a corpus
contamination overview, samples the flagged-doc sidecars for concrete
examples, and joins the sampled matched hashes against the shared
``hash → eval_id`` sidecar to attribute contamination to specific eval
records.
"""

from collections import Counter

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from marin.datakit.decon import DeconAttributes
from rigging.filesystem import StoragePath

from experiments.datakit.reports.common import StageReport, render_template, sample_rows, write_report

# Bounds on the report's parquet reads: flagged sidecars are sampled for the
# top-contaminated sources only, and each doc's text is truncated before embedding.
FLAGGED_SOURCE_LIMIT = 24
FLAGGED_ROWS_PER_SOURCE = 40
FLAGGED_TEXT_CHARS = 1200
TOP_EVALS = 15


def _eval_hits(index_path: str, hash_counts: Counter[int]) -> Counter[str]:
    """Stream the ``hash → eval_id`` sidecar and count sampled-hash hits per eval record.

    The sidecar holds ~20M rows; each batch is filtered vectorized against the
    (small) sampled-hash set before any Python-side boxing.
    """
    hits: Counter[str] = Counter()
    if not hash_counts:
        return hits
    wanted = pa.array(list(hash_counts), type=pa.uint64())
    with StoragePath(index_path).open("rb") as fh:
        for batch in pq.ParquetFile(fh).iter_batches(columns=["hash", "eval_id"]):
            matched = batch.filter(pc.is_in(batch.column("hash"), value_set=wanted))
            for h, eval_id in zip(
                matched.column("hash").to_pylist(), matched.column("eval_id").to_pylist(), strict=True
            ):
                hits[eval_id] += hash_counts[h]
    return hits


def decontam_report(output_path: str, sources: dict[str, DeconAttributes]) -> StageReport:
    """Render the decontam stage report from per-source :class:`DeconAttributes`.

    Args:
        output_path: Directory the rendered ``report.html`` is written under.
        sources: Source name → that source's decon artifact. All artifacts share
            one eval bloom, so the ``eval_hash_index_path`` of any of them is the
            corpus-wide attribution sidecar.
    """
    counts: list[tuple[float, str, int, int]] = []
    for name, attrs in sources.items():
        contaminated = int(attrs.counters.get("decon/contaminated", 0))
        clean = int(attrs.counters.get("decon/clean", 0))
        counts.append((contaminated / (contaminated + clean), name, contaminated, clean))
    counts.sort(key=lambda row: row[0], reverse=True)

    hash_counts: Counter[int] = Counter()
    flagged = []
    n_flagged_sampled = 0
    for _, name, _, _ in counts[:FLAGGED_SOURCE_LIMIT]:
        rows = sample_rows(
            sources[name].flagged_output_dir, ["id", "text", "max_overlap", "matched_hashes"], FLAGGED_ROWS_PER_SOURCE
        )
        for row in rows:
            hash_counts.update(row["matched_hashes"])
        if not rows:
            continue
        n_flagged_sampled += len(rows)
        flagged.append(
            {
                "source": name,
                "rows": [
                    {"id": r["id"], "text": r["text"][:FLAGGED_TEXT_CHARS], "max_overlap": r["max_overlap"]}
                    for r in rows
                ],
            }
        )

    index_path = next(iter(sources.values())).eval_hash_index_path
    hits = _eval_hits(index_path, hash_counts)

    contaminated_docs = sum(c for _, _, c, _ in counts)
    clean_docs = sum(k for _, _, _, k in counts)
    total_docs = contaminated_docs + clean_docs
    stats = {
        "contamination_rate": contaminated_docs / total_docs,
        "contaminated_docs": contaminated_docs,
        "clean_docs": clean_docs,
        "total_docs": total_docs,
        "n_sources": len(sources),
        "n_flagged_sampled": n_flagged_sampled,
        "n_evals_hit": len(hits),
    }
    data = {
        "meta": {
            "eval_hash_index_path": index_path,
            "sampling": (
                f"contamination rates from step counters (exact); flagged examples from the per-shard "
                f"reservoir sample side output, first {FLAGGED_ROWS_PER_SOURCE} rows per source (file order) "
                f"for the {FLAGGED_SOURCE_LIMIT} most-contaminated sources; eval attribution covers only "
                f"those sampled docs"
            ),
        },
        "stats": stats,
        "sources": [{"name": n, "contaminated": c, "clean": k, "rate": r} for r, n, c, k in counts],
        "flagged": flagged,
        "eval_hits": [{"eval_id": e, "hits": n} for e, n in hits.most_common(TOP_EVALS)],
    }
    page = render_template("decontam.html", title="Datakit decontam", data=data)
    return StageReport(html_path=write_report(output_path, page), stats=stats)
