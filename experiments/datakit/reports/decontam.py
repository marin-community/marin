# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage report for decontamination.

Aggregates each source's :class:`DeconAttributes` counters into a corpus contamination overview, and samples
the flagged-record sidecars for concrete examples: the excerpt of each record's best pair, its score, and the
evals of the items it holds.
"""

from collections import Counter

from marin.datakit.decon import DeconAttributes

from experiments.datakit.reports.common import StageReport, render_template, sample_rows, write_report

# Bounds on the report's parquet reads: flagged sidecars are sampled for the
# top-contaminated sources only, and each excerpt is truncated before embedding.
FLAGGED_SOURCE_LIMIT = 24
FLAGGED_ROWS_PER_SOURCE = 40
FLAGGED_TEXT_CHARS = 1200
TOP_EVALS = 15


def decontam_report(output_path: str, sources: dict[str, DeconAttributes]) -> StageReport:
    """Render the decontam stage report from per-source :class:`DeconAttributes`.

    Args:
        output_path: Directory the rendered ``report.html`` is written under.
        sources: Source name → that source's decon artifact.
    """
    counts: list[tuple[float, str, int, int]] = []
    for name, attrs in sources.items():
        contaminated = int(attrs.counters.get("decon/contaminated", 0))
        clean = int(attrs.counters.get("decon/clean", 0))
        counts.append((contaminated / max(contaminated + clean, 1), name, contaminated, clean))
    counts.sort(key=lambda row: row[0], reverse=True)

    hits: Counter[str] = Counter()
    flagged = []
    n_flagged_sampled = 0
    for _, name, _, _ in counts[:FLAGGED_SOURCE_LIMIT]:
        rows = sample_rows(
            sources[name].flagged_output_dir, ["id", "text", "max_score", "matched_evals"], FLAGGED_ROWS_PER_SOURCE
        )
        for row in rows:
            hits.update(row["matched_evals"])
        if not rows:
            continue
        n_flagged_sampled += len(rows)
        flagged.append(
            {
                "source": name,
                "rows": [
                    {
                        "id": r["id"],
                        "text": r["text"][:FLAGGED_TEXT_CHARS],
                        "max_score": r["max_score"],
                        "evals": list(r["matched_evals"]),
                    }
                    for r in rows
                ],
            }
        )

    contaminated_docs = sum(c for _, _, c, _ in counts)
    clean_docs = sum(k for _, _, _, k in counts)
    total_docs = contaminated_docs + clean_docs
    stats = {
        "contamination_rate": contaminated_docs / max(total_docs, 1),
        "contaminated_docs": contaminated_docs,
        "clean_docs": clean_docs,
        "total_docs": total_docs,
        "n_sources": len(sources),
        "n_flagged_sampled": n_flagged_sampled,
        "n_evals_hit": len(hits),
    }
    data = {
        "meta": {
            "sampling": (
                f"contamination rates from step counters (exact); flagged examples from the per-shard "
                f"reservoir sample side output, first {FLAGGED_ROWS_PER_SOURCE} rows per source (file order) "
                f"for the {FLAGGED_SOURCE_LIMIT} most-contaminated sources; eval attribution covers only "
                f"those sampled records"
            ),
        },
        "stats": stats,
        "sources": [{"name": n, "contaminated": c, "clean": k, "rate": r} for r, n, c, k in counts],
        "flagged": flagged,
        "eval_hits": [{"eval": e, "hits": n} for e, n in hits.most_common(TOP_EVALS)],
    }
    page = render_template("decontam.html", title="Datakit decontam", data=data)
    return StageReport(html_path=write_report(output_path, page), stats=stats)
