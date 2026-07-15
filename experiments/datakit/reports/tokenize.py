# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage report for tokenize: per-source attribute parquet (``{id, input_ids}``).

Doc and token totals come from each source's per-split ``tokenize/*`` counters;
the token-length histogram reads a bounded ``input_ids`` sample from the first
few sources' shards and bins the list lengths into power-of-two buckets.
"""

import os.path

from marin.processing.tokenize.attributes import TokenizedAttrData

from experiments.datakit.reports.common import StageReport, render_template, sample_rows, write_report

HIST_SOURCE_CAP = 8
HIST_ROWS_PER_SOURCE = 2000


def _log2_bins(lengths: list[int]) -> list[dict]:
    """Histogram of ``lengths`` into power-of-two bins (0, 1, 2-3, 4-7, ...)."""
    counts: dict[int, int] = {}
    for n in lengths:
        counts[n.bit_length()] = counts.get(n.bit_length(), 0) + 1
    bins = []
    for b in range(max(counts) + 1):
        lo = 0 if b == 0 else 1 << (b - 1)
        hi = (1 << b) - 1
        bins.append({"label": str(lo) if lo == hi else f"{lo}-{hi}", "count": counts.get(b, 0)})
    return bins


def tokenize_report(output_path: str, sources: dict[str, TokenizedAttrData], split: str) -> StageReport:
    """Render the tokenize stage report for ``split`` across all sources."""
    names = sorted(sources)
    rows = [
        {
            "name": name,
            "docs": sources[name].counters[split].get("tokenize/docs_out", 0),
            "tokens": sources[name].counters[split].get("tokenize/tokens_out", 0),
        }
        for name in names
    ]

    lengths: list[int] = []
    sampled = names[:HIST_SOURCE_CAP]
    for name in sampled:
        sample = sample_rows(sources[name].output_dirs[split], ["input_ids"], HIST_ROWS_PER_SOURCE)
        lengths.extend(len(r["input_ids"]) for r in sample)

    total_docs = sum(r["docs"] for r in rows)
    total_tokens = sum(r["tokens"] for r in rows)
    stats = {
        "total_docs": total_docs,
        "total_tokens": total_tokens,
        "avg_tokens_per_doc": round(total_tokens / total_docs, 2) if total_docs else 0.0,
        "n_sources": len(sources),
        "sampled_docs": len(lengths),
    }
    # Common parent of the per-source main dirs: char-wise prefix trimmed back to a path boundary.
    data_root = os.path.commonprefix([sources[name].source_main_dirs[split] for name in names]).rsplit("/", 1)[0]
    sampling = (
        f"docs/tokens from step counters (exact); token-length histogram from the first "
        f"{HIST_ROWS_PER_SOURCE} rows per source (file order) over {len(sampled)} of {len(sources)} sources"
    )
    data = {
        "meta": {
            "split": split,
            "tokenizer": ", ".join(sorted({s.tokenizer for s in sources.values()})),
            "tokenizer_backend": ", ".join(sorted({s.tokenizer_backend for s in sources.values()})),
            "data_root": data_root,
            "sampling": sampling,
        },
        "stats": stats,
        "sources": rows,
        "hist": {
            "bins": _log2_bins(lengths),
            "sampled_sources": len(sampled),
            "rows_per_source": HIST_ROWS_PER_SOURCE,
        },
    }
    page = render_template("tokenize.html", title="Datakit tokenize", data=data)
    return StageReport(html_path=write_report(output_path, page), stats=stats)
