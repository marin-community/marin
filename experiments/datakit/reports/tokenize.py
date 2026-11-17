# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage report for tokenize: per-source attribute parquet (``{id, chunk_index, input_ids}``).

Doc and token totals come from each source's per-split ``tokenize/*`` counters;
the token-length histogram reads a bounded ``input_ids`` sample from the first
few sources' shards and bins the list lengths into power-of-two buckets. The
histogram bins rows, so a document that the tokenizer split across rows appears
as its chunks. The totals above it count documents.
"""

import os.path

import pyarrow.compute as pc
from marin.processing.tokenize.attributes import TokenizedAttrData

from experiments.datakit.reports.common import StageReport, iter_batches, render_template, write_report

HIST_SOURCE_CAP = 8
HIST_ROWS_PER_SOURCE = 2000
# Small batches bound driver memory when a sampled row holds a very long document.
HIST_BATCH_ROWS = 64


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


def _sample_lengths(directory: str, limit: int) -> list[int]:
    """Token counts of up to ``limit`` rows, read without converting ``input_ids`` to Python lists."""
    lengths: list[int] = []
    for batch in iter_batches(directory, ["input_ids"], batch_size=HIST_BATCH_ROWS):
        lengths.extend(pc.list_value_length(batch.column("input_ids")).to_pylist()[: limit - len(lengths)])
        if len(lengths) >= limit:
            break
    return lengths


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
        lengths.extend(_sample_lengths(sources[name].output_dirs[split], HIST_ROWS_PER_SOURCE))

    total_docs = sum(r["docs"] for r in rows)
    total_tokens = sum(r["tokens"] for r in rows)
    stats = {
        "total_docs": total_docs,
        "total_tokens": total_tokens,
        "avg_tokens_per_doc": round(total_tokens / total_docs, 2) if total_docs else 0.0,
        "n_sources": len(sources),
        "sampled_docs": len(lengths),
    }
    # Common parent of the source keys: char-wise prefix trimmed back to a path boundary.
    data_root = os.path.commonprefix([sources[name].source_keys[split] for name in names]).rsplit("/", 1)[0]
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
