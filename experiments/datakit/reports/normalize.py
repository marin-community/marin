# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage report for the per-source normalize outputs.

Aggregates each source's :class:`NormalizedData` counters into a per-source
funnel table (records in, empty-text filtered, unique docs out, exact dups,
whitespace compactions) plus corpus totals, and spot-checks the main parquet
output by previewing the first ``SPOT_CHECK_ROWS_PER_SOURCE`` docs of every
source (file order), text clipped to ``SPOT_CHECK_TEXT_CHARS`` chars for
embedding; the page also shows a text-length histogram over the previewed docs.
The preview is a bounded head read (``sample_rows`` stops at the limit), so it
scans one row group per source rather than the full output.
"""

import os.path

from marin.datakit.normalize import NormalizedData

from experiments.datakit.reports.common import StageReport, render_template, sample_rows, write_report

SPOT_CHECK_ROWS_PER_SOURCE = 64
SPOT_CHECK_TEXT_CHARS = 64_000  # 64 KB/doc: show most docs in full, clip only long outliers


def normalize_report(output_path: str, sources: dict[str, NormalizedData]) -> StageReport:
    # Sample/smoke runs read pre-sampled source dirs whose counters come from the
    # sampler (sampler/*), not the normalize job (normalize/*, zephyr/*). Only the
    # sampled doc count is recorded there; the funnel counters are absent (they read
    # as 0), so the report hides them and shows a warning instead of fake zeros.
    is_sample = bool(sources) and all("sampler/rows_out" in d.counters for d in sources.values())
    rows = [
        {
            "source": name,
            "records_in": d.counters.get("zephyr/records_in", 0),
            "empty_filtered": d.counters.get("normalize/empty_text_filtered", 0),
            "docs_out": d.counters.get("normalize/unique_records_out", d.counters.get("sampler/rows_out", 0)),
            "dups": d.counters.get("normalize/duplicate_records_out", 0),
            "ws_compacted": d.counters.get("datakit_normalize_compacted_whitespace", 0),
            "bytes_processed": d.counters.get("zephyr/bytes_processed", 0),
            "main_output_dir": d.main_output_dir,
        }
        for name, d in sorted(sources.items())
    ]

    sampled_sources = sorted(sources)
    samples = {
        name: [
            {"id": r["id"], "n_chars": len(r["text"]), "text": r["text"][:SPOT_CHECK_TEXT_CHARS]}
            for r in sample_rows(sources[name].main_output_dir, ["id", "text"], SPOT_CHECK_ROWS_PER_SOURCE)
        ]
        for name in sampled_sources
    }

    stats = {
        "n_sources": len(sources),
        "records_in": sum(r["records_in"] for r in rows),
        "docs_out": sum(r["docs_out"] for r in rows),
        "dups": sum(r["dups"] for r in rows),
        "empty_filtered": sum(r["empty_filtered"] for r in rows),
        "ws_compacted": sum(r["ws_compacted"] for r in rows),
    }
    # Common parent of the per-source main dirs: char-wise prefix trimmed back to a path boundary.
    data_root = os.path.commonprefix([r["main_output_dir"] for r in rows]).rsplit("/", 1)[0]
    sampling = (
        f"per-source funnel from step counters (exact where present); spot-check docs + length "
        f"histogram from the first {SPOT_CHECK_ROWS_PER_SOURCE} docs per source (file order), "
        f"all {len(sources)} sources"
    )
    data = {
        "meta": {"data_root": data_root, "sampling": sampling, "is_sample": is_sample},
        "stats": stats,
        "rows": rows,
        "samples": samples,
        "spot_check": {
            "n_sampled": len(sampled_sources),
            "rows_per_source": SPOT_CHECK_ROWS_PER_SOURCE,
            "text_chars": SPOT_CHECK_TEXT_CHARS,
        },
    }
    page = render_template("normalize.html", title="Datakit normalize", data=data)
    return StageReport(html_path=write_report(output_path, page), stats=stats)
