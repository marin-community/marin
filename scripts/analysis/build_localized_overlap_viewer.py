# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build a static HTML viewer for exact localized decontamination overlaps.

The viewer reads one or more ``localized_overlaps.parquet`` files from
``localize_decon_contaminated_docs.py`` and writes a self-contained HTML page.
It does not fetch full Nemotron documents; each localized artifact already
contains source/eval snippets, coordinates, and sampled shared 13-word n-grams.

Run:

    uv run python scripts/analysis/build_localized_overlap_viewer.py

Default output:

    scratch/math_benchmark_localized_overlap_viewer.html
    gs://marin-us-east5/scratch/ahmed/math-decontamin-dir/decon_nemotron_cc_math_4plus/localized_overlap_viewer.html
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fsspec
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

DECON_OUTPUT_ROOT = "gs://marin-us-east5/scratch/ahmed/math-decontamin-dir/decon_nemotron_cc_math_4plus"
DEFAULT_LOCAL_OUTPUT = "scratch/math_benchmark_localized_overlap_viewer.html"
DEFAULT_GCS_OUTPUT = f"{DECON_OUTPUT_ROOT}/localized_overlap_viewer.html"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    label: str
    eval_label: str
    input_path: str


DEFAULT_DATASETS = [
    DatasetSpec(
        key="math500__test",
        label="MATH500 test",
        eval_label="Math500 eval paragraph",
        input_path=f"{DECON_OUTPUT_ROOT}/math500__test/localized_overlaps/localized_overlaps.parquet",
    ),
    DatasetSpec(
        key="gsm8k__main__train",
        label="GSM8K train",
        eval_label="GSM8K train eval paragraph",
        input_path=f"{DECON_OUTPUT_ROOT}/gsm8k__main__train/localized_overlaps/localized_overlaps.parquet",
    ),
    DatasetSpec(
        key="gsm8k__main__test",
        label="GSM8K test",
        eval_label="GSM8K test eval paragraph",
        input_path=f"{DECON_OUTPUT_ROOT}/gsm8k__main__test/localized_overlaps/localized_overlaps.parquet",
    ),
    DatasetSpec(
        key="aime24__train",
        label="AIME24 train",
        eval_label="AIME24 eval paragraph",
        input_path=f"{DECON_OUTPUT_ROOT}/aime24__train/localized_overlaps/localized_overlaps.parquet",
    ),
]

VIEWER_COLUMNS = [
    "doc_id",
    "source_parquet",
    "partition_id",
    "row_index_in_partition",
    "datakit_max_overlap",
    "eval_id",
    "source_paragraph_index",
    "source_char_start",
    "source_char_end",
    "source_token_count",
    "source_ngram_count",
    "record_intersection_count",
    "record_source_containment",
    "record_source_unique_containment",
    "record_eval_containment",
    "record_jaccard",
    "best_eval_paragraph_index",
    "best_eval_char_start",
    "best_eval_char_end",
    "best_eval_source_containment",
    "best_eval_jaccard",
    "source_snippet",
    "eval_snippet",
    "shared_ngrams",
]


def read_rows(path: str, max_rows: int | None) -> list[dict[str, Any]]:
    with fsspec.open(path, "rb") as handle:
        table = pq.read_table(handle, columns=VIEWER_COLUMNS)
    if max_rows is not None:
        table = table.slice(0, max_rows)
    return table.to_pylist()


def _round(value: Any, digits: int = 4) -> float:
    return round(float(value), digits)


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    source_parquet = str(row["source_parquet"])
    return {
        "doc": row["doc_id"],
        "eval": row["eval_id"],
        "sourceParquet": source_parquet,
        "sourceShard": source_parquet.rsplit("/", 1)[-1],
        "partition": int(row["partition_id"]),
        "row": int(row["row_index_in_partition"]),
        "datakit": _round(row["datakit_max_overlap"]),
        "sourceParagraph": int(row["source_paragraph_index"]),
        "sourceSpan": [int(row["source_char_start"]), int(row["source_char_end"])],
        "sourceTokens": int(row["source_token_count"]),
        "sourceNgrams": int(row["source_ngram_count"]),
        "evalParagraph": int(row["best_eval_paragraph_index"]),
        "evalSpan": [int(row["best_eval_char_start"]), int(row["best_eval_char_end"])],
        "intersection": int(row["record_intersection_count"]),
        "sourceContainment": _round(row["record_source_containment"], 6),
        "sourceUniqueContainment": _round(row["record_source_unique_containment"], 6),
        "evalContainment": _round(row["record_eval_containment"], 6),
        "jaccard": _round(row["record_jaccard"], 6),
        "bestEvalSourceContainment": _round(row["best_eval_source_containment"], 6),
        "bestEvalJaccard": _round(row["best_eval_jaccard"], 6),
        "source": row["source_snippet"] or "",
        "evalText": row["eval_snippet"] or "",
        "shared": [str(ngram) for ngram in row.get("shared_ngrams") or []],
    }


def summary_for_rows(rows: list[dict[str, Any]], *, input_path: str, row_limit: int | None) -> dict[str, Any]:
    eval_counts = Counter(str(row["eval_id"]) for row in rows)
    return {
        "input": input_path,
        "rowLimit": row_limit,
        "rows": len(rows),
        "docs": len({row["doc_id"] for row in rows}),
        "evalRecords": len(eval_counts),
        "maxJaccard": max((float(row["record_jaccard"]) for row in rows), default=0.0),
        "maxSourceContainment": max((float(row["record_source_containment"]) for row in rows), default=0.0),
        "topEvalIds": eval_counts.most_common(20),
    }


def viewer_payload(input_path: str, max_rows: int | None) -> dict[str, Any]:
    rows = read_rows(input_path, max_rows)
    compact = [compact_row(row) for row in rows]
    return {
        "summary": summary_for_rows(rows, input_path=input_path, row_limit=max_rows),
        "rows": compact,
    }


def dataset_payload(spec: DatasetSpec, max_rows: int | None) -> dict[str, Any]:
    payload = viewer_payload(spec.input_path, max_rows)
    return {
        "key": spec.key,
        "label": spec.label,
        "evalLabel": spec.eval_label,
        "summary": payload["summary"],
        "rows": payload["rows"],
    }


def multi_viewer_payload(specs: Sequence[DatasetSpec], max_rows: int | None) -> dict[str, Any]:
    datasets = [dataset_payload(spec, max_rows) for spec in specs]
    return {
        "schemaVersion": 2,
        "defaultDataset": datasets[0]["key"] if datasets else "",
        "datasets": datasets,
    }


HTML_TEMPLATE = r"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Math benchmark localized overlap viewer</title>
<style>
  :root {
    color-scheme: light;
    --bg: #f7f7f4;
    --panel: #ffffff;
    --line: #d8d7d1;
    --muted: #66675f;
    --text: #24251f;
    --accent: #006d77;
    --accent-soft: #d8f0f1;
    --mark: #ffe08a;
    --mark-border: #d59b1a;
    --danger: #8f2d2d;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    min-height: 100vh;
    font: 14px/1.45 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: var(--bg);
    color: var(--text);
    display: grid;
    grid-template-columns: minmax(320px, 400px) 1fr;
  }
  aside {
    min-height: 100vh;
    border-right: 1px solid var(--line);
    background: #eeeeea;
    display: grid;
    grid-template-rows: auto auto 1fr;
  }
  .summary, .filters {
    padding: 14px 16px;
    border-bottom: 1px solid var(--line);
  }
  h1 {
    font-size: 18px;
    line-height: 1.2;
    margin: 0 0 10px;
    letter-spacing: 0;
  }
  .metrics {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 8px;
  }
  .metric {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 6px;
    padding: 8px;
  }
  .metric b { display: block; font-size: 16px; }
  .metric span { color: var(--muted); font-size: 12px; }
  label {
    display: block;
    color: var(--muted);
    font-size: 12px;
    margin: 10px 0 4px;
  }
  input, select {
    width: 100%;
    border: 1px solid var(--line);
    border-radius: 6px;
    background: var(--panel);
    color: var(--text);
    padding: 8px 9px;
    font: inherit;
  }
  .list {
    overflow: auto;
    min-height: 0;
  }
  .item {
    padding: 10px 12px;
    border-bottom: 1px solid var(--line);
    cursor: pointer;
    background: transparent;
  }
  .item:hover { background: #f8f8f5; }
  .item.active {
    background: var(--accent-soft);
    border-left: 4px solid var(--accent);
    padding-left: 8px;
  }
  .item-title {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    font-weight: 650;
  }
  .item small {
    display: block;
    color: var(--muted);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  main {
    min-width: 0;
    min-height: 100vh;
    display: grid;
    grid-template-rows: auto auto 1fr;
  }
  .topbar {
    position: relative;
    padding: 14px 18px;
    padding-right: 320px;
    border-bottom: 1px solid var(--line);
    background: var(--panel);
  }
  .topbar h2 {
    font-size: 16px;
    margin: 0 0 8px;
    letter-spacing: 0;
  }
  .meta-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 8px;
  }
  .dataset-switch {
    position: absolute;
    top: 10px;
    right: 18px;
    width: min(280px, calc(100% - 36px));
  }
  .dataset-switch label {
    margin: 0 0 4px;
  }
  .meta {
    border: 1px solid var(--line);
    border-radius: 6px;
    padding: 8px;
    min-width: 0;
    background: #fbfbf8;
  }
  .meta span {
    color: var(--muted);
    display: block;
    font-size: 11px;
  }
  .meta code {
    font: 12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    word-break: break-all;
  }
  .shared {
    padding: 12px 18px;
    border-bottom: 1px solid var(--line);
    background: #fbfbf8;
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    max-height: 114px;
    overflow: auto;
  }
  .chip {
    border: 1px solid var(--line);
    border-radius: 999px;
    background: var(--panel);
    padding: 4px 8px;
    font: 12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  }
  .panes {
    min-height: 0;
    display: grid;
    grid-template-columns: 1fr 1fr;
  }
  .pane {
    min-width: 0;
    min-height: 0;
    overflow: auto;
    padding: 18px;
    border-right: 1px solid var(--line);
    background: var(--panel);
  }
  .pane:last-child { border-right: 0; }
  .pane h3 {
    margin: 0 0 12px;
    font-size: 14px;
    letter-spacing: 0;
  }
  pre {
    margin: 0;
    white-space: pre-wrap;
    word-break: break-word;
    font: 13px/1.55 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  }
  mark {
    background: var(--mark);
    border-bottom: 2px solid var(--mark-border);
    color: var(--text);
    padding: 0 1px;
  }
  .empty {
    color: var(--muted);
    padding: 18px;
  }
  @media (max-width: 900px) {
    body { grid-template-columns: 1fr; }
    aside { min-height: 44vh; }
    .topbar { padding-right: 18px; padding-top: 78px; }
    .dataset-switch { left: 18px; right: 18px; width: auto; }
    .panes, .meta-grid { grid-template-columns: 1fr; }
  }
</style>
<body>
  <aside>
    <section class="summary">
      <h1>Benchmark Localized Overlaps</h1>
      <div class="metrics">
        <div class="metric"><b id="metricRows"></b><span>localized rows</span></div>
        <div class="metric"><b id="metricDocs"></b><span>source docs</span></div>
        <div class="metric"><b id="metricEvals"></b><span>eval records</span></div>
        <div class="metric"><b id="metricMaxJ"></b><span>max Jaccard</span></div>
      </div>
    </section>
    <section class="filters">
      <label for="query">Search doc/eval/source text</label>
      <input id="query" placeholder="doc id, eval id, phrase">
      <label for="evalFilter">Eval record</label>
      <select id="evalFilter"><option value="">All eval records</option></select>
      <label for="sortMode">Sort</label>
      <select id="sortMode">
        <option value="sourceContainment">Source containment</option>
        <option value="jaccard">Jaccard</option>
        <option value="eval">Eval id</option>
        <option value="doc">Doc id</option>
      </select>
    </section>
    <section class="list" id="list"></section>
  </aside>
  <main>
    <section class="topbar">
      <div class="dataset-switch">
        <label for="datasetSelect">Dataset</label>
        <select id="datasetSelect"></select>
      </div>
      <h2 id="title">Select an overlap</h2>
      <div class="meta-grid" id="meta"></div>
    </section>
    <section class="shared" id="shared"></section>
    <section class="panes">
      <article class="pane">
        <h3>Nemotron source paragraph</h3>
        <pre id="sourcePane"></pre>
      </article>
      <article class="pane">
        <h3 id="evalPaneTitle">Eval paragraph</h3>
        <pre id="evalPane"></pre>
      </article>
    </section>
  </main>
<script>
const PAYLOAD = __PAYLOAD_JSON__;
const datasets = PAYLOAD.datasets ?? [{
  key: "default",
  label: "Dataset",
  evalLabel: "Eval paragraph",
  summary: PAYLOAD.summary,
  rows: PAYLOAD.rows
}];
const state = { datasetIndex: 0, filtered: [], selected: 0 };

function dataset() {
  return datasets[state.datasetIndex];
}

function rows() {
  return dataset().rows;
}

function summary() {
  return dataset().summary;
}

function fmt(n, digits = 3) {
  if (typeof n !== "number") return n;
  return n.toLocaleString(undefined, { maximumFractionDigits: digits });
}

function escapeHtml(text) {
  return String(text ?? "").replace(/[&<>"']/g, ch => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;"
  }[ch]));
}

function findRanges(text, phrases) {
  const ranges = [];
  const lower = text.toLowerCase();
  const sorted = [...phrases].filter(Boolean).sort((a, b) => b.length - a.length);
  for (const phrase of sorted) {
    const needle = phrase.toLowerCase();
    let start = 0;
    while (needle && true) {
      const idx = lower.indexOf(needle, start);
      if (idx < 0) break;
      const end = idx + phrase.length;
      if (!ranges.some(([a, b]) => idx < b && end > a)) ranges.push([idx, end]);
      start = idx + Math.max(1, phrase.length);
    }
  }
  return ranges.sort((a, b) => a[0] - b[0]);
}

function highlighted(text, phrases) {
  const ranges = findRanges(text, phrases);
  if (!ranges.length) return escapeHtml(text);
  const out = [];
  let pos = 0;
  for (const [start, end] of ranges) {
    out.push(escapeHtml(text.slice(pos, start)));
    out.push("<mark>" + escapeHtml(text.slice(start, end)) + "</mark>");
    pos = end;
  }
  out.push(escapeHtml(text.slice(pos)));
  return out.join("");
}

function renderMetrics() {
  document.getElementById("metricRows").textContent = fmt(summary().rows, 0);
  document.getElementById("metricDocs").textContent = fmt(summary().docs, 0);
  document.getElementById("metricEvals").textContent = fmt(summary().evalRecords, 0);
  document.getElementById("metricMaxJ").textContent = fmt(summary().maxJaccard, 4);
}

function setupDatasetSelect() {
  const select = document.getElementById("datasetSelect");
  select.innerHTML = "";
  datasets.forEach((ds, index) => {
    const option = document.createElement("option");
    option.value = ds.key;
    option.textContent = ds.label;
    if (index === state.datasetIndex) option.selected = true;
    select.appendChild(option);
  });
  select.onchange = () => {
    const selected = datasets.findIndex(ds => ds.key === select.value);
    state.datasetIndex = selected >= 0 ? selected : 0;
    state.selected = 0;
    document.getElementById("query").value = "";
    renderMetrics();
    setupEvalFilter();
    applyFilters();
  };
}

function setupEvalFilter() {
  const select = document.getElementById("evalFilter");
  select.innerHTML = '<option value="">All eval records</option>';
  const ids = [...new Set(rows().map(r => r.eval))].sort();
  for (const id of ids) {
    const option = document.createElement("option");
    option.value = id;
    option.textContent = id;
    select.appendChild(option);
  }
}

function applyFilters() {
  const query = document.getElementById("query").value.trim().toLowerCase();
  const evalId = document.getElementById("evalFilter").value;
  const sortMode = document.getElementById("sortMode").value;
  state.filtered = rows().filter(row => {
    if (evalId && row.eval !== evalId) return false;
    if (!query) return true;
    return row.doc.toLowerCase().includes(query)
      || row.eval.toLowerCase().includes(query)
      || row.source.toLowerCase().includes(query)
      || row.evalText.toLowerCase().includes(query);
  });
  state.filtered.sort((a, b) => {
    if (sortMode === "jaccard") return b.jaccard - a.jaccard || b.sourceContainment - a.sourceContainment;
    if (sortMode === "eval") return a.eval.localeCompare(b.eval) || b.sourceContainment - a.sourceContainment;
    if (sortMode === "doc") return a.doc.localeCompare(b.doc) || b.sourceContainment - a.sourceContainment;
    return b.sourceContainment - a.sourceContainment || b.jaccard - a.jaccard;
  });
  state.selected = 0;
  renderList();
  renderSelected();
}

function renderList() {
  const list = document.getElementById("list");
  list.innerHTML = "";
  if (!state.filtered.length) {
    list.innerHTML = '<div class="empty">No overlaps match the filters.</div>';
    return;
  }
  const capped = state.filtered.slice(0, 1000);
  capped.forEach((row, index) => {
    const item = document.createElement("div");
    const evalName = escapeHtml(row.eval.split("/").slice(-2).join("/"));
    const sourcePreview = escapeHtml(row.source.slice(0, 130));
    item.className = "item" + (index === state.selected ? " active" : "");
    item.innerHTML = `
      <div class="item-title"><span>${evalName}</span><span>${fmt(row.sourceContainment, 3)}</span></div>
      <small>J=${fmt(row.jaccard, 4)} | ${escapeHtml(row.doc.slice(0, 16))} | paragraph ${row.sourceParagraph}</small>
      <small>${sourcePreview}</small>`;
    item.onclick = () => {
      state.selected = index;
      renderList();
      renderSelected();
    };
    list.appendChild(item);
  });
  if (state.filtered.length > capped.length) {
    const more = document.createElement("div");
    more.className = "empty";
    more.textContent =
      `Showing first ${capped.length.toLocaleString()} of ` +
      `${state.filtered.length.toLocaleString()} filtered rows. Narrow the filter to see more.`;
    list.appendChild(more);
  }
}

function metaCell(label, value) {
  return `<div class="meta"><span>${label}</span><code>${escapeHtml(value)}</code></div>`;
}

function renderSelected() {
  const row = state.filtered[state.selected];
  if (!row) {
    document.getElementById("title").textContent = `${dataset().label}: no overlap selected`;
    document.getElementById("meta").innerHTML = "";
    document.getElementById("shared").innerHTML = "";
    document.getElementById("sourcePane").innerHTML = "";
    document.getElementById("evalPane").innerHTML = "";
    return;
  }
  document.getElementById("title").textContent = `${dataset().label}: ${row.doc} <-> ${row.eval}`;
  document.getElementById("evalPaneTitle").textContent = dataset().evalLabel;
  document.getElementById("meta").innerHTML = [
    metaCell("source parquet", row.sourceShard),
    metaCell("row / paragraph", `${row.row} / ${row.sourceParagraph}`),
    metaCell("source span", `${row.sourceSpan[0]}-${row.sourceSpan[1]}`),
    metaCell("eval paragraph / span", `${row.evalParagraph} / ${row.evalSpan[0]}-${row.evalSpan[1]}`),
    metaCell("source containment", fmt(row.sourceContainment, 6)),
    metaCell("source unique containment", fmt(row.sourceUniqueContainment, 6)),
    metaCell("eval containment", fmt(row.evalContainment, 6)),
    metaCell("jaccard", fmt(row.jaccard, 6)),
  ].join("");
  document.getElementById("shared").innerHTML = row.shared.length
    ? row.shared.map(s => `<span class="chip">${escapeHtml(s)}</span>`).join("")
    : '<span class="empty">No shared n-gram sample stored.</span>';
  document.getElementById("sourcePane").innerHTML = highlighted(row.source, row.shared);
  document.getElementById("evalPane").innerHTML = highlighted(row.evalText, row.shared);
}

setupDatasetSelect();
renderMetrics();
setupEvalFilter();
document.getElementById("query").addEventListener("input", applyFilters);
document.getElementById("evalFilter").addEventListener("change", applyFilters);
document.getElementById("sortMode").addEventListener("change", applyFilters);
applyFilters();
</script>
</body>
"""


def render_html(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return HTML_TEMPLATE.replace("__PAYLOAD_JSON__", data)


def write_text(path: str, text: str) -> None:
    if "://" in path:
        fs, _, paths = fsspec.get_fs_token_paths(path)
        parent = os.path.dirname(paths[0])
        if parent:
            fs.makedirs(parent, exist_ok=True)
        with fs.open(paths[0], "w", encoding="utf-8") as handle:
            handle.write(text)
        return
    local_path = Path(path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text(text, encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=None, help="Optional single localized_overlaps.parquet input.")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset spec as key=label=eval_label=input_path. If omitted, the four benchmark defaults are used.",
    )
    parser.add_argument("--local-output", default=DEFAULT_LOCAL_OUTPUT)
    parser.add_argument("--gcs-output", default=DEFAULT_GCS_OUTPUT)
    parser.add_argument("--max-rows", type=int, default=0, help="0 means include all rows.")
    parser.add_argument("--no-gcs-upload", action="store_true")
    return parser.parse_args(argv)


def parse_dataset_spec(value: str) -> DatasetSpec:
    parts = value.split("=", 3)
    if len(parts) != 4:
        raise ValueError(f"dataset spec must be key=label=eval_label=input_path, got {value!r}")
    key, label, eval_label, input_path = parts
    return DatasetSpec(key=key, label=label, eval_label=eval_label, input_path=input_path)


def selected_specs(args: argparse.Namespace) -> list[DatasetSpec]:
    if args.dataset:
        return [parse_dataset_spec(value) for value in args.dataset]
    if args.input:
        return [
            DatasetSpec(
                key="single",
                label="Localized overlaps",
                eval_label="Eval paragraph",
                input_path=args.input,
            )
        ]
    return list(DEFAULT_DATASETS)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    max_rows = None if args.max_rows == 0 else args.max_rows
    payload = multi_viewer_payload(selected_specs(args), max_rows)
    html_text = render_html(payload)
    write_text(args.local_output, html_text)
    logger.info("wrote local viewer: %s", args.local_output)
    if not args.no_gcs_upload:
        write_text(args.gcs_output, html_text)
        logger.info("wrote GCS viewer: %s", args.gcs_output)


if __name__ == "__main__":
    main()
