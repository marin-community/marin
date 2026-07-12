# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 3 (optional): render a single-file HTML report of N duplicate-cluster examples.

Reads the per-cluster examples parquet produced by :mod:`fetch_cluster_texts`,
samples N clusters (seeded; prefers ones with 2-4 members for readability),
and writes a self-contained HTML file with all data embedded inline. No
external dependencies — open it directly in any browser, share via Slack,
attach to a doc, etc.

Defaults read the GCS artifact and write ``/tmp/dup_examples.html`` locally.
Pass ``--upload gs://...`` to also push the rendered HTML to a bucket.

Run locally (small parquet, cheap cross-region read) or on iris in us-central2:

    uv run python experiments/datakit/dedup/ops/render_cluster_report.py
"""

import argparse
import html
import logging
import random

import pyarrow.parquet as pq
from rigging.filesystem import StoragePath
from rigging.log_setup import configure_logging

logger = logging.getLogger(__name__)

DEFAULT_INPUT = "gs://marin-us-central2/tmp/ttl=7d/rav/dedup_examples_dabe67c2/examples_zephyr"
DEFAULT_OUTPUT = "/tmp/dup_examples.html"


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Duplicate cluster examples</title>
<style>
:root {
  --bg: #fafafa;
  --card-bg: #fff;
  --border: #ddd;
  --canonical-bg: #e7f5ee;
  --canonical-border: #2e7d32;
  --duplicate-bg: #fff7e6;
  --duplicate-border: #b97000;
  --badge-bg: #eef2f7;
  --text: #222;
  --muted: #666;
}
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  margin: 0; padding: 16px 24px; background: var(--bg); color: var(--text);
}
h1 { margin: 0 0 6px; }
.summary { color: var(--muted); margin-bottom: 16px; font-size: 14px; }
.filters {
  position: sticky; top: 0; z-index: 10; background: var(--bg);
  padding: 8px 0; margin-bottom: 16px; border-bottom: 1px solid var(--border);
  display: flex; gap: 12px; flex-wrap: wrap;
}
.filters input, .filters select {
  padding: 6px 10px; font-size: 14px;
  border: 1px solid var(--border); border-radius: 4px;
}
.filters input[type=search] { width: 320px; }
.filters .count { color: var(--muted); align-self: center; font-size: 14px; }
.cluster {
  background: var(--card-bg); border: 1px solid var(--border); border-radius: 6px;
  margin-bottom: 14px; padding: 12px 14px;
}
.cluster-header {
  display: flex; align-items: baseline; gap: 12px;
  margin-bottom: 8px; flex-wrap: wrap;
}
.cluster-id {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12px; color: var(--muted);
}
.cluster-size { font-weight: 600; }
.sources-line { color: var(--muted); font-size: 13px; }
.member {
  border-left: 4px solid var(--border);
  padding: 6px 10px; margin: 8px 0;
  background: #fcfcfc; border-radius: 0 4px 4px 0;
}
.member.canonical { border-left-color: var(--canonical-border); background: var(--canonical-bg); }
.member.duplicate { border-left-color: var(--duplicate-border); background: var(--duplicate-bg); }
.member-header {
  display: flex; gap: 10px; align-items: center; flex-wrap: wrap;
  margin-bottom: 4px; font-size: 13px;
}
.badge {
  background: var(--badge-bg); border-radius: 10px; padding: 2px 8px;
  font-size: 12px; font-family: ui-monospace, monospace;
}
.badge.canonical { background: var(--canonical-bg); border: 1px solid var(--canonical-border); color: #1b5e20; }
.badge.duplicate { background: var(--duplicate-bg); border: 1px solid var(--duplicate-border); color: #7a4a00; }
.member-id { font-family: ui-monospace, monospace; font-size: 12px; color: var(--muted); }
.member-chars { color: var(--muted); font-size: 12px; }
details { margin-top: 4px; }
summary { cursor: pointer; user-select: none; color: #0067c0; font-size: 13px; }
summary:hover { text-decoration: underline; }
pre.text {
  white-space: pre-wrap; word-wrap: break-word;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12px; line-height: 1.45;
  background: #fff; border: 1px solid var(--border);
  padding: 8px; margin: 6px 0 0; max-height: 540px; overflow-y: auto;
}
.hidden { display: none !important; }
.actions { margin-left: auto; }
.actions button {
  padding: 4px 10px; font-size: 12px; border: 1px solid var(--border);
  border-radius: 4px; background: #fff; cursor: pointer;
}
.actions button:hover { background: #f0f4f9; }
</style>
</head>
<body>
<h1>Duplicate cluster examples</h1>
<div class="summary">__SUMMARY__</div>

<div class="filters">
  <input type="search" id="q" placeholder="search text / id / source / cluster_id...">
  <select id="src"><option value="">(any source)</option>__SOURCE_OPTIONS__</select>
  <select id="size">
    <option value="">(any size)</option>
    <option value="2">size 2</option>
    <option value="3">size 3</option>
    <option value="4">size 4</option>
    <option value="5+">size 5+</option>
  </select>
  <span class="count" id="count"></span>
  <span class="actions">
    <button id="expand-all">expand all</button>
    <button id="collapse-all">collapse all</button>
  </span>
</div>

<div id="clusters">__CLUSTERS__</div>

<script>
(function () {
  const cards = Array.from(document.querySelectorAll('.cluster'));
  const q = document.getElementById('q');
  const src = document.getElementById('src');
  const size = document.getElementById('size');
  const count = document.getElementById('count');

  function applyFilters() {
    const query = q.value.toLowerCase().trim();
    const wantSrc = src.value;
    const wantSize = size.value;
    let visible = 0;
    for (const c of cards) {
      let show = true;
      if (wantSize) {
        const n = +c.dataset.size;
        show = wantSize === '5+' ? n >= 5 : n === +wantSize;
      }
      if (show && wantSrc) {
        const has = c.querySelector(`.member[data-source="${wantSrc.replace(/"/g, '\\"')}"]`);
        show = !!has;
      }
      if (show && query) {
        show = c.textContent.toLowerCase().includes(query);
      }
      c.classList.toggle('hidden', !show);
      if (show) visible++;
    }
    count.textContent = visible + ' of ' + cards.length + ' shown';
  }

  q.addEventListener('input', applyFilters);
  src.addEventListener('change', applyFilters);
  size.addEventListener('change', applyFilters);
  document.getElementById('expand-all').addEventListener('click', () => {
    document.querySelectorAll('.cluster:not(.hidden) details').forEach(d => d.open = true);
  });
  document.getElementById('collapse-all').addEventListener('click', () => {
    document.querySelectorAll('details').forEach(d => d.open = false);
  });
  applyFilters();
})();
</script>
</body>
</html>
"""


def _read_examples(path: str) -> list[dict]:
    """Read examples from either a single parquet file or a directory of shards."""
    location = StoragePath(path)
    if location.isdir():
        files = sorted((p for p in location.ls() if str(p).endswith(".parquet")), key=str)
        if not files:
            raise FileNotFoundError(f"no parquet files under {path}")
    else:
        files = [location]
    rows: list[dict] = []
    for f in files:
        with f.open("rb") as fh:
            rows.extend(pq.read_table(fh).to_pylist())
    return rows


def _primary_source(c: dict) -> str:
    """A cluster's primary source: the canonical member's source, or the first
    member if no canonical is present in our (capped) members list."""
    for m in c["members"]:
        if m.get("is_canonical"):
            return m["source"]
    return c["members"][0]["source"]


def _sample_clusters(rows: list[dict], n: int, min_m: int, max_m: int, seed: int, strategy: str) -> list[dict]:
    rng = random.Random(seed)
    preferred = [r for r in rows if min_m <= r["num_members"] <= max_m]
    if len(preferred) >= n:
        pool = preferred
    else:
        logger.info(
            "preferred pool (%d <= size <= %d) only has %d clusters; falling back to all non-singletons",
            min_m,
            max_m,
            len(preferred),
        )
        pool = [r for r in rows if r["num_members"] >= 2]
    if len(pool) <= n:
        return list(pool)

    if strategy == "random":
        return rng.sample(pool, n)

    # stratified: round-robin across primary sources to spread coverage evenly.
    buckets: dict[str, list[dict]] = {}
    for c in pool:
        buckets.setdefault(_primary_source(c), []).append(c)
    for v in buckets.values():
        rng.shuffle(v)
    sources_ordered = sorted(buckets.keys())
    rng.shuffle(sources_ordered)  # randomize starting source so seed varies coverage
    logger.info("stratified sample across %d primary sources", len(sources_ordered))
    picked: list[dict] = []
    while len(picked) < n:
        progress = False
        for s in sources_ordered:
            if not buckets[s]:
                continue
            picked.append(buckets[s].pop())
            progress = True
            if len(picked) == n:
                break
        if not progress:
            break
    return picked


def _short_id(cid: str, n: int = 16) -> str:
    return cid[:n] + ("..." if len(cid) > n else "")


def _render_cluster(c: dict) -> str:
    sources = sorted({m["source"] for m in c["members"]})
    cluster_id_full = html.escape(c["cluster_id"])
    cluster_id_short = html.escape(_short_id(c["cluster_id"]))
    parts: list[str] = [
        f'<div class="cluster" data-size="{c["num_members"]}">',
        '<div class="cluster-header">',
        f'<span class="cluster-size">size={c["num_members"]}</span>',
        f'<span class="cluster-id" title="{cluster_id_full}">{cluster_id_short}</span>',
        f'<span class="sources-line">{html.escape(", ".join(sources))}</span>',
        "</div>",
    ]
    for m in c["members"]:
        tag = "canonical" if m["is_canonical"] else "duplicate"
        parts.append(f'<div class="member {tag}" data-source="{html.escape(m["source"], quote=True)}">')
        parts.append('<div class="member-header">')
        parts.append(f'<span class="badge {tag}">{tag.upper()}</span>')
        parts.append(f'<span class="badge">{html.escape(m["source"])}</span>')
        parts.append(f'<span class="member-id">{html.escape(m["id"])}</span>')
        parts.append(f'<span class="member-chars">{m["chars"]:,} chars</span>')
        parts.append("</div>")
        parts.append("<details><summary>show text</summary>")
        parts.append(f'<pre class="text">{html.escape(m["text"])}</pre>')
        parts.append("</details>")
        parts.append("</div>")
    parts.append("</div>")
    return "".join(parts)


def _render_html(rows_total: int, sample: list[dict], seed: int, min_m: int, max_m: int, strategy: str) -> str:
    sources_set = sorted({m["source"] for c in sample for m in c["members"]})
    primary_sources = {_primary_source(c) for c in sample}
    source_options = "".join(
        f'<option value="{html.escape(s, quote=True)}">{html.escape(s)}</option>' for s in sources_set
    )
    summary = (
        f"sampled {len(sample):,} clusters via {strategy} (seed={seed}, prefer {min_m}-{max_m} members; "
        f"non-singleton pool: {rows_total:,}; {len(primary_sources)} primary sources, "
        f"{len(sources_set)} sources across all members)"
    )
    cards = "\n".join(_render_cluster(c) for c in sample)
    return (
        HTML_TEMPLATE.replace("__SUMMARY__", html.escape(summary))
        .replace("__SOURCE_OPTIONS__", source_options)
        .replace("__CLUSTERS__", cards)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Examples parquet from fetch_cluster_texts.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Local path to write the HTML report.")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-members", type=int, default=2)
    parser.add_argument("--max-members", type=int, default=4)
    parser.add_argument("--upload", default=None, help="Optional gs:// path to also upload the report.")
    parser.add_argument(
        "--strategy",
        choices=("stratified", "random"),
        default="stratified",
        help="stratified: round-robin across primary sources (default); random: uniform sample.",
    )
    args = parser.parse_args()

    configure_logging(logging.INFO)
    logger.info("reading %s", args.input)
    rows = _read_examples(args.input)
    non_singleton = [r for r in rows if r["num_members"] >= 2]
    logger.info("loaded %d cluster rows (%d non-singleton)", len(rows), len(non_singleton))

    sample = _sample_clusters(rows, args.sample_size, args.min_members, args.max_members, args.seed, args.strategy)
    logger.info("rendering report for %d clusters (strategy=%s)", len(sample), args.strategy)

    body = _render_html(len(non_singleton), sample, args.seed, args.min_members, args.max_members, args.strategy)
    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write(body)
    logger.info("wrote %s (%d bytes)", args.output, len(body))

    if args.upload:
        StoragePath(args.upload).write_bytes(body.encode("utf-8"))
        logger.info("uploaded to %s", args.upload)


if __name__ == "__main__":
    main()
