# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build a side-by-side HTML viewer for high-Jaccard val/train pairs.

Reads verified pairs (scratch/verified_pairs/*.parquet), samples pairs across
Jaccard bands, fetches val texts from val_docs and train texts from a capped
set of normalized shards (~400 MB each, hence the cap), and writes a
self-contained HTML page with word-level diff highlighting.

Run:
    .venv/bin/python scripts/analysis/jaccard_pair_viewer.py
Open:
    scratch/jaccard_viewer.html
"""

import difflib
import html
import json
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gcsfs
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

VERIFIED_DIR = Path("scratch/verified_pairs")
VAL_DOCS = "marin-us-east5/scratch/ahmed/midtrain_dedup/val_docs"
NORM = "marin-us-east5/normalized/nemotron_cc_math_v1/4plus_b05688a8/outputs/main"
ALL_IDS = Path("scratch/nemotron_math_all_ids.npy")
SHARD_ROWS = Path("scratch/nemotron_math_shard_rows.json")
CACHE = Path("scratch/jaccard_viewer_cache.json")
OUTPUT = Path("scratch/jaccard_viewer.html")

BANDS = [(0.95, 1.01), (0.9, 0.95), (0.85, 0.9), (0.8, 0.85), (0.75, 0.8), (0.5, 0.75)]
PAIRS_PER_BAND = 25
MAX_TRAIN_SHARDS = 2
TOP_PAIRS = 25  # always include the globally highest-Jaccard pairs, fetched via row filters
MAX_DOC_CHARS = 40_000


def load_pairs() -> pd.DataFrame:
    df = pd.concat([pd.read_parquet(p) for p in sorted(VERIFIED_DIR.glob("*.parquet"))], ignore_index=True)
    df = df.drop_duplicates(["val_id", "other_id"]).sort_values("jaccard", ascending=False)
    logger.info("verified pairs loaded: %d", len(df))
    return df


def train_shard_for_ids(ids: list[str]) -> dict[str, int]:
    """Map xxh3 hex ids -> normalized shard index via the global id dump."""
    all_ids = np.load(ALL_IDS)  # (45M, 2) uint64
    order = np.lexsort((all_ids[:, 1], all_ids[:, 0]))
    sorted_ids = all_ids[order]
    shard_rows = json.loads(SHARD_ROWS.read_text())
    cum = np.concatenate([[0], np.cumsum([shard_rows[s] for s in sorted(shard_rows)])])

    arr = np.array([[int(i[:16], 16), int(i[16:], 16)] for i in ids], dtype=np.uint64)
    pos = np.searchsorted(sorted_ids[:, 0], arr[:, 0], "left")
    out = {}
    for oid, hi, lo, p in zip(ids, arr[:, 0], arr[:, 1], pos, strict=True):
        while p < len(sorted_ids) and sorted_ids[p, 0] == hi:
            if sorted_ids[p, 1] == lo:
                out[oid] = int(np.searchsorted(cum, order[p], "right") - 1)
                break
            p += 1
    return out


def sample_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Pick pairs per Jaccard band, restricted to a few train shards to bound egress."""
    shard_of = train_shard_for_ids(df["other_id"].unique().tolist())
    df = df.assign(shard=df["other_id"].map(shard_of)).dropna(subset=["shard"])

    coverage = defaultdict(int)
    for lo, hi in BANDS:
        per_shard = df[(df.jaccard >= lo) & (df.jaccard < hi)]["shard"].value_counts()
        for shard, n in per_shard.items():
            coverage[shard] += min(n, PAIRS_PER_BAND)
    keep = sorted(coverage, key=coverage.get, reverse=True)[:MAX_TRAIN_SHARDS]
    logger.info("train shards: %s", keep)
    pool = df[df["shard"].isin(keep)]
    parts = [df.head(TOP_PAIRS)] + [
        pool[(pool.jaccard >= lo) & (pool.jaccard < hi)].head(PAIRS_PER_BAND) for lo, hi in BANDS
    ]
    return pd.concat(parts, ignore_index=True).drop_duplicates(["val_id", "other_id"])


def fetch_texts(fs, sample: pd.DataFrame) -> dict[str, str]:
    if CACHE.exists():
        texts = json.loads(CACHE.read_text())
        if set(sample.val_id) | set(sample.other_id) <= texts.keys():
            return texts

    def read_ids(path: str, ids: set[str]) -> dict[str, str]:
        # Shards are sorted by id, so the `in` filter prunes to a few row groups.
        t = pq.read_table(path, columns=["id", "text"], filters=[("id", "in", ids)], filesystem=fs)
        return dict(zip(t.column("id").to_pylist(), t.column("text").to_pylist(), strict=True))

    val_ids = set(sample.val_id)
    val_files = fs.glob(f"{VAL_DOCS}/*.parquet")
    norm_files = sorted(fs.glob(f"{NORM}/*.parquet"))
    by_shard = sample.groupby("shard")["other_id"].apply(set)
    jobs = [(f, val_ids) for f in val_files] + [(norm_files[s], ids) for s, ids in by_shard.items()]

    texts: dict[str, str] = {}
    with ThreadPoolExecutor(16) as ex:
        for chunk in ex.map(lambda j: read_ids(*j), jobs):
            texts.update(chunk)
    CACHE.write_text(json.dumps(texts))
    logger.info("fetched %d texts", len(texts))
    return texts


def diff_html(a: str, b: str) -> tuple[str, str]:
    """Word-level diff: shared text plain, unique text highlighted per side."""
    aw, bw = a.split(" "), b.split(" ")
    left, right = [], []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, aw, bw, autojunk=False).get_opcodes():
        ta, tb = html.escape(" ".join(aw[i1:i2])), html.escape(" ".join(bw[j1:j2]))
        left.append(ta if op == "equal" else f'<mark class="l">{ta}</mark>')
        right.append(tb if op == "equal" else f'<mark class="r">{tb}</mark>')
    return " ".join(left), " ".join(right)


HTML_TEMPLATE = """<!doctype html>
<meta charset="utf-8"><title>nemotron math val/train high-Jaccard pairs</title>
<style>
  body { margin:0; font:14px ui-monospace,monospace; background:#101216; color:#d6d8de; display:flex; height:100vh; }
  #side { width:260px; overflow-y:auto; border-right:1px solid #2a2e38; }
  #side div { padding:6px 12px; cursor:pointer; border-bottom:1px solid #1b1e26; }
  #side div:hover { background:#1d2630; }
  #side div.on { background:#274059; }
  #main { flex:1; display:flex; flex-direction:column; }
  #meta { padding:8px 16px; border-bottom:1px solid #2a2e38; color:#9fb6c9; }
  #panes { flex:1; display:flex; min-height:0; }
  .pane { flex:1; padding:14px; white-space:pre-wrap; word-wrap:break-word; line-height:1.45; overflow-y:auto; }
  .pane h3 { color:#9fb6c9; margin:0 0 8px; font-size:13px; }
  mark.l { background:#5b2a35; color:#ffb8c5; }
  mark.r { background:#1f4733; color:#9af0c0; }
  .j { color:#f4c95d; }
</style>
<body>
<div id="side"></div>
<div id="main"><div id="meta"></div><div id="panes">
  <div class="pane"><h3>val doc</h3><div id="a"></div></div>
  <div class="pane"><h3>train doc</h3><div id="b"></div></div>
</div></div>
<script>
const PAIRS = __DATA__;
const side = document.getElementById("side");
function show(i, el) {
  document.querySelectorAll("#side div").forEach(d => d.classList.remove("on"));
  el.classList.add("on");
  const p = PAIRS[i];
  document.getElementById("meta").innerHTML =
    `J=<span class="j">${p.j.toFixed(4)}</span> | val ${p.v} → train ${p.o}`;
  document.getElementById("a").innerHTML = p.a;
  document.getElementById("b").innerHTML = p.b;
  document.querySelectorAll(".pane").forEach(d => d.scrollTop = 0);
}
PAIRS.forEach((p, i) => {
  const d = document.createElement("div");
  d.textContent = `J=${p.j.toFixed(3)}  ${p.v.slice(0, 8)}…`;
  d.onclick = () => show(i, d);
  side.appendChild(d);
});
show(0, side.firstChild);
</script>
"""


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    fs = gcsfs.GCSFileSystem()
    sample = sample_pairs(load_pairs())
    texts = fetch_texts(fs, sample)

    pairs = []
    for row in sample.itertuples():
        a, b = texts.get(row.val_id), texts.get(row.other_id)
        if a is None or b is None:
            continue
        left, right = diff_html(a[:MAX_DOC_CHARS], b[:MAX_DOC_CHARS])
        pairs.append({"v": row.val_id, "o": row.other_id, "j": row.jaccard, "a": left, "b": right})
    pairs.sort(key=lambda p: -p["j"])

    OUTPUT.write_text(HTML_TEMPLATE.replace("__DATA__", json.dumps(pairs)))
    logger.info("wrote %s with %d pairs", OUTPUT, len(pairs))


if __name__ == "__main__":
    main()
