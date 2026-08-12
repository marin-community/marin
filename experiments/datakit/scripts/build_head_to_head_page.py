# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render the head-to-head document viewer from the payload :mod:`fusion_vs_v3_doc_table` writes.

The published v3 overview shows every source family split into the five
calibrated quality buckets, with example documents to read underneath. This page
keeps that shape and changes one thing: every panel and every document carries
both scorers, the deployed v3 and the fusion candidate, side by side.

The payload is embedded as JSON and the page renders from it, so a family's
documents enter the DOM when the family is opened rather than all at once.
"""

# This module is mostly a CSS/JS/HTML asset held in string literals: E501 would
# reflow stylesheet and template lines that are easier to read intact, and the
# character RUF001 flags is the multiplication sign the page prints beside an
# enrichment ratio.
# ruff: noqa: E501, RUF001

import argparse
import json
import logging
import pathlib

from rigging.log_setup import configure_logging

logger = logging.getLogger(__name__)

TITLE = "Quality scorers head to head"

# Crawled documents carry bytes that survive scoring but not embedding in a web
# page: U+FFFD left by an upstream decoder, and C0 control characters. Both are
# stripped from the displayed window text only — the scores were computed on the
# document as stored, and the payload on object storage keeps it verbatim.
LOST_CHARACTER = "�"
LOST_MARKER = "□"
CONTROL_CHARACTERS = {c: None for c in range(0x20) if c not in (0x09, 0x0A, 0x0D)} | {0x7F: None}

STYLE = """
:root {
  --ink:#1f1e1b; --muted:#6e6456; --paper:#b5aa9f; --panel:#f4efe7; --rule:#d8cdbc;
  --blue:#224a82; --field:#ffffff; --track:#e6ded1; --inset:#faf7f1; --chip-ink:#f7f3ec;
  --q0:#a23e2a; --q1:#8f6b38; --q2:#6e6456; --q3:#385c8f; --q4:#224a82;
  font-family:"IBM Plex Sans",-apple-system,"Segoe UI",Roboto,sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ink:#ece4d6; --muted:#a1968a; --paper:#14120e; --panel:#201d17; --rule:#3a3428;
    --blue:#8fb3e6; --field:#191610; --track:#332e24; --inset:#1a1712; --chip-ink:#14120e;
    --q0:#d9755c; --q1:#d1a15c; --q2:#a29788; --q3:#6f9ad2; --q4:#a8c6ef;
  }
}
:root[data-theme="dark"] {
  --ink:#ece4d6; --muted:#a1968a; --paper:#14120e; --panel:#201d17; --rule:#3a3428;
  --blue:#8fb3e6; --field:#191610; --track:#332e24; --inset:#1a1712; --chip-ink:#14120e;
  --q0:#d9755c; --q1:#d1a15c; --q2:#a29788; --q3:#6f9ad2; --q4:#a8c6ef;
}
* { box-sizing:border-box; } html { scroll-behavior:smooth; }
body { margin:0; color:var(--ink); background:var(--paper); }
main { width:min(1480px,calc(100% - 32px)); margin:34px auto 80px; }
.eyebrow,.cluster-id,.bucket-id,.confidence,.qnum,.enr,.qcount,.who,.mono { font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace; }
.eyebrow { color:var(--blue); font-size:12px; font-weight:600; letter-spacing:.13em; text-transform:uppercase; }
h1 { max-width:900px; margin:8px 0; font-size:clamp(28px,5vw,48px); letter-spacing:-.035em; line-height:1.04; text-wrap:balance; }
.lede { max-width:850px; color:var(--muted); line-height:1.55; }
.lede b { color:var(--ink); }
.corpus-pair { display:flex; flex-direction:column; gap:8px; margin:22px 0 6px; }
.model-row { display:grid; grid-template-columns:64px 1fr; gap:10px; align-items:stretch; }
.rowlabel { display:flex; align-items:center; justify-content:flex-end; padding-right:2px;
  font:600 12px "IBM Plex Mono",monospace; letter-spacing:.06em; text-transform:uppercase; color:var(--muted); }
.corpus { display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:10px; }
.sum { background:var(--panel); border:1px solid var(--rule); border-radius:10px; padding:12px; }
.toolbar { position:sticky; top:0; z-index:4; display:flex; flex-wrap:wrap; gap:12px; align-items:center; margin:24px 0; padding:12px;
  background:var(--panel); border:1px solid var(--rule); border-radius:12px; }
.toolbar input { flex:1 1 220px; min-width:180px; border:1px solid var(--rule); border-radius:8px; padding:11px 13px;
  background:var(--field); color:var(--ink); font:inherit; }
.toolbar select { border:1px solid var(--rule); border-radius:8px; padding:11px 13px; background:var(--field); color:var(--ink);
  font:500 13px "IBM Plex Mono",monospace; cursor:pointer; }
.toolbar button { white-space:nowrap; border:0; border-radius:8px; padding:11px 13px; background:var(--blue); color:var(--chip-ink);
  font:600 13px "IBM Plex Mono",monospace; cursor:pointer; }
.toolbar :focus-visible { outline:2px solid var(--blue); outline-offset:2px; }
.index { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:8px; margin:18px 0 28px; }
.index a { display:flex; gap:10px; align-items:center; padding:10px 12px; color:var(--ink); background:var(--panel);
  border:1px solid var(--rule); border-radius:9px; text-decoration:none; font-size:14px; }
.index b { color:var(--blue); font-family:"IBM Plex Mono",monospace; }
.cluster { margin:12px 0; background:var(--panel); border:1px solid var(--rule); border-radius:13px; overflow:hidden; }
.cluster>summary { display:grid; grid-template-columns:auto 1fr auto; gap:14px; align-items:center; padding:16px 18px; cursor:pointer; list-style:none; }
.cluster>summary::-webkit-details-marker { display:none; }
.cluster-id { color:var(--blue); font-size:22px; font-weight:600; }
.cluster strong { display:block; font-size:20px; }
.cluster small { display:block; max-width:900px; margin-top:4px; color:var(--muted); font-size:13px; line-height:1.4; }
.confidence { color:var(--muted); font-size:11px; text-transform:uppercase; white-space:nowrap; }
.cluster-body { padding:0 18px 18px; }
.quality-grid { display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:10px; }
.quality { min-width:0; border:1px solid var(--rule); border-top:4px solid var(--muted); border-radius:10px; padding:12px; background:var(--inset); }
.qhead { display:flex; justify-content:space-between; align-items:baseline; }
.bucket-id { color:var(--blue); font-size:13px; font-weight:600; }
.qlabel { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.08em; }
.share { margin-top:8px; }
.who { font-size:10px; letter-spacing:.09em; text-transform:uppercase; color:var(--muted); }
.qnum { font-size:20px; margin:2px 0 4px; font-variant-numeric:tabular-nums; }
.bar { height:6px; background:var(--track); border-radius:4px; overflow:hidden; }
.bar i { display:block; height:100%; }
.enr { margin-top:5px; font-size:12px; color:var(--muted); font-variant-numeric:tabular-nums; }
.qcount { margin-top:8px; font-size:11px; color:var(--muted); font-variant-numeric:tabular-nums; }
.ex { margin-top:8px; border-top:1px solid var(--rule); padding-top:6px; }
.ex>summary { cursor:pointer; font-size:12px; color:var(--blue); font-family:"IBM Plex Mono",monospace; word-break:break-all; }
.ex>summary:focus-visible { outline:2px solid var(--blue); outline-offset:2px; }
.exsrc { display:block; }
.chips { display:flex; flex-wrap:wrap; gap:4px; margin-top:4px; }
.chip { display:inline-block; white-space:nowrap; font:600 10px "IBM Plex Mono",monospace; padding:1px 5px;
  border-radius:4px; border:1px solid var(--rule); color:var(--chip-ink); }
.chip.flat { background:transparent; color:var(--muted); }
.verdict { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:6px; margin:6px 0 0; }
.model { border:1px solid var(--rule); border-radius:6px; padding:6px 7px; background:var(--panel); }
.model .who { display:block; }
.model .val { font:600 15px "IBM Plex Mono",monospace; font-variant-numeric:tabular-nums; margin:2px 0 3px; }
.model .meta { margin-top:3px; font-size:10px; color:var(--muted); font-family:"IBM Plex Mono",monospace; line-height:1.45; word-break:break-word; }
.delta { margin-top:5px; font:600 11px "IBM Plex Mono",monospace; color:var(--muted); font-variant-numeric:tabular-nums; }
.wlabel { margin:6px 0 2px; font:600 10px "IBM Plex Mono",monospace; letter-spacing:.08em; text-transform:uppercase; color:var(--muted); }
.ex pre { white-space:pre-wrap; word-break:break-word; font-size:11px; line-height:1.4; color:var(--ink); max-height:260px; overflow:auto;
  background:var(--field); border:1px solid var(--rule); border-radius:6px; padding:8px; margin:0; }
.noex { margin-top:8px; font-size:11px; color:var(--muted); font-style:italic; }
.note { max-width:850px; margin:26px 0 0; padding:14px 16px; background:var(--panel); border:1px solid var(--rule);
  border-radius:10px; color:var(--muted); font-size:14px; line-height:1.55; }
.note b { color:var(--ink); }
@media (max-width:900px) {
  .quality-grid,.corpus { grid-template-columns:repeat(2,minmax(0,1fr)); }
  .model-row { grid-template-columns:1fr; }
  .rowlabel { justify-content:flex-start; }
}
"""

SCRIPT = r"""
const DATA = JSON.parse(document.getElementById("payload").textContent);
const LABELS = DATA.bucket_labels, FAMS = DATA.families, DOCS = DATA.docs, SRC = DATA.source_names;
const CORPUS_V3 = DATA.corpus.v3.map(c => c / DATA.corpus.n);
const CORPUS_FU = DATA.corpus.fusion.map(c => c / DATA.corpus.n);
const WIN3 = ["begin", "middle", "end"], WIN1 = ["whole document"];

const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const pct = x => (100 * x).toFixed(1) + "%";
const num = n => n.toLocaleString("en-US");
const sign = x => (x > 0 ? "+" : "") + x;

// Search text per family: the family name plus every source under it, which is
// what the index links match on.
const HAY = FAMS.map(f => (f.name + " " + f.sources.map(i => SRC[i]).join(" ")).toLowerCase());

const KEEP = {
  all: () => true,
  disagree: d => d[8] !== d[4],
  up: d => d[8] > d[4],
  down: d => d[8] < d[4],
  far: d => Math.abs(d[8] - d[4]) >= 2,
};
const ORDER = {
  sampled: null,
  move: (a, b) => Math.abs(b[8] - b[4]) - Math.abs(a[8] - a[4]) || Math.abs(b[7] - b[3]) - Math.abs(a[7] - a[3]),
  gap: (a, b) => Math.abs(b[7] - b[3]) - Math.abs(a[7] - a[3]),
  v3: (a, b) => b[3] - a[3],
  fusion: (a, b) => b[7] - a[7],
};

function corpusRow(shares, counts) {
  return shares.map((s, b) =>
    `<div class="sum"><span class="bucket-id">q${b}</span><div class="qnum">${pct(s)}</div>` +
    `<div class="bar"><i style="width:${(100 * s).toFixed(1)}%;background:var(--q${b})"></i></div>` +
    `<div class="qcount">${num(counts[b])}</div></div>`).join("");
}

function verdict(d) {
  const wins = d[5], dScore = d[7] - d[3], dBucket = d[8] - d[4];
  const winTxt = wins.length === 3 ? WIN3 : WIN1;
  const perWindow = wins.map((v, i) => `${winTxt[i]} ${v.toFixed(3)}`).join(" · ");
  return `<div class="verdict">` +
    `<div class="model"><span class="who">v3 · ${esc(d[6])}</span><div class="val">${d[3].toFixed(3)}` +
    ` <span class="chip" style="background:var(--q${d[4]})">q${d[4]} ${LABELS[d[4]]}</span></div>` +
    `<div class="meta">${wins.length} window${wins.length > 1 ? "s" : ""} · ${perWindow}</div></div>` +
    `<div class="model"><span class="who">fusion · ${esc(d[10])}</span><div class="val">${d[7].toFixed(3)}` +
    ` <span class="chip" style="background:var(--q${d[8]})">q${d[8]} ${LABELS[d[8]]}</span></div>` +
    `<div class="meta">1 window · begin 512 tok ${d[9].toFixed(3)} raw · + 1024-d doc embedding</div></div>` +
    `</div><div class="delta">Δ score ${sign(dScore.toFixed(3))} · Δ bucket ${sign(dBucket)} · ${num(d[2])} chars</div>`;
}

function example(d) {
  const wins = d[11], winTxt = wins.length === 3 ? WIN3 : WIN1;
  const moved = d[8] !== d[4];
  const chip = moved
    ? `<span class="chip" style="background:var(--q${d[8]})">q${d[4]}→q${d[8]}</span>`
    : `<span class="chip flat">q${d[4]} held</span>`;
  const body = wins.map((w, i) => `<div class="wlabel">${winTxt[i]}</div><pre>${esc(w)}</pre>`).join("");
  return `<details class="ex"><summary><span class="exsrc">${esc(SRC[d[1]])}</span>` +
    `<span class="chips">${chip}<span class="chip flat">${esc(String(d[0]).slice(0, 22))}</span></span></summary>` +
    verdict(d) + body + `</details>`;
}

function panel(fam, b, keep, order) {
  const v3Share = fam.v3[b] / fam.n, fuShare = fam.fusion[b] / fam.n;
  const v3Enr = CORPUS_V3[b] ? v3Share / CORPUS_V3[b] : 0;
  const fuEnr = CORPUS_FU[b] ? fuShare / CORPUS_FU[b] : 0;
  let picks = (fam.cells[String(b)] || []).map(i => DOCS[i]).filter(keep);
  if (order) picks = picks.slice().sort(order);
  const body = picks.length
    ? picks.map(example).join("")
    : `<div class="noex">no sampled document under this filter</div>`;
  return `<div class="quality" style="border-top-color:var(--q${b})">` +
    `<div class="qhead"><span class="bucket-id">q${b}</span><span class="qlabel">${LABELS[b]}</span></div>` +
    `<div class="share"><span class="who">v3</span><div class="qnum">${pct(v3Share)}</div>` +
    `<div class="bar"><i style="width:${(100 * v3Share).toFixed(1)}%;background:var(--q${b})"></i></div>` +
    `<div class="enr">${v3Enr.toFixed(2)}× baseline</div></div>` +
    `<div class="share"><span class="who">fusion</span><div class="qnum">${pct(fuShare)}</div>` +
    `<div class="bar"><i style="width:${(100 * fuShare).toFixed(1)}%;background:var(--q${b})"></i></div>` +
    `<div class="enr">${fuEnr.toFixed(2)}× baseline</div></div>` +
    `<div class="qcount">${num(fam.v3[b])} v3 · ${num(fam.fusion[b])} fusion</div>` +
    body + `</div>`;
}

const search = document.getElementById("search");
const deltaSel = document.getElementById("delta");
const sortSel = document.getElementById("sort");
const toggle = document.getElementById("toggle");
const indexEl = document.getElementById("index");
const listEl = document.getElementById("families");

indexEl.innerHTML = FAMS.map((f, i) =>
  `<a href="#fam-${i}"><b>${String(i).padStart(2, "0")}</b>${esc(f.name)}</a>`).join("");

listEl.innerHTML = FAMS.map((f, i) => {
  const names = f.sources.map(j => SRC[j]);
  const shown = names.slice(0, 6).map(esc).join(" · ");
  const rest = names.length > 6 ? ` · +${names.length - 6} more` : "";
  return `<details class="cluster" id="fam-${i}"><summary>` +
    `<span class="cluster-id">${String(i).padStart(2, "0")}</span>` +
    `<span><strong>${esc(f.name)}</strong><small>${names.length} source(s) · ${num(f.n)} documents · ` +
    `${shown}${rest}</small></span>` +
    `<span class="confidence">same bucket ${Math.round(100 * f.agree / f.n)}%</span>` +
    `</summary><div class="cluster-body"></div></details>`;
}).join("");

const clusters = [...listEl.querySelectorAll(".cluster")];
const links = [...indexEl.querySelectorAll("a")];

// A body is rebuilt when it is opened and whenever the filter or sort changes
// under it, so an open family never shows a stale selection.
let generation = 0;
function fill(details, i) {
  const body = details.querySelector(".cluster-body");
  if (body.dataset.generation === String(generation)) return;
  const keep = KEEP[deltaSel.value], order = ORDER[sortSel.value];
  body.innerHTML = `<div class="quality-grid">` +
    [0, 1, 2, 3, 4].map(b => panel(FAMS[i], b, keep, order)).join("") + `</div>`;
  body.dataset.generation = String(generation);
}

clusters.forEach((d, i) => d.addEventListener("toggle", () => { if (d.open) fill(d, i); }));

function invalidate() {
  generation += 1;
  clusters.forEach((d, i) => { if (d.open) fill(d, i); });
}
deltaSel.addEventListener("change", invalidate);
sortSel.addEventListener("change", invalidate);

search.addEventListener("input", () => {
  const q = search.value.toLowerCase();
  clusters.forEach((d, i) => {
    const hit = HAY[i].includes(q);
    d.style.display = hit ? "" : "none";
    links[i].style.display = hit ? "" : "none";
  });
});

toggle.addEventListener("click", () => {
  const visible = clusters.filter(d => d.style.display !== "none");
  const allOpen = visible.every(d => d.open);
  visible.forEach(d => { d.open = !allOpen; });
  toggle.textContent = allOpen ? "Open visible families" : "Close visible families";
});
"""


def displayable(text: str) -> str:
    """Window text with the characters an HTML page cannot carry taken out."""
    return text.replace(LOST_CHARACTER, LOST_MARKER).translate(CONTROL_CHARACTERS)


def render(payload: dict) -> str:
    meta = payload["meta"]
    corpus = payload["corpus"]
    agree = corpus["agree"] / corpus["n"]
    window_text = payload["doc_fields"].index("window_text")
    for doc in payload["docs"]:
        doc[window_text] = [displayable(w) for w in doc[window_text]]
    blob = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).replace("<", "\\u003c")
    return f"""<title>{TITLE} — v3 and the fusion candidate</title>
<style>{STYLE}</style>
<main>
<div class="eyebrow">datakit · v3 vs fusion · {meta["sources"]} sources · {meta["families"]} families</div>
<h1>Document quality, head to head</h1>
<p class="lede">The {num(meta["population"])}-document domain evaluation set, scored twice: by the deployed
per-type calibrated <b>v3</b> classifier, which mean-pools begin/middle/end windows of the whole document, and by
the <b>fusion</b> candidate, which reads one 512-token begin window plus the document's 1024-d harrier embedding.
Panels are keyed on the v3 bucket and carry both models' share of it. Open a family, then open a document to read
the windows each model saw and the score each returned. The two models put the same document in the same bucket
{pct(agree)} of the time.</p>
<div class="corpus-pair">
  <div class="model-row"><span class="rowlabel">v3</span><div class="corpus" id="corpus-v3"></div></div>
  <div class="model-row"><span class="rowlabel">fusion</span><div class="corpus" id="corpus-fusion"></div></div>
</div>
<p class="lede" style="margin-top:8px">Bucket shares across all {num(corpus["n"])} scored documents.</p>
<div class="toolbar">
  <input id="search" type="search" placeholder="Filter by family or source name">
  <select id="delta" aria-label="Bucket agreement filter">
    <option value="all">All documents</option>
    <option value="disagree">Buckets disagree</option>
    <option value="up">Fusion higher</option>
    <option value="down">Fusion lower</option>
    <option value="far">Moves 2+ buckets</option>
  </select>
  <select id="sort" aria-label="Document order">
    <option value="sampled">Sampled order</option>
    <option value="move">Largest bucket move</option>
    <option value="gap">Largest score gap</option>
    <option value="v3">Highest v3 score</option>
    <option value="fusion">Highest fusion score</option>
  </select>
  <button id="toggle" type="button">Open visible families</button>
</div>
<nav class="index" id="index"></nav>
<div id="families"></div>
<div class="note"><b>How to read this.</b> Enrichment is the family's share of a bucket divided by the whole
evaluation set's share of that bucket, so <b>1.00×</b> means the family lands in that bucket exactly as often as
the population does. Each model's share is measured against its own baseline. A document's v3 score is the mean of
its per-window scores, listed beside it; the fusion score comes from the begin window and the whole-document
embedding. The two models never read the same input.
<br><br><b>Sampling.</b> Shares, counts and enrichment are computed over all {num(corpus["n"])} scored documents.
The {num(meta["sampled"])} readable documents are a deterministic stratified draw: every non-empty
family × v3-bucket cell gets at least {meta["min_per_cell"]} documents and at most {meta["max_per_cell"]}, with the
remaining budget spread proportional to cell population, and within a cell documents are ordered by a hash of their
id. The draw is representative rather than disagreement-weighted, so the filters find disagreements at the rate the
evaluation set actually has them. Window text is truncated to {num(meta["window_chars"])} characters per window, and
the bytes a page cannot carry — control characters, and the {LOST_MARKER} an upstream decoder left behind — are
substituted in the display only. Both models scored the document as stored.
<br><br><b>Models.</b> v3 <span class="mono">{meta["v3_model"]}</span>; fusion
<span class="mono">{meta["fusion_model"]}</span>; documents <span class="mono">{meta["eval_docs"]}</span>.</div>
</main>
<script id="payload" type="application/json">{blob}</script>
<script>{SCRIPT}
document.getElementById("corpus-v3").innerHTML = corpusRow(CORPUS_V3, DATA.corpus.v3);
document.getElementById("corpus-fusion").innerHTML = corpusRow(CORPUS_FU, DATA.corpus.fusion);
</script>
"""


def num(n: int) -> str:
    return f"{n:,}"


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--payload", required=True, help="local path to the payload JSON")
    p.add_argument("--out", required=True, help="local path for the rendered page")
    args = p.parse_args()
    configure_logging(logging.INFO)

    payload = json.loads(pathlib.Path(args.payload).read_text())
    html = render(payload)
    pathlib.Path(args.out).write_text(html)
    logger.info("wrote %s — %.2f MB, %d documents", args.out, len(html.encode()) / 1e6, payload["meta"]["sampled"])


if __name__ == "__main__":
    main()
