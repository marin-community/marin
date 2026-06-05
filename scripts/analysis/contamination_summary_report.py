# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build an HTML summary of nemotron math val contamination per isoflop scale.

Reads the precomputed exposure artifacts (no GCS access needed):
- scratch/nemotron_math_isoflop_jaccard_threshold_percentages.json
- scratch/nemotron_math_isoflop_contamination_exposure.json

Run:
    .venv/bin/python scripts/analysis/contamination_summary_report.py
Open:
    scratch/contamination_summary.html
"""

import json
from pathlib import Path

THRESHOLDS_FILE = Path("scratch/nemotron_math_isoflop_jaccard_threshold_percentages.json")
EXPOSURE_FILE = Path("scratch/nemotron_math_isoflop_contamination_exposure.json")
OUTPUT = Path("scratch/contamination_summary.html")

MIX_LABELS = {"p33m67": "67% math", "p50m50": "50% math", "p67m33": "33% math"}
BAR_MAX_PCT = 40.0  # full bar width at the envelope's top end


def fmt_tokens(n: float) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    return f"{n / 1e3:.0f}K"


def pct_cell(pct: float, tokens: float) -> str:
    width = min(pct / BAR_MAX_PCT, 1.0) * 100
    return (
        f'<td><div class="bar" style="width:{width:.1f}%"></div>'
        f'<span class="p">{pct:.2f}%</span> <span class="t">{fmt_tokens(tokens)}</span></td>'
    )


def envelope_section(thr: dict) -> str:
    rows = []
    for e in thr["full_corpus_upper_envelope"]:
        rows.append(
            f'<tr><td>&ge;{e["threshold"]:.2f}</td>'
            + pct_cell(e["val_token_pct"] * 100, e["val_tokens"])
            + f'<td>{e["val_docs"]:,} / 57,243</td><td>{e["val_windows"]:,} / 12,500</td>'
            f'<td>{e["train_pairs"]:,}</td></tr>'
        )
    return (
        "<h2>Full-corpus upper envelope (any train near-dup, regardless of sampling)</h2>"
        "<table><tr><th>Jaccard</th><th>val tokens contaminated</th><th>val docs</th>"
        "<th>val windows</th><th>train pairs</th></tr>" + "".join(rows) + "</table>"
    )


def per_scale_section(thr: dict, mix: str) -> str:
    rows = [r for r in thr["rows"] if r["mix"] == mix]
    scales = sorted({r["scale"] for r in rows}, key=float)
    thresholds = sorted({r["threshold"] for r in rows})
    by = {(r["scale"], r["threshold"]): r for r in rows}
    header = "<tr><th>scale</th><th>math tokens seen</th>" + "".join(f"<th>J&ge;{t:.2f}</th>" for t in thresholds)
    body = []
    for s in scales:
        cells = "".join(
            pct_cell(by[s, t]["val_token_pct"] * 100, by[s, t]["val_tokens_contaminated"]) for t in thresholds
        )
        body.append(f'<tr><td>{s}</td><td>{fmt_tokens(by[s, thresholds[0]]["math_tokens_seen"])}</td>{cells}</tr>')
    return (
        f"<h2>Sampled exposure by isoflop scale — {MIX_LABELS[mix]} ({mix})</h2>"
        "<p>% of the 51.2M val tokens whose source doc has an <em>actually sampled</em> train near-dup "
        "at the given Jaccard cutoff; bar full scale = 40%.</p>"
        f"<table>{header}</tr>{''.join(body)}</table>"
    )


def combined_section(exp: dict) -> str:
    scales = sorted({r["scale"] for r in exp["rows"]}, key=float)
    by = {(r["scale"], r["mix"]): r for r in exp["rows"]}
    mixes = ["p33m67", "p50m50", "p67m33"]
    header = "<tr><th>scale</th>" + "".join(f"<th>{MIX_LABELS[m]}</th>" for m in mixes)
    body = []
    for s in scales:
        cells = "".join(
            pct_cell(by[s, m]["combined_val_token_fraction"] * 100, by[s, m]["combined_val_tokens"]) for m in mixes
        )
        body.append(f"<tr><td>{s}</td>{cells}</tr>")
    return (
        "<h2>Combined exposure: near-dup (J&ge;0.75) + same-source-doc window overlap</h2>"
        "<p>Adds the window-split mode: val windows excluded from train, but other windows of the "
        "same source doc are trained on. K=0.20 runs; bar full scale = 40%.</p>"
        f"<table>{header}</tr>{''.join(body)}</table>"
    )


STYLE = """<style>
body{margin:0 auto;max-width:1100px;padding:24px;font:14px ui-monospace,monospace;background:#101216;color:#d6d8de}
h1{font-size:20px}h2{font-size:16px;color:#9fb6c9;margin:28px 0 6px}p{color:#8a8f9c;margin:4px 0 10px}
table{border-collapse:collapse;width:100%}
th,td{border:1px solid #2a2e38;padding:5px 9px;text-align:left;position:relative}
th{background:#1b1e26;color:#9fb6c9}td .bar{position:absolute;left:0;top:0;bottom:0;background:#27405955;z-index:0}
td .p{position:relative;font-weight:600}td .t{position:relative;color:#8a8f9c;font-size:12px}
.facts{display:flex;gap:24px;margin:12px 0}.facts div{background:#1b1e26;padding:10px 16px;border:1px solid #2a2e38}
.facts b{display:block;font-size:18px;color:#e8eaf0}
</style>"""


def main() -> None:
    thr = json.loads(THRESHOLDS_FILE.read_text())
    exp = json.loads(EXPOSURE_FILE.read_text())
    v = thr["val_set"]
    facts = (
        f'<div class="facts"><div><b>{v["windows"]:,}</b>val windows</div>'
        f'<div><b>{v["tokens"] / 1e6:.1f}M</b>val tokens</div>'
        f'<div><b>{v["docs"]:,}</b>val source docs</div>'
        f'<div><b>{v["seq_len"]}</b>tokens/window</div></div>'
    )
    html = (
        "<!doctype html><meta charset='utf-8'><title>nemotron math val contamination by isoflop</title>"
        + STYLE
        + "<h1>Nemotron math val contamination by isoflop scale</h1>"
        + facts
        + envelope_section(thr)
        + "".join(per_scale_section(thr, m) for m in ["p33m67", "p50m50", "p67m33"])
        + combined_section(exp)
        + f"<p>{thr['definition']}</p><p>K=0.20 CPT runs, seed 0 replay, verified 5-char-shingle Jaccard.</p>"
    )
    OUTPUT.write_text(html)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
