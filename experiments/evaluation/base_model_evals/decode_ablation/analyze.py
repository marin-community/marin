# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Decode-ablation analysis: IFBench scores, self-surprisal, post-trained-Qwen gaps and leak rates.

Inputs (under ../data):
  summaries/summary-<author>-<cell>.json   per-cell IFBench summaries (decode_ablation_extract.py)
  stores/<author>/store-<cell>/            Evalchemy stores (for paired per-prompt accuracy)
  traces/traces-<author>-<cell>.jsonl      trace files (100 prompts per cell)
  scores/<scorer>.jsonl.gz                 rescoring of all new traces under each scorer
  selfscores/selfscore-<author>-<cell>.jsonl.gz  each cell's traces under its own generator
The as-run baseline (09-25 / 10-26 runs) comes from ../../trace-likelihood/.

Conventions follow trace-likelihood/analyze_traces.py and snowball-sft-improbability/: each token's
NLL is spread evenly over its characters; gaps are ratio-of-sums nats/char over the characters
every scorer scored; CIs are 95% percentile bootstraps over prompts (4,000 replicates).

Run: uv run --no-project --with numpy --with pyarrow python analyze.py
"""

import glob
import gzip
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"
TL = HERE.parent / "trace-likelihood"
OUT = HERE / "results"

SCORERS = [
    "qwen3.5-35b-a3b-base",
    "qwen3.5-35b-a3b",
    "qwen3.6-35b-a3b",
    "snowball-67b-a2b-base-skew8",
    "grug-67b-a2b-datakit-sft",
]
SHORT = dict(zip(SCORERS, ["Q3.5 base", "Q3.5 post", "Q3.6 post", "Snowball base", "Snowball SFT"]))
BASE, Q35POST, Q36POST = SCORERS[:3]
AUTHORS = ["grugsft471", "snowballbase", "glm47flash", "kimilinear", "qwen36"]
CELLS = ["asrun", "shipped", "greedy", "t06"]
CELL_NAME = {"asrun": "as run (09-25/10-26)", "shipped": "shipped default", "greedy": "greedy", "t06": "T0.6/top_p0.95"}
REF = ["glm47flash", "kimilinear"]
SELF_SCORER = {
    "grugsft471": "grug-67b-a2b-datakit-sft",
    "snowballbase": "snowball-67b-a2b-base-skew8",
    "qwen36": "qwen3.6-35b-a3b",
}

LEAK = re.compile(r"(?<![A-Za-z])(?:user|assistant)[ \t]*\n|(?:^|\n)\[?(?:User|Assistant)\]?:")
LINE_ROLE = re.compile(r"(^|\n)\s*(user|assistant|User|Assistant)\s*(\n|:)")
NREP = 4000
P01, P001 = math.log(100), math.log(1000)


def fused_cut(text: str) -> int:
    m = LEAK.search(text)
    return m.start() if m else len(text)


def line_cut(text: str) -> int:
    m = LINE_ROLE.search(text)
    return m.start() if m else len(text)


def char_nll(n: int, offsets, nlls) -> np.ndarray:
    values = np.full(n, np.nan)
    for (start, end), nll in zip(offsets, nlls, strict=True):
        if end > start:
            values[start : min(end, n)] = nll / (end - start)
    return values


def read_gz(path) -> dict[str, dict]:
    return {r["trace_id"]: r for r in map(json.loads, gzip.open(path, "rt"))}


def load():
    """Returns traces[(author, cell)] -> {key: trace}, and char NLLs / self token NLLs keyed by trace_id."""
    traces = defaultdict(dict)
    for line in (TL / "ifbench-11-authors-100-prompts.jsonl").read_text().splitlines():
        t = json.loads(line)
        if t["author"] in AUTHORS:
            t["cell"] = "asrun"
            t["full_response"] = None
            traces[(t["author"], "asrun")][t["prompt_key"]] = t
    for path in sorted(glob.glob(str(DATA / "traces" / "traces-*.jsonl"))):
        for line in Path(path).read_text().splitlines():
            t = json.loads(line)
            author = t["author"].split("@")[0]
            t["author"] = author
            traces[(author, t["cell"])][t["prompt_key"]] = t
    by_id = {t["trace_id"]: t for d in traces.values() for t in d.values()}

    chars = defaultdict(dict)  # trace_id -> scorer -> per-char nll
    for s in SCORERS:
        files = [TL / "data" / f"{s}.jsonl.gz", DATA / "scores" / f"{s}.jsonl.gz"]
        for f in files:
            if not f.exists():
                continue
            for tid, r in read_gz(f).items():
                if tid in by_id:
                    chars[tid][s] = char_nll(len(by_id[tid]["response"]), r["offsets"], r["nll"])

    self_tok = {}  # trace_id -> (offsets array, nll array) under the generating model
    for author, scorer in SELF_SCORER.items():
        f = TL / "data" / f"{scorer}.jsonl.gz"
        for tid, r in read_gz(f).items():
            if tid in by_id and by_id[tid]["author"] == author:
                self_tok[tid] = (np.array(r["offsets"]).reshape(-1, 2), np.array(r["nll"]))
    for f in sorted(glob.glob(str(DATA / "selfscores" / "selfscore-*.jsonl.gz"))):
        for tid, r in read_gz(f).items():
            self_tok[tid] = (np.array(r["offsets"]).reshape(-1, 2), np.array(r["nll"]))
    return traces, chars, self_tok


def boot_ratio(num: np.ndarray, den: np.ndarray, rng) -> tuple[float, float, float]:
    idx = rng.integers(0, len(num), size=(NREP, len(num)))
    stats = num[idx].sum(1) / den[idx].sum(1)
    return num.sum() / den.sum(), np.percentile(stats, 2.5), np.percentile(stats, 97.5)


def boot_diff(num_a, den_a, num_b, den_b, rng):
    """Paired (same prompts) difference of two ratio-of-sums gaps."""
    idx = rng.integers(0, len(num_a), size=(NREP, len(num_a)))
    stats = num_a[idx].sum(1) / den_a[idx].sum(1) - num_b[idx].sum(1) / den_b[idx].sum(1)
    est = num_a.sum() / den_a.sum() - num_b.sum() / den_b.sum()
    return est, np.percentile(stats, 2.5), np.percentile(stats, 97.5)


def fmt(ci, digits=3, sign=True):
    est, lo, hi = ci
    s = "+" if sign else ""
    return f"{est:{s}.{digits}f} [{lo:{s}.{digits}f}, {hi:{s}.{digits}f}]"


def per_prompt_gap(cell_traces, chars, scorer, cut, keys=None):
    """Per-prompt (sum of scorer-base nats, char count) over the cut text, chars scored by all scorers."""
    keys = sorted(cell_traces) if keys is None else keys
    num, den, absn = [], [], defaultdict(list)
    for k in keys:
        t = cell_traces[k]
        c = chars[t["trace_id"]]
        if not all(s in c for s in SCORERS):
            return None
        text = t["response"]
        end = {"none": len(text), "line": line_cut(text), "fused": fused_cut(text)}[cut]
        arrs = {s: c[s][:end] for s in SCORERS}
        mask = np.all([~np.isnan(a) for a in arrs.values()], axis=0)
        num.append(np.sum(arrs[scorer][mask] - arrs[BASE][mask]))
        den.append(mask.sum())
        for s in SCORERS:
            absn[s].append(np.sum(arrs[s][mask]))
    return np.array(num), np.maximum(np.array(den), 1), {s: np.array(v) for s, v in absn.items()}


def self_surprisal(cell_traces, self_tok, cut):
    tot = n01 = n001 = 0
    early, late = [], []
    for t in cell_traces.values():
        if t["trace_id"] not in self_tok:
            return None
        offs, nll = self_tok[t["trace_id"]]
        text = t["response"]
        end = {"none": len(text), "fused": fused_cut(text)}[cut]
        keep = offs[:, 0] < end
        v = nll[keep]
        tot += len(v)
        n01 += int(np.sum(v > P01))
        n001 += int(np.sum(v > P001))
        early.extend(nll[keep & (offs[:, 0] < 300)])
        late.extend(nll[keep & (offs[:, 0] >= 2000)])
    return (
        n01 / tot,
        n001 / tot,
        float(np.mean(early)) if early else float("nan"),
        float(np.mean(late)) if late else float("nan"),
        tot,
    )


def store_rows(author, cell):
    files = glob.glob(str(DATA / "stores" / author / f"store-{cell}" / "data" / "samples" / "*" / "*" / "*.parquet"))
    out = {}
    for f in files:
        for r in pq.read_table(f).to_pylist():
            out[json.loads(r["doc"])["key"]] = dict(r["metrics"])["strict_prompt_accuracy"] == 1.0
    return out


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    traces, chars, self_tok = load()
    lines = []
    emit = lines.append
    results = {}

    # 1. IFBench strict prompt-level accuracy (full 300 prompts).
    emit("## IFBench strict prompt-level accuracy (300 prompts)\n")
    emit("CI: 95% bootstrap over prompts. Δ: paired difference vs the shipped-default cell.\n")
    emit(
        "| author | as run | shipped | greedy | T0.6/p0.95 | Δ greedy − shipped | Δ T0.6 − shipped | finish=length (shipped/greedy/T0.6) |"
    )
    emit("|---|---:|---:|---:|---:|---:|---:|---:|")
    asrun_scores = {}
    for line in (TL / "ifbench-11-authors-100-prompts.jsonl").read_text().splitlines():
        t = json.loads(line)
        asrun_scores[t["author"]] = t["author_ifbench"]
    for author in AUTHORS:
        rows = {c: store_rows(author, c) for c in ("shipped", "greedy", "t06")}
        summ = {}
        for c in ("shipped", "greedy", "t06"):
            p = DATA / "summaries" / f"summary-{author}-{c}.json"
            summ[c] = json.loads(p.read_text()) if p.exists() else None
        cells_txt = []
        for c in ("shipped", "greedy", "t06"):
            r = rows[c]
            if not r:
                cells_txt.append("—")
                continue
            v = np.array(list(r.values()), dtype=float)
            idx = rng.integers(0, len(v), size=(NREP, len(v)))
            st = v[idx].mean(1)
            cells_txt.append(f"{v.mean():.3f} [{np.percentile(st, 2.5):.3f}, {np.percentile(st, 97.5):.3f}]")
            results.setdefault("ifbench", {}).setdefault(author, {})[c] = float(v.mean())
        deltas = []
        for c in ("greedy", "t06"):
            a, b = rows[c], rows["shipped"]
            keys = sorted(set(a) & set(b))
            if not keys:
                deltas.append("—")
                continue
            d = np.array([a[k] - b[k] for k in keys], dtype=float)
            idx = rng.integers(0, len(d), size=(NREP, len(d)))
            st = d[idx].mean(1)
            deltas.append(f"{d.mean():+.3f} [{np.percentile(st, 2.5):+.3f}, {np.percentile(st, 97.5):+.3f}]")
        fin = "/".join(
            str((summ[c] or {}).get("finish_reasons", {}).get("length", 0)) if summ[c] else "—"
            for c in ("shipped", "greedy", "t06")
        )
        emit(f"| {author} | {asrun_scores[author]:.3f} | {' | '.join(cells_txt)} | {' | '.join(deltas)} | {fin} |")
    emit("")

    # 2. Self-surprisal under the generating model.
    emit("## Self-surprisal on the 100 trace responses (generating model, fused-leak cut)\n")
    emit(
        "Share of response tokens with p<0.01 / p<0.001 under the model that generated them; mean self-NLL "
        "(nats/token) over the first 300 characters and after character 2,000. As-run rows exist only "
        "for authors that were also scorers.\n"
    )
    emit("| author | cell | p<.01 | p<.001 | NLL 0–300 | NLL ≥2000 | tokens |")
    emit("|---|---|---:|---:|---:|---:|---:|")
    for author in AUTHORS:
        for cell in CELLS:
            ct = traces.get((author, cell))
            if not ct:
                continue
            s = self_surprisal(ct, self_tok, "fused")
            if s is None:
                continue
            results.setdefault("self", {}).setdefault(author, {})[cell] = s[:2]
            emit(f"| {author} | {CELL_NAME[cell]} | {s[0]:.3f} | {s[1]:.4f} | {s[2]:.2f} | {s[3]:.2f} | {s[4]} |")
    emit("")

    # 3. Leak rates.
    emit("## Leak rate (share of the 100 trace responses with a leaked turn)\n")
    emit(
        "Line cut: a role word on its own line or a `User:` line. Fused cut: also a role word fused onto the "
        "end of a line (`...built.user\\n`). New cells are checked on the full response; the as-run rows "
        "only on the 4,000-character trace text. Also: share of responses cut at 4,000 characters.\n"
    )
    emit("| author | cell | line-cut leaks | fused-cut leaks | >4,000 chars |")
    emit("|---|---|---:|---:|---:|")
    for author in AUTHORS:
        for cell in CELLS:
            ct = traces.get((author, cell))
            if not ct:
                continue
            texts = [t["full_response"] or t["response"] for t in ct.values()]
            ln = sum(line_cut(x) < len(x) for x in texts) / len(texts)
            fu = sum(fused_cut(x) < len(x) for x in texts) / len(texts)
            tr = sum(bool(t["truncated"]) for t in ct.values()) / len(ct)
            results.setdefault("leak", {}).setdefault(author, {})[cell] = (ln, fu)
            emit(f"| {author} | {CELL_NAME[cell]} | {ln:.2f} | {fu:.2f} | {tr:.2f} |")
    emit("")

    # 4. Gaps vs Qwen3.5 base.
    for cut, label in (("fused", "fused-leak cut"), ("none", "uncut 4,000-character text"), ("line", "line cut")):
        emit(f"## Gap vs Qwen3.5 base, nats/char ({label})\n")
        emit(
            "Scorer − Qwen3.5 base, ratio of sums over characters all five scorers scored; 95% bootstrap CI over "
            "prompts. Qwen3.6's own text under Qwen3.6 is a self-score.\n"
        )
        emit("| author | cell | Q3.5 post − base | Q3.6 post − base |")
        emit("|---|---|---:|---:|")
        for author in AUTHORS:
            for cell in CELLS:
                ct = traces.get((author, cell))
                if not ct:
                    continue
                g = {}
                for s in (Q35POST, Q36POST):
                    r = per_prompt_gap(ct, chars, s, cut)
                    if r is None:
                        break
                    g[s] = boot_ratio(r[0], r[1], rng)
                if len(g) < 2:
                    continue
                results.setdefault(f"gap_{cut}", {}).setdefault(author, {})[cell] = {s: g[s][0] for s in g}
                emit(f"| {author} | {CELL_NAME[cell]} | {fmt(g[Q35POST])} | {fmt(g[Q36POST])} |")
        emit("")

    # 4b. Gaps on traces that ended at end-of-turn (excludes length-capped repetition loops).
    emit("## Gap vs Qwen3.5 base on end-of-turn traces only (fused cut)\n")
    emit("Only traces whose generation finished with `stop`; relative gap = scorer/base − 1 in nats/char.\n")
    emit("| author | cell | n | Q3.5 post − base | Q3.6 post − base | Q3.5 post rel. | Q3.6 post rel. |")
    emit("|---|---|---:|---:|---:|---:|---:|")
    for author in AUTHORS:
        for cell in ("shipped", "greedy", "t06"):
            ct = traces.get((author, cell))
            if not ct:
                continue
            keep = {k: t for k, t in ct.items() if t.get("finish_reason") == "stop"}
            g = {}
            for sc in (Q35POST, Q36POST):
                r = per_prompt_gap(keep, chars, sc, "fused")
                g[sc] = (boot_ratio(r[0], r[1], rng), r)
            base = g[Q35POST][1][2][BASE].sum()
            rel = {sc: g[sc][1][2][sc].sum() / base - 1 for sc in g}
            emit(
                f"| {author} | {CELL_NAME[cell]} | {len(keep)} | {fmt(g[Q35POST][0])} | {fmt(g[Q36POST][0])} | "
                f"{rel[Q35POST]:+.1%} | {rel[Q36POST]:+.1%} |"
            )
    emit("")

    # 5. SFT vs reference authors within each cell (paired by prompt).
    emit("## Snowball SFT excess over the reference authors (fused cut)\n")
    emit(
        "SFT gap − mean gap of GLM-4.7-Flash and Kimi-Linear on the same prompts, and SFT − Snowball base; "
        "paired bootstrap over prompts.\n"
    )
    emit("| cell | scorer | SFT gap | REF gap | SFT − REF | SFT − Snowball base |")
    emit("|---|---|---:|---:|---:|---:|")
    for cell in CELLS:
        for s in (Q35POST, Q36POST):
            per = {}
            group = ["grugsft471", "snowballbase", *REF]
            if not all(traces.get((a, cell)) for a in group):
                continue
            common = sorted(set.intersection(*(set(traces[(a, cell)]) for a in group)))
            for a in group:
                ct = traces.get((a, cell))
                r = per_prompt_gap(ct, chars, s, "fused", common)
                if r is None:
                    break
                per[a] = (r[0], r[1])
            if len(per) < 4:
                continue
            ref_num = sum(per[a][0] / per[a][1] for a in REF) / len(REF)  # per-prompt mean of per-prompt gaps
            sft_n, sft_d = per["grugsft471"]
            idx = rng.integers(0, len(sft_n), size=(NREP, len(sft_n)))
            ref_rs = [(per[a][0][idx].sum(1) / per[a][1][idx].sum(1)) for a in REF]
            sft_rs = sft_n[idx].sum(1) / sft_d[idx].sum(1)
            ref_est = np.mean([per[a][0].sum() / per[a][1].sum() for a in REF])
            sft_est = sft_n.sum() / sft_d.sum()
            diff = sft_rs - np.mean(ref_rs, axis=0)
            dref = (sft_est - ref_est, np.percentile(diff, 2.5), np.percentile(diff, 97.5))
            dbase = boot_diff(sft_n, sft_d, *per["snowballbase"], rng)
            del ref_num
            results.setdefault("sft_vs_ref", {}).setdefault(cell, {})[s] = (sft_est, ref_est, dref[0], dbase[0])
            emit(
                f"| {CELL_NAME[cell]} | {SHORT[s]} | {sft_est:+.4f} | {ref_est:+.4f} | {fmt(dref, 4)} | {fmt(dbase, 4)} |"
            )
    emit("")

    # 6. Absolute nats/char per scorer (fused cut).
    emit("## Nats/char under each scorer (fused cut)\n")
    emit("| author | cell | " + " | ".join(SHORT[s] for s in SCORERS) + " |")
    emit("|---|---|" + "---:|" * len(SCORERS))
    for author in AUTHORS:
        for cell in CELLS:
            ct = traces.get((author, cell))
            r = per_prompt_gap(ct, chars, Q35POST, "fused") if ct else None
            if r is None:
                continue
            emit(
                f"| {author} | {CELL_NAME[cell]} | "
                + " | ".join(f"{r[2][s].sum() / r[1].sum():.3f}" for s in SCORERS)
                + " |"
            )
    emit("")

    (OUT / "tables.md").write_text("\n".join(lines) + "\n")
    (OUT / "results.json").write_text(json.dumps(results, indent=1, default=float))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
