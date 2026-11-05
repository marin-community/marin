# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Analyze per-token response NLLs of the IFBench trace set under several scorer models.

Inputs: one gzip JSONL per scorer (from generational_perplexity.py --traces) in DATA_DIR,
plus the trace set itself. Token NLLs are spread evenly over each token's characters,
so scorers with different tokenizers are compared per character of the same text.

Writes summary.md (tables) and gaps.json (largest per-word gaps with context) next to this file.
"""

import gzip
import json
import math
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
TRACES = Path(sys.argv[1])
SCORERS = {
    "qwen3.5-35b-a3b-base": "Qwen3.5 base",
    "qwen3.5-35b-a3b": "Qwen3.5 post",
    "qwen3.6-35b-a3b": "Qwen3.6 post",
    "snowball-67b-a2b-base-skew8": "Snowball base",
    "grug-67b-a2b-datakit-sft": "Grug SFT",
}
SELF = {
    "qwen3.6-35b-a3b": "qwen36",
    "snowball-67b-a2b-base-skew8": "snowballbase",
    "grug-67b-a2b-datakit-sft": "grugsft471",
}
POSITION_EDGES = [0, 20, 100, 300, 1000, 2000, 4000]
WORD = re.compile(r"\S+")


def char_class(text: str, index: int) -> str:
    ch = text[index]
    line_start = index == 0 or text[index - 1] == "\n"
    if ch == "\n":
        return "newline"
    if ch in "#*|`>_~" or (line_start and ch in "-+"):
        return "markdown syntax"
    if ch.isdigit():
        return "digit"
    if ch.isspace():
        return "space"
    if not ch.isalnum():
        return "punctuation"
    word_start = index == 0 or not text[index - 1].isalnum()
    return "word start" if word_start else "word interior"


def char_nll(text: str, offsets, nlls) -> list[float | None]:
    values: list[float | None] = [None] * len(text)
    for (start, end), nll in zip(offsets, nlls, strict=True):
        if end > start:
            for i in range(start, min(end, len(text))):
                values[i] = nll / (end - start)
    return values


def spearman(xs, ys) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for rank, i in enumerate(order):
            r[i] = rank
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return num / math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))


def main() -> None:
    traces = {t["trace_id"]: t for t in map(json.loads, TRACES.read_text().splitlines())}
    authors = sorted(
        {t["author"] for t in traces.values()},
        key=lambda a: -next(t["author_ifbench"] for t in traces.values() if t["author"] == a),
    )
    ifbench = {a: next(t["author_ifbench"] for t in traces.values() if t["author"] == a) for a in authors}
    per_char = {}
    for scorer in SCORERS:
        path = DATA_DIR / f"{scorer}.jsonl.gz"
        records = [json.loads(line) for line in gzip.open(path, "rt")]
        assert len(records) == len(traces), (scorer, len(records))
        for r in records:
            per_char[(scorer, r["trace_id"])] = char_nll(traces[r["trace_id"]]["response"], r["offsets"], r["nll"])

    # Characters scored by every scorer (truncation to the token budget can differ by tokenizer).
    common = {}
    for tid, t in traces.items():
        n = len(t["response"])
        common[tid] = [i for i in range(n) if all(per_char[(s, tid)][i] is not None for s in SCORERS)]

    def mean_nll(scorer, tids, select=lambda tid, i: True):
        total = count = 0.0
        for tid in tids:
            values = per_char[(scorer, tid)]
            for i in common[tid]:
                if select(tid, i):
                    total += values[i]
                    count += 1
        return total / count if count else float("nan")

    by_author = {a: [tid for tid, t in traces.items() if t["author"] == a] for a in authors}
    lines = [
        "# Trace likelihood of IFBench responses",
        "",
        "Nats per character of each author's responses (100 shared prompts), under each scorer. "
        "Lower = more expected. * marks a scorer scoring its own outputs.",
        "",
        "| author (IFBench) | " + " | ".join(SCORERS.values()) + " |",
        "|---|" + "---:|" * len(SCORERS),
    ]
    table = {}
    for a in authors:
        row = []
        for s in SCORERS:
            table[(s, a)] = mean_nll(s, by_author[a])
            row.append(f"{table[(s, a)]:.3f}" + ("*" if SELF.get(s) == a else ""))
        lines.append(f"| {a} ({ifbench[a]:.3f}) | " + " | ".join(row) + " |")

    lines += [
        "",
        "Spearman correlation between author IFBench score and nats/char (negative = stronger "
        "authors' text is more expected), excluding the scorer's own outputs:",
        "",
    ]
    for s in SCORERS:
        pool = [a for a in authors if SELF.get(s) != a]
        lines.append(f"- {SCORERS[s]}: {spearman([ifbench[a] for a in pool], [table[(s, a)] for a in pool]):+.2f}")

    base = "qwen3.5-35b-a3b-base"
    lines += [
        "",
        "Gap vs Qwen3.5 base (scorer − base, nats/char; positive = less expected than to the base):",
        "",
        "| author | " + " | ".join(SCORERS[s] for s in SCORERS if s != base) + " |",
        "|---|" + "---:|" * (len(SCORERS) - 1),
    ]
    for a in authors:
        lines.append(
            f"| {a} | " + " | ".join(f"{table[(s, a)] - table[(base, a)]:+.3f}" for s in SCORERS if s != base) + " |"
        )

    def position_bucket(i):
        for lo, hi in zip(POSITION_EDGES, POSITION_EDGES[1:]):
            if lo <= i < hi:
                return f"{lo}–{hi}"
        return f"{POSITION_EDGES[-1]}+"

    strong = [a for a in authors if ifbench[a] >= 0.34]
    weak = [a for a in authors if ifbench[a] < 0.26]
    for title, key in [
        ("character position in the response", lambda tid, i: position_bucket(i)),
        ("character class", lambda tid, i: char_class(traces[tid]["response"], i)),
    ]:
        buckets = sorted(
            {key(tid, i) for tid in traces for i in common[tid]},
            key=lambda b: (POSITION_EDGES.index(int(b.split("–")[0])) if "–" in b else 0, b),
        )
        lines += [
            "",
            f"By {title}: Qwen3.6 post − Qwen3.5 base, and Snowball base − Qwen3.5 base (nats/char), "
            f"for strong (IFBench ≥ 0.34: {', '.join(strong)}) and weak (< 0.26: {', '.join(weak)}) authors:",
            "",
            "| bucket | share of chars | Q3.6−base strong | Q3.6−base weak | Snowball−base strong | Snowball−base weak |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        total_chars = sum(len(common[tid]) for tid in traces)
        for b in buckets:
            sel = lambda tid, i, b=b: key(tid, i) == b
            share = sum(1 for tid in traces for i in common[tid] if key(tid, i) == b) / total_chars
            cells = []
            for scorer in ["qwen3.6-35b-a3b", "snowball-67b-a2b-base-skew8"]:
                for group in (strong, weak):
                    tids = [tid for a in group if SELF.get(scorer) != a for tid in by_author[a]]
                    cells.append(f"{mean_nll(scorer, tids, sel) - mean_nll(base, tids, sel):+.3f}")
            lines.append(f"| {b} | {share:.1%} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "Passed vs failed IFBench responses (nats/char), pooled over authors except the scorer's own:",
        "",
        "| scorer | passed | failed |",
        "|---|---:|---:|",
    ]
    for s in SCORERS:
        p = [tid for tid, t in traces.items() if t["passed"] and SELF.get(s) != t["author"]]
        f = [tid for tid, t in traces.items() if not t["passed"] and SELF.get(s) != t["author"]]
        lines.append(f"| {SCORERS[s]} | {mean_nll(s, p):.3f} | {mean_nll(s, f):.3f} |")

    # Largest per-word gaps between Qwen3.6 post and Qwen3.5 base, with context.
    gaps = []
    for tid, t in traces.items():
        if t["author"] == "qwen36":
            continue
        text, ok = t["response"], set(common[tid])
        q, b = per_char[("qwen3.6-35b-a3b", tid)], per_char[(base, tid)]
        for m in WORD.finditer(text):
            idx = [i for i in range(m.start(), m.end()) if i in ok]
            if len(idx) == m.end() - m.start():
                gap = sum(q[i] - b[i] for i in idx)
                gaps.append((gap, tid, m.start(), m.end()))
    gaps.sort()

    def example(g):
        gap, tid, s, e = g
        text = traces[tid]["response"]
        return {
            "gap_nats": round(gap, 2),
            "author": traces[tid]["author"],
            "trace_id": tid,
            "word": text[s:e],
            "position": s,
            "context": text[max(0, s - 120) : s] + "⟦" + text[s:e] + "⟧" + text[e : e + 60],
        }

    (HERE / "gaps.json").write_text(
        json.dumps(
            {
                "more_expected_to_qwen36": [example(g) for g in gaps[:60]],
                "less_expected_to_qwen36": [example(g) for g in gaps[-60:][::-1]],
            },
            indent=1,
            ensure_ascii=False,
        )
    )
    (HERE / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
