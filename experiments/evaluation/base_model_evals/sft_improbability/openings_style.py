# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""H1 opening conventions, H3 charset/detokenization artifacts, H6 style provenance."""

import re
from collections import Counter, defaultdict

import numpy as np
from fulldata import QS, load_full
from surprisal_match import REF, SFT, tokens_in

F = load_full()
AUTHORS = sorted({f.author for f in F})


def opener(text, prompt):
    t = text.lstrip()
    head = t[:80]
    if not t:
        return "empty"
    if re.match(r"\*\*Reflection", t):
        return "**Reflection"
    if t.startswith("#"):
        return "markdown header"
    if t.startswith("**"):
        return "bold"
    if re.match(r"([-*•]|\d+[.)])\s", t):
        return "list"
    if re.match(r"(Here('s| is| are)|Sure|Certainly|Of course|Absolutely|Below)", t):
        return "here-is preamble"
    if re.match(r"(I (can(not|'t)|must|won't|am unable)|I'm (sorry|unable))", t):
        return "refusal/meta"
    if re.match(r"(Okay|OK|Alright|Let me|Let's|The user|We need|I need|I'll|I will|To (answer|address))", t):
        return "reasoning/plan preamble"
    if re.match(r"(Perfect|Great|Nice|There now|Taking|Beyond|Ha|Yes|No,|Right|Wait|Thanks|Because)", t):
        return "continuation/conversational"
    pw = set(re.findall(r"[a-z]+", prompt.lower()[:300]))
    hw = re.findall(r"[a-z]+", head.lower())
    if hw and np.mean([w in pw for w in hw]) > 0.8 and len(hw) >= 6:
        return "echoes prompt"
    if not head[:1].isascii():
        return "non-Latin start"
    return "plain prose"


cats = defaultdict(Counter)
open_gap = defaultdict(lambda: defaultdict(list))
for f in F:
    c = opener(f.text[: f.cut_line], f.prompt)
    cats[f.author][c] += 1
    offs, tok = tokens_in(f, f.cut_line)
    sel = offs[:, 1] <= 100
    ch = (offs[sel, 1] - offs[sel, 0]).sum()
    if ch:
        for s in QS[1:]:
            open_gap[(f.author, s)][c].append(((tok[s][sel] - tok[QS[0]][sel]).sum(), ch, tok[QS[0]][sel].sum()))
allc = sorted({c for a in cats for c in cats[a]}, key=lambda c: -cats[SFT][c])
print("opener category counts (of 100)")
print(f"{'':28s}" + "".join(f"{a[:9]:>10s}" for a in AUTHORS))
for c in allc:
    print(f"{c:28s}" + "".join(f"{cats[a][c]:10d}" for a in AUTHORS))

print("\nfirst-100-char gap (nats/char) by opener, SFT vs REF pooled  [q36 | q35post | base nll/char]")
for c in allc:
    row = []
    for grp in ([SFT], REF):
        vals = [v for a in grp for v in open_gap[(a, QS[2])][c]]
        vals35 = [v for a in grp for v in open_gap[(a, QS[1])][c]]
        if not vals:
            row.append(f"{'n=0':>28s}")
            continue
        n = sum(v[1] for v in vals)
        row.append(
            f"n={len(vals):3d} {sum(v[0] for v in vals)/n:+.3f} {sum(v[0] for v in vals35)/n:+.3f} {sum(v[2] for v in vals)/n:.2f}"
        )
    print(f"{c:28s} SFT {row[0]:>30s}   REF {row[1]:>30s}")

# position curve: first 200 chars in 20-char bins, gap and base, SFT vs REF, with prompt bootstrap
print("\nposition curve (chars, q36 gap, q35 gap, base) SFT | REF")
EDG = [0, 5, 10, 20, 40, 60, 100, 150, 200, 300]
for lo, hi in zip(EDG, EDG[1:]):
    out = []
    for grp in ([SFT], REF):
        g36 = g35 = b = n = 0
        for f in F:
            if f.author not in grp:
                continue
            offs, tok = tokens_in(f, f.cut_line)
            sel = (offs[:, 0] >= lo) & (offs[:, 0] < hi)
            n += (offs[sel, 1] - offs[sel, 0]).sum()
            g36 += (tok[QS[2]][sel] - tok[QS[0]][sel]).sum()
            g35 += (tok[QS[1]][sel] - tok[QS[0]][sel]).sum()
            b += tok[QS[0]][sel].sum()
        out.append(f"{g36/n:+.3f} {g35/n:+.3f} {b/n:.2f}")
    print(f"{lo:3d}-{hi:<3d}  SFT {out[0]}  | REF {out[1]}")

print("\nH3/H6 character and style markers per 10k chars (line-start cut)")
MARK = {
    "U+FFFD": "\ufffd",
    "NBSP U+00A0": "\u00a0",
    "NNBSP U+202F": "\u202f",
    "NB-hyphen U+2011": "\u2011",
    "zero-width": "[\u200b\u200c\u200d\ufeff]",
    "em dash": "\u2014",
    "en dash": "\u2013",
    "curly quotes": "[\u2018\u2019\u201c\u201d]",
    "straight apostrophe": "'",
    "<|...|> token text": r"<\|[^|]{1,30}\|>",
    "<think>-like": r"</?think|start_think|end_think",
    "mojibake â€/Ã": "â€|Ã.",
    "space before ,.;:": r" [,.;:](?=\s)",
    "double space mid-line": r"(?<=\S)  (?=\S)",
    "trailing 2-space": r"  \n",
    "lower-upper fused": r"[a-z]{2,}[A-Z][a-z]",
    "sentence fused (.X)": r"[a-z]\.[A-Z][a-z]",
    "**bold**": r"\*\*",
    "header line": r"(?m)^#{1,6} ",
    "list line": r"(?m)^\s*([-*•]|\d+[.)]) ",
    "LaTeX $": r"\$",
    "\\( or \\[": r"\\[(\[]",
    "Let's / let me": r"\b(Let's|let's|Let me)\b",
    "Wait,": r"\bWait\b",
    "Hmm": r"\bHmm\b",
    "verify/check": r"\b(verify|double-check|check)\b",
    "semicolon": ";",
    "exclamation": "!",
    "emoji": "[\U0001f300-\U0001faff\u2600-\u27bf]",
}
tot = Counter()
counts = defaultdict(Counter)
for f in F:
    t = f.text[: f.cut_line]
    tot[f.author] += len(t)
    for k, p in MARK.items():
        counts[f.author][k] += len(re.findall(p, t))
print(f"{'':24s}" + "".join(f"{a[:9]:>10s}" for a in AUTHORS))
for k in MARK:
    print(f"{k:24s}" + "".join(f"{1e4*counts[a][k]/tot[a]:10.2f}" for a in AUTHORS))
