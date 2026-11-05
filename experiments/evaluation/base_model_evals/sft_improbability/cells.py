# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-character decomposition of the SFT gap into segment x position x character class.

Each Qwen token's observed gap (post - base) and surprisal-matched predicted gap
(reference curve at its base NLL, see surprisal_match.py) are spread over its characters.
A cell's contribution is its summed gap divided by all SFT characters, so contributions add
up to the SFT's gap. Text is the line-start leak cut (the +0.035 Qwen3.6 figure); the
`fused-leak tail` segment is what lies between a fused role word (`...built.user\\n`) and
the line-start cut.
"""

import re
import sys
import unicodedata
from collections import defaultdict

import numpy as np
from fulldata import QS, load_full
from surprisal_match import EDGES, NB, POS, REF, SFT, curve_from, per_prompt_tables, tokens_in

LIST_LINE = re.compile(r"(?m)^[ \t]*(?:[-*+•]|\d+[.)])[ \t]")


def ext_class(text: str, i: int, list_marker: np.ndarray) -> str:
    ch = text[i]
    line_start = i == 0 or text[i - 1] == "\n"
    if ch == "\n":
        return (
            "newline (blank-line run)"
            if (i > 0 and text[i - 1] == "\n") or text[i + 1 : i + 2] == "\n"
            else "newline (single)"
        )
    if list_marker[i]:
        return "list marker"
    if ch == "#":
        return "header #"
    if ch == "*":
        return "bold/italic *"
    if ch in "|`>_~" or (line_start and ch in "-+"):
        return "other markdown"
    if ch in "—–":
        return "dash (em/en)"
    if ch in "‘’“”":
        return "curly quote"
    if ord(ch) > 127 and ch.isalpha():
        name = unicodedata.name(ch, "")
        if any(k in name for k in ("CJK", "HIRAGANA", "KATAKANA", "HANGUL")):
            return "CJK"
        return "non-ASCII letter"
    if ch.isdigit():
        return "digit"
    if ch.isspace():
        return "space"
    if not ch.isalnum():
        return "punctuation"
    return "word start" if (i == 0 or not text[i - 1].isalnum()) else "word interior"


def list_mask(text: str) -> np.ndarray:
    m = np.zeros(len(text), bool)
    for x in LIST_LINE.finditer(text):
        m[x.start() : x.end() - 1] = True
    return m


def char_frame(f, scorer: str, curve: np.ndarray, cut: int, base: str = QS[0]):
    """Per covered char: obs gap, pred gap, base nll, position."""
    offs, tok = tokens_in(f, cut)
    n = len(f.text)
    obs = np.full(n, np.nan)
    pred = np.full(n, np.nan)
    bnll = np.full(n, np.nan)
    b = np.clip(np.digitize(tok[base], EDGES) - 1, 0, NB - 1)
    for (s, e), g, p, bn in zip(offs, tok[scorer] - tok[base], curve[b], tok[base], strict=True):
        obs[s:e] = g / (e - s)
        pred[s:e] = p / (e - s)
        bnll[s:e] = bn / (e - s)
    return obs, pred, bnll


def pos_bin(i: int) -> str:
    for lo, hi in zip(POS, POS[1:]):
        if lo <= i < hi:
            return f"{lo}-{hi - 1 if hi == 4001 else hi}"
    return "4000+"


def cell_sums(F, scorer: str, authors, cut_attr: str):
    """{(segment, pos, cls): per-prompt arrays of [chars, obs, pred, base]} and per-prompt chars."""
    T = per_prompt_tables(F, scorer, cut_attr)
    curve = curve_from(T, np.arange(100))
    prompts = sorted({f.prompt_key for f in F})
    pidx = {p: i for i, p in enumerate(prompts)}
    cells = defaultdict(lambda: np.zeros((len(prompts), 4)))
    for f in F:
        if f.author not in authors:
            continue
        cut = getattr(f, cut_attr)
        obs, pred, bn = char_frame(f, scorer, curve, cut)
        lm = list_mask(f.text)
        p = pidx[f.prompt_key]
        for i in range(cut):
            if np.isnan(obs[i]):
                continue
            seg = "fused-leak tail" if i >= f.cut_fused else "answer"
            key = (seg, pos_bin(i), ext_class(f.text, i, lm))
            cells[key][p] += (1, obs[i], pred[i], bn[i])
    return cells


def table(cells, group, total_chars):
    agg = defaultdict(lambda: np.zeros(cells[next(iter(cells))].shape))
    for k, v in cells.items():
        agg[group(k)] += v
    return agg


def boot_contrib(arr, total, n=2000, seed=0):
    """arr: per-prompt (100,) numerator; total: per-prompt chars. Ratio-of-sums CI."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), (n, len(arr)))
    s = arr[idx].sum(1) / total[idx].sum(1)
    return np.percentile(s, [2.5, 97.5])


if __name__ == "__main__":
    F = load_full()
    scorer = sys.argv[1] if len(sys.argv) > 1 else QS[2]
    sft = cell_sums(F, scorer, {SFT}, "cut_line")
    ref = cell_sums(F, scorer, set(REF), "cut_line")
    tot = sum(v[:, 0] for v in sft.values())
    rtot = sum(v[:, 0] for v in ref.values())
    print(
        f"scorer {scorer}: SFT gap {sum(v[:, 1].sum() for v in sft.values()) / tot.sum():+.4f}"
        f"  pred {sum(v[:, 2].sum() for v in sft.values()) / tot.sum():+.4f}"
        f"  ref gap {sum(v[:, 1].sum() for v in ref.values()) / rtot.sum():+.4f}"
    )
    for title, grp in [
        ("segment", lambda k: k[0]),
        ("position (answer only)", lambda k: k[1] if k[0] == "answer" else None),
        ("class (answer only)", lambda k: k[2] if k[0] == "answer" else None),
    ]:
        a, r = table(sft, grp, tot), table(ref, grp, rtot)
        print(
            f"\n-- by {title}: share | SFT rate  ref rate | base nll/char SFT ref | contrib obs [CI] pred resid | excess-vs-ref"
        )
        for g in sorted(a, key=lambda g: -a[g][:, 1].sum()):
            if g is None:
                continue
            v = a[g]
            rv = r.get(g, np.zeros_like(v))
            ch = v[:, 0].sum()
            rate = v[:, 1].sum() / ch
            rrate = rv[:, 1].sum() / max(rv[:, 0].sum(), 1)
            lo, hi = boot_contrib(v[:, 1], tot)
            c_obs = v[:, 1].sum() / tot.sum()
            c_pred = v[:, 2].sum() / tot.sum()
            excess = (v[:, 1].sum() - rrate * ch) / tot.sum()
            print(
                f"{g:26s} {ch / tot.sum():6.3f} {rv[:, 0].sum() / rtot.sum():6.3f} | {rate:+.3f} {rrate:+.3f} | "
                f"{v[:, 3].sum() / ch:.3f} {rv[:, 3].sum() / max(rv[:, 0].sum(), 1):.3f} | {c_obs:+.4f} [{lo:+.4f},{hi:+.4f}] "
                f"{c_pred:+.4f} {c_obs - c_pred:+.4f} | {excess:+.4f}"
            )
    # pos x class grid of SFT contributions and residuals (answer only)
    print("\n-- answer pos x class: contrib obs / resid (x1e-4)")
    poss = [pos_bin(p) for p in POS[:-1]]
    classes = sorted({k[2] for k in sft}, key=lambda c: -sum(v[:, 1].sum() for k, v in sft.items() if k[2] == c))
    print(" " * 26 + "".join(f"{p:>14s}" for p in poss))
    for c in classes:
        row = []
        for p in poss:
            v = sft.get(("answer", p, c))
            row.append(
                ""
                if v is None
                else f"{1e4 * v[:, 1].sum() / tot.sum():+6.0f}/{1e4 * (v[:, 1].sum() - v[:, 2].sum()) / tot.sum():+5.0f}"
            )
        print(f"{c:26s}" + "".join(f"{x:>14s}" for x in row))
