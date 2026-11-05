# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-response features (H3-H6) and how the SFT gap distributes across responses."""

import re
import unicodedata

import numpy as np
from fulldata import QS, load_full
from scipy.stats import spearmanr
from surprisal_match import REF, SFT, tokens_in

F = load_full()
WORD = re.compile(r"[A-Za-z]+")
AUTHORS = sorted({f.author for f in F})


def script(ch):
    if ch.isascii():
        return "latin" if ch.isalpha() else None
    n = unicodedata.name(ch, "")
    for k in ("CJK", "HIRAGANA", "KATAKANA", "HANGUL", "CYRILLIC", "ARABIC", "DEVANAGARI", "GREEK", "THAI", "HEBREW"):
        if k in n:
            return k
    return "latin" if "LATIN" in n else None


def feats(text, prompt):
    words = text.split()
    ng = [tuple(words[i : i + 4]) for i in range(len(words) - 3)]
    rep4 = 1 - len(set(ng)) / len(ng) if ng else 0.0
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    rep_lines = 1 - len(set(lines)) / len(lines) if lines else 0.0
    scr = [script(c) for c in text]
    scr = [s for s in scr if s]
    switches = sum(a != b for a, b in zip(scr, scr[1:]))
    nonlatin = sum(s != "latin" for s in scr) / max(len(scr), 1)
    pscr = [script(c) for c in prompt]
    p_nonlatin = sum(s not in (None, "latin") for s in pscr) / max(sum(s is not None for s in pscr), 1)
    aw = WORD.findall(text)
    longw = sum(len(w) >= 16 for w in aw) / max(len(aw), 1)
    camel = len(re.findall(r"[a-z]{2,}[A-Z][a-z]", text)) / max(len(text), 1) * 1000
    return dict(
        len=len(text),
        rep4=rep4,
        rep_lines=rep_lines,
        switches_per_k=1000 * switches / max(len(text), 1),
        nonlatin=nonlatin,
        lang_mismatch=float(nonlatin > 0.3 and p_nonlatin < 0.05),
        longword=longw,
        camel_per_k=camel,
        md_lines=sum(bool(re.match(r"\s*(#|[-*+]\s|\d+[.)]\s|\*\*)", l)) for l in lines) / max(len(lines), 1),
    )


def resp_stats(cut_attr):
    rows = []
    for f in F:
        cut = getattr(f, cut_attr)
        offs, tok = tokens_in(f, cut)
        ch = (offs[:, 1] - offs[:, 0]).sum()
        if ch == 0:
            continue
        r = dict(
            author=f.author,
            tid=f.trace_id,
            pk=f.prompt_key,
            passed=f.passed,
            chars=ch,
            base=tok[QS[0]].sum(),
            g35=(tok[QS[1]] - tok[QS[0]]).sum(),
            g36=(tok[QS[2]] - tok[QS[0]]).sum(),
            hi3=(tok[QS[0]] >= 3).mean(),
            text=f.text[:cut],
        )
        r.update(feats(f.text[:cut], f.prompt))
        rows.append(r)
    return rows


for cut in ["cut_line", "cut_fused"]:
    R = resp_stats(cut)
    print(f"\n=========== {cut}")
    print("author          len  base/ch hi3tok rep4  replines sw/k nonlatin mismatch longw camel/k mdlines")
    for a in AUTHORS:
        rs = [r for r in R if r["author"] == a]
        m = lambda k: np.mean([r[k] for r in rs])
        print(
            f"{a:14s} {np.median([r['len'] for r in rs]):5.0f} {sum(r['base'] for r in rs)/sum(r['chars'] for r in rs):.3f} {m('hi3'):.3f} "
            f"{m('rep4'):.3f} {m('rep_lines'):.3f} {m('switches_per_k'):5.2f} {m('nonlatin'):.3f} {sum(r['lang_mismatch'] for r in rs):3.0f} "
            f"{m('longword'):.4f} {m('camel_per_k'):.3f} {m('md_lines'):.2f}"
        )
    S = [r for r in R if r["author"] == SFT]
    for g in ["g36", "g35"]:
        num = np.array([r[g] for r in S])
        den = np.array([r["chars"] for r in S])
        order = np.argsort(-num)
        tot = num.sum()
        print(
            f"\n{g}: SFT gap {tot/den.sum():+.4f}; top-k dropped:",
            " ".join(f"{k}:{num[order[k:]].sum()/den[order[k:]].sum():+.4f}" for k in (1, 3, 5, 10, 20)),
        )
        print(
            f"   share of total gap nats from top 10% responses: {num[order[:10]].sum()/tot:.2f}; responses with positive gap: {(num>0).mean():.2f}"
        )
        rate = num / den
        print(
            "   top-10 by gap nats:",
            [(S[i]["tid"], round(num[i], 1), round(rate[i], 3), S[i]["len"]) for i in order[:10]],
        )
        for k in ["base", "len", "rep4", "rep_lines", "switches_per_k", "nonlatin", "md_lines", "hi3"]:
            x = np.array([r[k] / r["chars"] if k == "base" else r[k] for r in S])
            rho = spearmanr(x, rate).statistic
            print(f"   spearman(per-response {g} rate, {k}) = {rho:+.2f}")
        # pooled across all non-self authors: does per-response base nll predict rate?
        allr = [r for r in R if r["author"] not in (SFT, "qwen36")]
        print(
            f"   pooled non-SFT responses: spearman(rate, base/ch) = {spearmanr([r['base']/r['chars'] for r in allr], [r[g]/r['chars'] for r in allr]).statistic:+.2f}"
        )
    # paired by prompt: SFT minus mean of REF, against SFT-minus-ref base nll
    for g in ["g36", "g35"]:
        d_gap, d_base = [], []
        for pk in sorted({r["pk"] for r in R}):
            s = [r for r in S if r["pk"] == pk]
            rr = [r for r in R if r["pk"] == pk and r["author"] in REF]
            if not s:
                continue
            d_gap.append(s[0][g] / s[0]["chars"] - np.mean([r[g] / r["chars"] for r in rr]))
            d_base.append(s[0]["base"] / s[0]["chars"] - np.mean([r["base"] / r["chars"] for r in rr]))
        print(
            f"paired-by-prompt {g}: spearman(SFT-ref gap rate, SFT-ref base nll/char) = {spearmanr(d_gap, d_base).statistic:+.2f}; SFT>ref on {np.mean(np.array(d_gap)>0):.2f} of prompts"
        )
