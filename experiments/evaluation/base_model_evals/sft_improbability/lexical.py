# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Where the SFT's extra base surprisal sits: out-of-vocabulary words, non-Latin runs,
word-start vs word-interior tokens, and the >=3-nat token tail."""

import re
from collections import Counter, defaultdict

import numpy as np
from fulldata import QS, load_full
from surprisal_match import REF, SFT, tokens_in

F = load_full()
AUTHORS = sorted({f.author for f in F})
WEB2 = {w.strip().lower() for w in open("/usr/share/dict/words")}
WORD = re.compile(r"[A-Za-z]+")
used = defaultdict(set)
for f in F:
    for w in WORD.findall(f.text[: f.cut_line]):
        used[w.lower()].add(f.author)
SNOW = {SFT, "snowballbase"}


def known(w, author):
    w = w.lower()
    if w in WEB2 or len(used[w] - SNOW - {author}) >= 2:
        return True
    for suf in ("s", "es", "ed", "d", "ing", "ly", "er", "est", "ness", "ment", "ies"):
        if w.endswith(suf) and (w[: -len(suf)] in WEB2 or w[: -len(suf)] + "y" in WEB2 or w[: -len(suf)] + "e" in WEB2):
            return True
    return False


print("author         OOV-word rate  OOV char share  gap/char q36 OOV vs rest   q35 OOV vs rest   base/char OOV vs rest")
oov_examples = Counter()
for a in AUTHORS:
    nw = no = 0
    acc = np.zeros((2, 4))  # [oov, rest] x [chars, g36, g35, base]
    for f in F:
        if f.author != a:
            continue
        t = f.text[: f.cut_line]
        mark = np.zeros(len(t), bool)
        for m in WORD.finditer(t):
            if len(m.group()) < 3:
                continue
            nw += 1
            if not known(m.group(), a):
                no += 1
                mark[m.start() : m.end()] = True
                if a == SFT:
                    oov_examples[m.group()] += 1
        offs, tok = tokens_in(f, f.cut_line)
        for (s, e), b, p5, p6 in zip(offs, tok[QS[0]], tok[QS[1]], tok[QS[2]]):
            k = 0 if mark[s:e].any() else 1
            acc[k] += (e - s, p6 - b, p5 - b, b)
    tot = acc[:, 0].sum()
    print(
        f"{a:14s} {no/nw:.4f}        {acc[0,0]/tot:.4f}        {acc[0,1]/acc[0,0]:+.3f} {acc[1,1]/acc[1,0]:+.3f}     {acc[0,2]/acc[0,0]:+.3f} {acc[1,2]/acc[1,0]:+.3f}     {acc[0,3]/acc[0,0]:.2f} {acc[1,3]/acc[1,0]:.3f}"
        + (f"   OOV contrib q36 {acc[0,1]/tot:+.4f} q35 {acc[0,2]/tot:+.4f}" if a == SFT else "")
    )
print("SFT OOV examples:", [w for w, _ in oov_examples.most_common(60)])

# composition of >=3-nat tokens (fused cut)
print("\n>=3-nat (base) tokens: share of tokens, and share of their base-NLL mass by token kind")
for grp, name in (([SFT], "SFT"), (REF, "REF")):
    kinds = Counter()
    mass = Counter()
    g35 = Counter()
    ntok = 0
    for f in F:
        if f.author not in grp:
            continue
        offs, tok = tokens_in(f, f.cut_fused)
        ntok += len(offs)
        for (s, e), b, p5 in zip(offs, tok[QS[0]], tok[QS[1]]):
            if b < 3:
                continue
            piece = f.text[s:e]
            core = piece.strip()
            if not core:
                k = "whitespace/newline"
            elif any(ord(c) > 0x2E7F for c in core):
                k = "CJK/other script"
            elif core[0].isalpha() and (piece[0] == " " or s == 0 or not f.text[s - 1].isalnum()):
                k = "word start"
            elif core[0].isalpha():
                k = "word continuation"
            elif core[0].isdigit():
                k = "digit"
            elif core[0] in "#*-|`>_":
                k = "markdown"
            else:
                k = "punctuation/other"
            kinds[k] += 1
            mass[k] += b
            g35[k] += p5 - b
        # end
    tk = sum(kinds.values())
    print(f"{name}: {tk/ntok:.3f} of tokens are >=3 nats")
    for k in sorted(kinds, key=lambda k: -mass[k]):
        print(
            f"   {k:20s} tok share {kinds[k]/tk:.3f}  base mass share {mass[k]/sum(mass.values()):.3f}  q35 gap/tok {g35[k]/kinds[k]:+.3f}"
        )
