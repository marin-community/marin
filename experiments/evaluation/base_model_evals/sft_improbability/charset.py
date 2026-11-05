# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""H3: zero-width chars and apostrophe style -- where they occur and their gap contribution."""

import re
from collections import Counter

from fulldata import QS, load_full
from surprisal_match import REF, SFT, tokens_in

F = load_full()
zw = Counter()
ex = []
for f in F:
    if f.author != SFT:
        continue
    t = f.text[: f.cut_line]
    for m in re.finditer("[​‌‍﻿]", t):
        zw[(f.trace_id, hex(ord(m.group())))] += 1
        if len(ex) < 6:
            ex.append((f.trace_id, repr(t[max(0, m.start() - 30) : m.start() + 30])))
print("zero-width by response:", zw.most_common(10))
print(*ex, sep="\n")


def class_gap(pattern, authors, scorer):
    """Gap of tokens that contain a char matching pattern; per char of all text (contribution) and per matched token."""
    g = b = n = tot = 0
    for f in F:
        if f.author not in authors:
            continue
        offs, tok = tokens_in(f, f.cut_line)
        tot += (offs[:, 1] - offs[:, 0]).sum()
        for (s, e), x, y in zip(offs, tok[scorer], tok[QS[0]]):
            if re.search(pattern, f.text[s:e]):
                g += x - y
                b += y
                n += 1
    return n, g / max(n, 1), b / max(n, 1), g / tot


for name, pat in [
    ("zero-width", "[​‌‍﻿]"),
    ("straight apostrophe", "'"),
    ("curly apostrophe", "’"),
    ("em dash", "—"),
    ("CJK", "[぀-ヿ一-鿿가-힯]"),
]:
    for s in QS[1:]:
        r = [class_gap(pat, [SFT], s), class_gap(pat, REF, s)]
        print(
            f"{name:20s} {s:18s} SFT n={r[0][0]:5d} gap/tok {r[0][1]:+.3f} base/tok {r[0][2]:.2f} contrib {r[0][3]:+.4f} | REF n={r[1][0]:5d} gap/tok {r[1][1]:+.3f} base/tok {r[1][2]:.2f}"
        )
