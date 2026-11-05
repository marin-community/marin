# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from tokens import QS, load_tokens

D = load_tokens()
rows = []
for tid, d in D.items():
    if d["author"] != "grugsft471":
        continue
    o = d["offsets"]
    t = d["text"]
    k = min(4, len(o))
    if k == 0:
        rows.append((0, 0, ""))
        continue
    end = o[k - 1][1]
    g5 = (d[QS[1]][:k] - d[QS[0]][:k]).sum()
    g6 = (d[QS[2]][:k] - d[QS[0]][:k]).sum()
    b = d[QS[0]][:k].sum()
    rows.append((g6, g5, b, t[:70].replace("\n", "⏎")))
rows.sort(key=lambda r: -r[0])
for r in rows:
    print(f"q36 {r[0]:+6.2f} q35 {r[1]:+6.2f} base {r[2]:5.2f} | {r[3] if len(r)>3 else ''}")
