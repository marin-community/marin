# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from tokens import QS, load_tokens

D = load_tokens()
REF = ["gemma4", "glm47flash", "kimilinear", "kolibri", "linglite", "nemotron3nano", "trinitymini"]
EDGES = np.array([0, 0.01, 0.05, 0.2, 0.5, 1, 2, 3, 5, 8, 100])


def pool(authors, s):
    b = np.concatenate([D[t][QS[0]] for t in D if D[t]["author"] in authors])
    p = np.concatenate([D[t][s] for t in D if D[t]["author"] in authors])
    return b, p


for s in QS[1:]:
    print("==", s)
    rb, rp = pool(REF, s)
    rbin = np.digitize(rb, EDGES) - 1
    curve = np.array([np.mean(rp[rbin == k] - rb[rbin == k]) for k in range(len(EDGES) - 1)])
    print("bin      " + " ".join(f"{EDGES[k]:>6.2f}" for k in range(len(EDGES) - 1)))
    print("ref gap  " + " ".join(f"{c:+6.3f}" for c in curve))
    for a in ["grugsft471", "snowballbase", "kimilinear", "trinitymini", "gemma4", "qwen3next"]:
        b, p = pool([a], s)
        k = np.digitize(b, EDGES) - 1
        chars = sum(D[t]["offsets"][:, 1].sum() - D[t]["offsets"][:, 0].sum() for t in D if D[t]["author"] == a)
        obs = (p - b).sum() / chars
        pred = curve[k].sum() / chars
        share = np.bincount(k, minlength=len(curve)) / len(k)
        cur = [np.mean(p[k == j] - b[k == j]) if (k == j).any() else np.nan for j in range(len(curve))]
        print(f"{a:13s} obs={obs:+.3f} pred_from_ref_curve={pred:+.3f}  base-nll/char={b.sum()/chars:.3f}")
        print("   own gap " + " ".join(f"{c:+6.3f}" for c in cur))
        print("   share   " + " ".join(f"{c:6.3f}" for c in share))
