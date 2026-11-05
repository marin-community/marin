# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Token-level table for the three Qwen-tokenizer scorers (identical tokenization), leak-cut."""

import gzip
import json
import pickle

import numpy as np
from common import HERE, TL, TRACES, leak_cut

QS = ["qwen3.5-35b-a3b-base", "qwen3.5-35b-a3b", "qwen3.6-35b-a3b"]
CACHE = HERE / "tokens.pkl"


def load_tokens():
    if CACHE.exists():
        return pickle.loads(CACHE.read_bytes())
    raw = {t["trace_id"]: t for t in map(json.loads, TRACES.read_text().splitlines())}
    recs = {s: {} for s in QS}
    for s in QS:
        for line in gzip.open(TL / "data" / f"{s}.jsonl.gz", "rt"):
            r = json.loads(line)
            recs[s][r["trace_id"]] = r
    out = {}
    mism = 0
    for tid, t in raw.items():
        offs = [tuple(o) for o in recs[QS[0]][tid]["offsets"]]
        n = min(len(recs[s][tid]["offsets"]) for s in QS)
        for s in QS[1:]:
            if [tuple(o) for o in recs[s][tid]["offsets"][:n]] != offs[:n]:
                mism += 1
        cut = leak_cut(t["response"])
        keep = [i for i in range(n) if offs[i][1] <= cut and offs[i][1] > offs[i][0]]
        out[tid] = dict(
            author=t["author"],
            text=t["response"],
            cut=cut,
            offsets=np.array([offs[i] for i in keep]).reshape(-1, 2),
            **{s: np.array([recs[s][tid]["nll"][i] for i in keep]) for s in QS},
        )
    print("offset mismatches:", mism)
    CACHE.write_bytes(pickle.dumps(out))
    return out


if __name__ == "__main__":
    d = load_tokens()
    print(len(d))
