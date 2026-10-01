# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Write rows_step{k}.pkl holding one profiled step each (rows inside [launch_k, launch_k+1))."""
import os
import pickle
import sys

path = sys.argv[1]
d = pickle.load(open(path, "rb"))
L = d["launches"]
rows = d["rows"]
end = max(r[3] for r in rows if r[6] == "jit_train_step")
bounds = list(zip(L, [*L[1:], end]))
for k, (lo, hi) in enumerate(bounds):
    sub = [r for r in rows if lo <= r[2] < hi]
    pickle.dump({"rows": sub, "launches": [lo]}, open(os.path.join(os.path.dirname(path), f"rows_step{k}.pkl"), "wb"))
    print(k, len(sub), round((hi - lo) * 1e-12, 3))
