# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Write rows_step{k}.pkl next to a rows.pkl (tfop_dump.py), one profiled step each.

Lets the per-step tools (anatomy.py, exposure.py, ragged_order.py) skip an outlier step. A single-step file
has one launch, so its span ends at the step's last kernel and excludes the tail.

Usage: python split_steps.py <rows.pkl>
"""

import os
import pickle
import sys


def main():
    path = sys.argv[1]
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    launches = data["launches"]
    rows = data["rows"]
    end = max(r[3] for r in rows if r[6] == "jit_train_step")
    for k, (lo, hi) in enumerate(zip(launches, [*launches[1:], end], strict=True)):
        sub = [r for r in rows if lo <= r[2] < hi]
        with open(os.path.join(os.path.dirname(path), f"rows_step{k}.pkl"), "wb") as fh:
            pickle.dump({"rows": sub, "launches": [lo]}, fh)
        print(k, len(sub), round((hi - lo) * 1e-12, 3))


if __name__ == "__main__":
    main()
