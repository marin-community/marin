#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
mask = im == 0
rows = np.flatnonzero((~mask).sum(axis=1))
cols = np.flatnonzero((~mask).sum(axis=0))
if rows.shape[0] == 0:
    result = np.array([])
else:
    result = im[rows.min():rows.max()+1, cols.min():cols.max()+1]


PY
