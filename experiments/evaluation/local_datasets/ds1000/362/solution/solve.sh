#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
mask = (del_col <= a.shape[1])
del_col = del_col[mask] - 1
result = np.delete(a, del_col, axis=1)


PY
