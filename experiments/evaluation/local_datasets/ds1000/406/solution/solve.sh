#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
uni = np.unique(index)
result = np.zeros(np.amax(index)+1)
for i in uni:
    result[i] = np.max(a[index==i])


PY
