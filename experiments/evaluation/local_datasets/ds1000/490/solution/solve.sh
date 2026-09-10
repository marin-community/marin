#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
for a, t1, t2 in zip(arr, n1, n2):
    temp = a.copy()
    a[np.where(temp < t1)] = 0
    a[np.where(temp >= t2)] = 30
    a[np.logical_and(temp >= t1, temp < t2)] += 5


PY
