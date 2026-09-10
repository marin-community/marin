#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def solution(xs, shift):
    e = np.empty_like(xs)
    for i, n in enumerate(shift):
        if n >= 0:
            e[i,:n] = np.nan
            e[i,n:] = xs[i,:-n]
        else:
            e[i,n:] = np.nan
            e[i,:n] = xs[i,-n:]
    return e
result = solution(a, shift)

PY
