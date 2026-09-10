#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def ecdf_result(x):
    xs = np.sort(x)
    ys = np.arange(1, len(xs)+1)/float(len(xs))
    return xs, ys
resultx, resulty = ecdf_result(grades)
t = (resulty > threshold).argmax()
low = resultx[0]
high = resultx[t]
PY
