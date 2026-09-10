#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def ecdf_result(x):
    xs = np.sort(x)
    ys = np.arange(1, len(xs)+1)/float(len(xs))
    return xs, ys
resultx, resulty = ecdf_result(grades)
result = np.zeros_like(eval, dtype=float)
for i, element in enumerate(eval):
    if element < resultx[0]:
        result[i] = 0
    elif element >= resultx[-1]:
        result[i] = 1
    else:
        result[i] = resulty[(resultx > element).argmax()-1]
PY
