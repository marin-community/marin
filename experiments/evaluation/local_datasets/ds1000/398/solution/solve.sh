#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
B = np.empty(len(A))
for k in range(0, len(B)):
    if k == 0:
        B[k] = a*A[k]
    else:
        B[k] = a*A[k] + b*B[k-1]

PY
