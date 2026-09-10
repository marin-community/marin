#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
col = ( A.shape[0] // ncol) * ncol
B = A[len(A)-col:][::-1]
B = np.reshape(B, (-1, ncol))

PY
