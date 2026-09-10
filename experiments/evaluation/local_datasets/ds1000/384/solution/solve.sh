#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
result = np.linalg.matrix_power(A, n)

PY
