#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
result = np.mean([a, b, c], axis=0)

PY
