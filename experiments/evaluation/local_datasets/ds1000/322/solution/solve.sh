#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
result = np.argwhere(a == np.min(a))

PY
