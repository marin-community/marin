#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
result = np.unravel_index(a.argmax(), a.shape)

PY
