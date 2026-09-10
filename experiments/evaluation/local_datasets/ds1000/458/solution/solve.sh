#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
result = np.triu(np.linalg.norm(a - a[:, None], axis = -1))



PY
