#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
z = np.any(np.isnan(a), axis = 1)
a = a[~z, :]


PY
