#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
selection = np.ones((len(a), 1), dtype = bool)
selection[1:] = a[1:] != a[:-1]
selection &= a != 0
result = a[selection].reshape(-1, 1)


PY
