#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
ax = plt.gca()
ax.set(xlabel=None)
PY
