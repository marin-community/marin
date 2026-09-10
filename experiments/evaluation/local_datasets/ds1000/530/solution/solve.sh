#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
l.set_markeredgecolor((0, 0, 0, 1))
PY
