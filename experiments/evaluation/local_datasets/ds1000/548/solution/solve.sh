#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
ax.lines[0].set_linestyle("dashed")
PY
