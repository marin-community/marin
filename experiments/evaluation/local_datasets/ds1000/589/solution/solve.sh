#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
plt.plot(*zip(*points))
plt.yscale("log")
PY
