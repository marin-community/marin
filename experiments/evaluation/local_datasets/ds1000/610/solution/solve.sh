#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
plt.scatter(x, y, linewidth=0, hatch="|")
PY
