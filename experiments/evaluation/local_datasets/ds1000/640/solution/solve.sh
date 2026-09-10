#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
plt.scatter(x, y, clip_on=False)
PY
