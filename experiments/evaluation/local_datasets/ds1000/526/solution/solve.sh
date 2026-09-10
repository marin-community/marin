#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
plt.plot(x, y, "+", mew=7, ms=20)
PY
