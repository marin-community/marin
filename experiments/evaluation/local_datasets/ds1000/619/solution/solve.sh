#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
plt.plot(y, x)
plt.grid(color="blue", linestyle="dashed")
PY
