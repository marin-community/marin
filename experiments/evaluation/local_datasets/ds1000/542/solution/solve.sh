#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
ax = plt.gca()
ax.set_xticks([0, 1.5])
PY
