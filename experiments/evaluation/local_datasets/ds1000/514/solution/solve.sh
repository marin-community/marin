#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
plt.minorticks_on()
ax = plt.gca()
ax.tick_params(axis="y", which="minor", tick1On=False)
PY
