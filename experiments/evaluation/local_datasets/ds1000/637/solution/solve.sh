#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
plt.plot(x, y)
plt.title(r"$\bf{Figure}$ 1")
PY
