#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
fig, (ax1, ax2) = plt.subplots(nrows=2, subplot_kw=dict(frameon=False))

plt.subplots_adjust(hspace=0.0)
ax1.grid()
ax2.grid()

ax1.plot(x, y1, color="r")
ax2.plot(x, y2, color="b", linestyle="--")
PY
