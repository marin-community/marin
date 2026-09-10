#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
fig, ax = plt.subplots()
ax.plot(x, y)
ax.axis([1, 1000, 1, 1000])
ax.loglog()

from matplotlib.ticker import ScalarFormatter

for axis in [ax.xaxis, ax.yaxis]:
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    axis.set_major_formatter(formatter)
PY
