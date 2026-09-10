#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
from matplotlib import lines

styles = lines.lineMarkers
nstyles = len(styles)
for i, sty in enumerate(styles):
    y = np.random.randn(*x.shape)
    plt.plot(x, y, marker=sty)
PY
