#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
gs = gridspec.GridSpec(
    nrow,
    ncol,
    wspace=0.0,
    hspace=0.0,
    top=1.0 - 0.5 / (nrow + 1),
    bottom=0.5 / (nrow + 1),
    left=0.5 / (ncol + 1),
    right=1 - 0.5 / (ncol + 1),
)

for i in range(nrow):
    for j in range(ncol):
        ax = plt.subplot(gs[i, j])
        ax.imshow(x)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
PY
