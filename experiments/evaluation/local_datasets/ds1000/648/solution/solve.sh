#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(5, 5))
for ax in axes.flatten():
    ax.plot(x, y)
fig.tight_layout()
PY
