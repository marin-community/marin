#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
fig = plt.figure(constrained_layout=True)
axs = fig.subplots(1, 2)
for ax in axs.flat:
    ax.plot(x, y)
fig.suptitle("Figure")
PY
