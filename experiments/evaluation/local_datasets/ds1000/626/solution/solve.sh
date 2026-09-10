#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
fig, ax = plt.subplots(2, 1)
(l1,) = ax[0].plot(x, y, color="red", label="y")
(l2,) = ax[1].plot(a, z, color="blue", label="z")
ax[0].legend([l1, l2], ["z", "y"])
PY
