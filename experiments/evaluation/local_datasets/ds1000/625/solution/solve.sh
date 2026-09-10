#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
# Position of bars on x-axis
ind = np.arange(len(blue_bar))

# Figure size
plt.figure(figsize=(10, 5))

# Width of a bar
width = 0.3
plt.bar(ind, blue_bar, width, label="Blue bar label")
plt.bar(ind + width, orange_bar, width, label="Orange bar label")
PY
