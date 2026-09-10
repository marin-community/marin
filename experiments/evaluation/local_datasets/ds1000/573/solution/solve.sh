#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
plt.axvline(x=0.22058956)
plt.axvline(x=0.33088437)
plt.axvline(x=2.20589566)
PY
