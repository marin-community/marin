#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
import numpy as np
df['#1'] = np.roll(df['#1'], shift=1)
df['#2'] = np.roll(df['#2'], shift=-1)
PY
