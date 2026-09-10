#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
res = np.array([0])
for i in a:
    res = res ^ i
result = (((res[:,None] & (1 << np.arange(m))[::-1])) > 0).astype(int)

PY
