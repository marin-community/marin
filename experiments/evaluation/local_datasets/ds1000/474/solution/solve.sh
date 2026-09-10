#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
temp_c = c.copy()
temp_c[np.isnan(temp_c)] = 0
result = False
for arr in CNTS:
    temp = arr.copy()
    temp[np.isnan(temp)] = 0
    result |= np.array_equal(temp_c, temp) and (np.isnan(c) == np.isnan(arr)).all()


PY
