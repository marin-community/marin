#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
import math
def g(df):
    return df.join(df.apply(lambda x: 1/x).add_prefix('inv_')).replace(math.inf, 0)

result = g(df.copy())

PY
