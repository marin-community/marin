#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
import math
def g(df):
    return df.join(df.apply(lambda x: 1/(1+math.e**(-x))).add_prefix('sigmoid_'))

result = g(df.copy())

PY
