#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df):
    return df.groupby(df.index // 3).mean()

result = g(df.copy())

PY
