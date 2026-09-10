#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df):
    rows = df.max(axis=1) == 2
    cols = df.max(axis=0) == 2
    df.loc[rows] = 0
    df.loc[:,cols] = 0
    return df

result = g(df.copy())

PY
