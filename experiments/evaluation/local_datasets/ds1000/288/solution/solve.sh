#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df):
    df['bar'] = df['bar'].replace("NULL", 0)
    res = df.groupby(["id1", "id2"])[["foo", "bar"]].mean()
    return res

result = g(df.copy())

PY
