#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df):
    df = df.set_index('cat')
    res = df.div(df.sum(axis=1), axis=0)
    return res.reset_index()

df = g(df.copy())

PY
