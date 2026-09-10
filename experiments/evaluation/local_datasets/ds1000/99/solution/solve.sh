#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df):
    mask = (df.filter(like='Value').abs() > 1).any(axis=1)
    cols = {}
    for col in list(df.filter(like='Value')):
        cols[col]=col.replace("Value_","")
    df.rename(columns=cols, inplace=True)
    return df[mask]

df = g(df.copy())

PY
