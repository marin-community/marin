#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df):
    mask = (df.filter(like='Value').abs() < 1).all(axis=1)
    return df[mask]

df = g(df.copy())

PY
