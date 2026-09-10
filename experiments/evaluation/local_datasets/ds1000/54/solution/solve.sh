#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df):
    return df.mask((df == df.min()).cumsum().astype(bool))[::-1].idxmax()

result = g(df.copy())

PY
