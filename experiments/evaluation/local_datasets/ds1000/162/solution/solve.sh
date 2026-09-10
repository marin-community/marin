#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df):
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=['Caps','Lower'])
    return df

df = g(df.copy())

PY
