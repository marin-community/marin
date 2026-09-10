#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df):
    df1 = df.groupby('Date').agg(lambda x: (x%2==0).sum())
    df2 = df.groupby('Date').agg(lambda x: (x%2==1).sum())
    return df1, df2

result1, result2 = g(df.copy())

PY
