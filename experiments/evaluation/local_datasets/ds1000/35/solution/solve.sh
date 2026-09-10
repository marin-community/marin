#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df):
    return df.groupby('group').agg(lambda x : x.head(1) if x.dtype=='object' else x.mean() if x.name.endswith('2') else x.sum())

result = g(df.copy())

PY
