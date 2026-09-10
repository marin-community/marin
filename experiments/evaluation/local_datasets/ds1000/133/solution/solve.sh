#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df):
    cols = list(df.filter(like='col'))
    df['index_original'] = df.groupby(cols)[cols[0]].transform('idxmin')
    return df[df.duplicated(subset=cols, keep='first')]

result = g(df.copy())

PY
