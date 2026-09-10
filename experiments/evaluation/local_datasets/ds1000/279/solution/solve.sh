#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df):
    to_delete = ['2020-02-17', '2020-02-18']
    df = df[~(df.index.strftime('%Y-%m-%d').isin(to_delete))]
    df.index = df.index.strftime('%d-%b-%Y %A')
    return df

result = g(df.copy())

PY
