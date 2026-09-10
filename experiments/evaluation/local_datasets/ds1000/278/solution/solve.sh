#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df):
    to_delete = ['2020-02-17', '2020-02-18']
    return df[~(df.index.strftime('%Y-%m-%d').isin(to_delete))]

result = g(df.copy())

PY
