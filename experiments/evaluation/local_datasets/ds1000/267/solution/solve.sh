#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df):
    cols = (df.columns[df.iloc[0,:].fillna('Nan') != df.iloc[8,:].fillna('Nan')]).values
    result = []
    for col in cols:
        result.append((df.loc[0, col], df.loc[8, col]))
    return result

result = g(df.copy())

PY
