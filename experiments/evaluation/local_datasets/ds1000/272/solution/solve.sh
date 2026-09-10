#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df):
    for i in df.index:
        if str(df.loc[i, 'dogs']) != '<NA>' and str(df.loc[i, 'cats']) != '<NA>':
            df.loc[i, 'dogs'] = round(df.loc[i, 'dogs'], 2)
            df.loc[i, 'cats'] = round(df.loc[i, 'cats'], 2)
    return df

df = g(df.copy())

PY
