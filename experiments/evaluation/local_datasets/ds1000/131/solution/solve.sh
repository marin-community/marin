#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df):
    df['index_original'] = df.groupby(['col1', 'col2']).col1.transform('idxmax')
    for i in range(len(df)):
        i = len(df) - 1 - i
        origin = df.loc[i, 'index_original']
        if i <= origin:
            continue
        if origin == df.loc[origin, 'index_original']:
            df.loc[origin, 'index_original'] = i
        df.loc[i, 'index_original'] = df.loc[origin, 'index_original']
    return df[df.duplicated(subset=['col1', 'col2'], keep='last')]

result = g(df.copy())

PY
