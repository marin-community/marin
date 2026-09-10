#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df):
    for i in df.index:
        df.loc[i, 'codes'] = sorted(df.loc[i, 'codes'])
    df = df.codes.apply(pd.Series)
    cols = list(df)
    for i in range(len(cols)):
        cols[i]+=1
    df.columns = cols
    return df.add_prefix('code_')

result = g(df.copy())

PY
