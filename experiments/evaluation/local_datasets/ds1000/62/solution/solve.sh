#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df):
    F = {}
    cnt = 0
    for i in range(len(df)):
        if df['a'].iloc[i] not in F.keys():
            cnt += 1
            F[df['a'].iloc[i]] = cnt
        df.loc[i, 'a'] = F[df.loc[i, 'a']]
    return df

result = g(df.copy())

PY
