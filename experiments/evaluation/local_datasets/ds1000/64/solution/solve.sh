#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df):
    df['ID'] = df["name"].map(str) +"-"+ df["a"].map(str)
    cnt = 0
    F = {}
    for i in range(len(df)):
        if df['ID'].iloc[i] not in F.keys():
            cnt += 1
            F[df['ID'].iloc[i]] = cnt
        df.loc[i,'ID'] = F[df.loc[i,'ID']]
    del df['name']
    del df['a']
    df = df[['ID', 'b', 'c']]
    return df

result = g(df.copy())

PY
