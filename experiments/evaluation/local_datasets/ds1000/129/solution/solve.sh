#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df):
    for i in range(len(df)):
        tot = 0
        if i != 0:
            if df.loc[i, 'UserId'] == df.loc[i-1, 'UserId']:
                continue
        for j in range(len(df)):
            if df.loc[i, 'UserId'] == df.loc[j, 'UserId']:
                tot += 1
        l = int(0.2*tot)
        dfupdate = df.iloc[i:i+tot].sample(l, random_state=0)
        dfupdate.Quantity = 0
        df.update(dfupdate)
    return df

df = g(df.copy())

PY
