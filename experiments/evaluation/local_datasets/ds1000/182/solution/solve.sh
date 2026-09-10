#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(dict, df):
    df["Date"] = df["Member"].apply(lambda x: dict.get(x)).fillna(np.NAN)
    for i in range(len(df)):
        if df.loc[i, 'Member'] not in dict.keys():
            df.loc[i, 'Date'] = '17/8/1926'
    return df

df = g(dict.copy(),df.copy())

PY
