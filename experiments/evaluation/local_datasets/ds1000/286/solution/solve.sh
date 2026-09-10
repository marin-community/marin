#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df):
    cols = list(df)
    Mode = df.mode(axis=1)
    df['frequent'] = df['bit1'].astype(object)
    for i in df.index:
        df.at[i, 'frequent'] = []
    for i in df.index:
        for col in list(Mode):
            if pd.isna(Mode.loc[i, col])==False:
                df.at[i, 'frequent'].append(Mode.loc[i, col])
        df.at[i, 'frequent'] = sorted(df.at[i, 'frequent'])
        df.loc[i, 'freq_count'] = (df[cols].iloc[i]==df.loc[i, 'frequent'][0]).sum()
    return df

df = g(df.copy())

PY
