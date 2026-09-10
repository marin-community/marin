#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df):
    df['TIME'] = pd.to_datetime(df['TIME'])
    df['TIME'] = df['TIME'].dt.strftime('%d-%b-%Y %a %T')
    df['RANK'] = df.groupby('ID')['TIME'].rank(ascending=False)
    return df

df = g(df.copy())

PY
