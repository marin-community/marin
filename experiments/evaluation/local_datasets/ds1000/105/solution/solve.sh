#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df):
    df.loc[df['name'].str.split().str.len() == 2, 'last_name'] = df['name'].str.split().str[-1]
    df.loc[df['name'].str.split().str.len() == 2, 'name'] = df['name'].str.split().str[0]
    df.rename(columns={'name': 'first_name'}, inplace=True)
    return df

df = g(df.copy())

PY
