#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df):
    df.loc[df['name'].str.split().str.len() >= 3, 'middle_name'] = df['name'].str.split().str[1:-1]
    for i in range(len(df)):
        if len(df.loc[i, 'name'].split()) >= 3:
            l = df.loc[i, 'name'].split()[1:-1]
            s = l[0]
            for j in range(1,len(l)):
                s += ' '+l[j]
            df.loc[i, 'middle_name'] = s
    df.loc[df['name'].str.split().str.len() >= 2, 'last_name'] = df['name'].str.split().str[-1]
    df.loc[df['name'].str.split().str.len() >= 2, 'name'] = df['name'].str.split().str[0]
    df.rename(columns={'name': 'first name'}, inplace=True)
    return df

df = g(df.copy())

PY
