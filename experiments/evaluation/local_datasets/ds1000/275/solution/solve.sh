#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df, list_of_my_columns):
    df['Avg'] = df[list_of_my_columns].mean(axis=1)
    df['Min'] = df[list_of_my_columns].min(axis=1)
    df['Max'] = df[list_of_my_columns].max(axis=1)
    df['Median'] = df[list_of_my_columns].median(axis=1)
    return df

df = g(df.copy(),list_of_my_columns.copy())

PY
