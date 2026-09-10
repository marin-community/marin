#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df):
    for col in df.columns:
        vc = df[col].value_counts()
        if col == 'Qu1':
            df[col] = df[col].apply(lambda x: x if vc[x] >= 3 or x == 'apple' else 'other')
        else:
            df[col] = df[col].apply(lambda x: x if vc[x] >= 2 or x == 'apple' else 'other')
    return df

result = g(df.copy())

PY
