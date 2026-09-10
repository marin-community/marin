#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df):
    return df.columns[df.iloc[0,:].fillna('Nan') == df.iloc[8,:].fillna('Nan')]

result = g(df.copy())

PY
