#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df):
    return pd.Series(', '.join(df['text'].to_list()), name='text')

result = g(df.copy())

PY
