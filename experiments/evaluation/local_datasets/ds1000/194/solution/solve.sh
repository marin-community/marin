#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df):
    return df.drop('var2', axis=1).join(df.var2.str.split(',', expand=True).stack().
                                        reset_index(drop=True, level=1).rename('var2'))

result = g(df.copy())

PY
