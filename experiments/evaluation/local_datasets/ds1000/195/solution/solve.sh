#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df):
    return df.join(pd.DataFrame(df.var2.str.split(',', expand=True).stack().reset_index(level=1, drop=True),columns=['var2 '])).\
        drop('var2',1).rename(columns=str.strip).reset_index(drop=True)

result = g(df.copy())

PY
