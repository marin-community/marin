#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df1, df2):
    df = pd.concat([df1,df2.merge(df1[['id','city','district']], how='left', on='id')],sort=False).reset_index(drop=True)
    return df.sort_values(by=['id','date']).reset_index(drop=True)

result = g(df1.copy(),df2.copy())

PY
