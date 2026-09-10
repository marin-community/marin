#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(C, D):
    df = pd.concat([C,D]).drop_duplicates('A', keep='last').sort_values(by=['A']).reset_index(drop=True)
    for i in range(len(C)):
        if df.loc[i, 'A'] in D.A.values:
            df.loc[i, 'dulplicated'] = True
        else:
            df.loc[i, 'dulplicated'] = False
    for i in range(len(C), len(df)):
        df.loc[i, 'dulplicated'] = False
    return df

result = g(C.copy(),D.copy())

PY
