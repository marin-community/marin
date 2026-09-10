#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df):
    uniq_indx = (df.sort_values(by="bank", na_position='last').dropna(subset=['firstname', 'lastname', 'email'])
             .applymap(lambda s: s.lower() if type(s) == str else s)
             .applymap(lambda x: x.replace(" ", "") if type(x) == str else x)
             .drop_duplicates(subset=['firstname', 'lastname', 'email'], keep='first')).index
    return df.loc[uniq_indx]

result = g(df.copy())

PY
