#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df, row_list, column_list):
    return df[column_list].iloc[row_list].mean(axis=0)

result = g(df.copy(),row_list,column_list)

PY
