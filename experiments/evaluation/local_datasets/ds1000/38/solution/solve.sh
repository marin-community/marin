#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df, row_list, column_list):
    result = df[column_list].iloc[row_list].sum(axis=0)
    return result.drop(result.index[result.argmax()])

result = g(df.copy(), row_list, column_list)

PY
