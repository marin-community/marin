#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def get_relation(df, col1, col2):
    first_max = df[[col1, col2]].groupby(col1).count().max()[0]
    second_max = df[[col1, col2]].groupby(col2).count().max()[0]
    if first_max==1:
        if second_max==1:
            return 'one-to-one'
        else:
            return 'one-to-many'
    else:
        if second_max==1:
            return 'many-to-one'
        else:
            return 'many-to-many'


from itertools import product
def g(df):
    result = []
    for col_i, col_j in product(df.columns, df.columns):
        if col_i == col_j:
            continue
        result.append(col_i+' '+col_j+' '+get_relation(df, col_i, col_j))
    return result

result = g(df.copy())

PY
