#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df):
    family = []
    for i in range(len(df)):
        if df.loc[i, 'SibSp'] == 0 and df.loc[i, 'Parch'] == 0:
            family.append('No Family')
        elif df.loc[i, 'SibSp'] == 1 and df.loc[i, 'Parch'] == 1:
            family.append('Has Family')
        elif df.loc[i, 'SibSp'] == 0 and df.loc[i, 'Parch'] == 1:
            family.append('New Family')
        else:
            family.append('Old Family')
    return df.groupby(family)['Survived'].mean()

result = g(df.copy())

PY
