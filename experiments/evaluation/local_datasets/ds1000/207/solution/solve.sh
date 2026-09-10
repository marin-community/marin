#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df):
    label = [1,]
    for i in range(1, len(df)):
        if df.loc[i, 'Close'] > df.loc[i-1, 'Close']:
            label.append(1)
        elif df.loc[i, 'Close'] == df.loc[i-1, 'Close']:
            label.append(0)
        else:
            label.append(-1)
    df['label'] = label
    return df

df = g(df.copy())

PY
