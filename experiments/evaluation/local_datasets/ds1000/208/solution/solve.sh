#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df):
    label = []
    for i in range(len(df)-1):
        if df.loc[i, 'Close'] > df.loc[i+1, 'Close']:
            label.append(1)
        elif df.loc[i, 'Close'] == df.loc[i+1, 'Close']:
            label.append(0)
        else:
            label.append(-1)
    label.append(1)
    df['label'] = label
    df["DateTime"] = df["DateTime"].dt.strftime('%d-%b-%Y')
    return df

df = g(df.copy())

PY
