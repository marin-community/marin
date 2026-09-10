#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df):
    l = []
    for i in range(2*(len(df) // 5) + (len(df) % 5) // 3 + 1):
        l.append(0)
    for i in range(len(df)):
        idx = 2*(i // 5) + (i % 5) // 3
        if i % 5 < 3:
            l[idx] += df['col1'].iloc[i]
        elif i % 5 == 3:
            l[idx] = df['col1'].iloc[i]
        else:
            l[idx] = (l[idx] + df['col1'].iloc[i]) / 2
    return pd.DataFrame({'col1': l})

result = g(df.copy())

PY
