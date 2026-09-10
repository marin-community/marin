#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
categories = []
for i in range(len(df)):
    l = []
    for col in df.columns:
        if df[col].iloc[i] == 1:
            l.append(col)
    categories.append(l)
df["category"] = categories

PY
