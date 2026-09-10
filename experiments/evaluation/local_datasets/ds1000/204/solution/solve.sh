#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
    cols = list(df)[1:]
    for idx in df.index:
        s = 0
        cnt = 0
        for col in cols:
            if df.loc[idx, col] != 0:
                cnt = min(cnt+1, 2)
                s = (s + df.loc[idx, col]) / cnt
            df.loc[idx, col] = s
    result = df

    return result

PY
