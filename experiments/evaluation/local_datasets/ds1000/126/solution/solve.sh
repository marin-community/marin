#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
import numpy as np
def g(df):
    df["keywords_all"] = df.filter(like='keyword').apply(lambda x: '-'.join(x.dropna()), axis=1)
    for i in range(len(df)):
        df.loc[i, "keywords_all"] = df.loc[i, "keywords_all"][::-1]
    return df

df = g(df.copy())

PY
