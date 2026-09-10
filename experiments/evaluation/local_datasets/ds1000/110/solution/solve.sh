#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
import numpy as np
def g(df):
    df['state'] = np.where((df['col2'] <= 50) & (df['col3'] <= 50), df['col1'], df[['col1', 'col2', 'col3']].max(axis=1))
    return df

df = g(df.copy())

PY
