#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
import numpy as np
def g(df):
    df['arrival_time'] = pd.to_datetime(df['arrival_time'].replace('0', np.nan))
    df['departure_time'] = pd.to_datetime(df['departure_time'])
    df['Duration'] = (df['arrival_time'] - df.groupby('id')['departure_time'].shift()).dt.total_seconds()
    return df

df = g(df.copy())

PY
