#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df):
    df[['time', 'number']] = df.duration.str.extract(r'\s*(.*)(\d+)', expand=True)
    for i in df.index:
        df.loc[i, 'time'] = df.loc[i, 'time'].strip()
    df['time_days'] = df['time'].replace(['year', 'month', 'week', 'day'], [365, 30, 7, 1], regex=True)
    return df

df = g(df.copy())

PY
