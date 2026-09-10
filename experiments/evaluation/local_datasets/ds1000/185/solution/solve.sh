#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df):
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%y')
    y = df['Date'].dt.year
    m = df['Date'].dt.month


    df['Count_d'] = df.groupby('Date')['Date'].transform('size')
    df['Count_m'] = df.groupby([y, m])['Date'].transform('size')
    df['Count_y'] = df.groupby(y)['Date'].transform('size')
    return df

df = g(df.copy())

PY
