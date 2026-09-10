#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df, X):
    t = df['date']
    df['date'] = pd.to_datetime(df['date'])
    filter_ids = [0]
    last_day = df.loc[0, "date"]
    for index, row in df[1:].iterrows():
        if (row["date"] - last_day).days > X:
            filter_ids.append(index)
            last_day = row["date"]
    df['date'] = t
    return df.loc[filter_ids, :]

result = g(df.copy(), X)

PY
