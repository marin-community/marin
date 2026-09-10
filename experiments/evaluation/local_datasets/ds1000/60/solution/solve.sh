#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(df):
    df.dt = pd.to_datetime(df.dt)
    result = df.set_index(['dt', 'user']).unstack(fill_value=-11414).asfreq('D', fill_value=-11414)
    for col in result.columns:
        Max = result[col].max()
        for idx in result.index:
            if result.loc[idx, col] == -11414:
                result.loc[idx, col] = Max
    result = result.stack().sort_index(level=1).reset_index()
    result['dt'] = result['dt'].dt.strftime('%d-%b-%Y')
    return result

result = g(df.copy())

PY
