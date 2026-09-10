#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
slopes = []
for col in df1.columns:
    if col == "Time":
        continue
    mask = ~np.isnan(df1[col])
    x = np.atleast_2d(df1.Time[mask].values).T
    y = np.atleast_2d(df1[col][mask].values).T
    reg = LinearRegression().fit(x, y)
    slopes.append(reg.coef_[0])
slopes = np.array(slopes).reshape(-1)
PY
