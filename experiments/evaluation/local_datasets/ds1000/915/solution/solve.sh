#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
close_buy1 = close[:-1]
m5 = ma_50[:-1]
m10 = ma_100[:-1]
ma20 = ma_200[:-1]
# b = np.concatenate([close_buy1, m5, m10, ma20], axis=1)

predict = clf.predict(pd.concat([close_buy1, m5, m10, ma20], axis=1))
PY
