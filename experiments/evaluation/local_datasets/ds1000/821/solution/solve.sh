#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()

df_out = df.join(
    pd.DataFrame(
        mlb.fit_transform(df.pop('Col3')),
        index=df.index,
        columns=mlb.classes_))
PY
