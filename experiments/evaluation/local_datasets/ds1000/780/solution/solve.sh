#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
indices = [('1415777_at Pnliprp1', 'data'), ('1415777_at Pnliprp1', 'zscore'), ('1415805_at Clps', 'data'), ('1415805_at Clps', 'zscore'), ('1415884_at Cela3b', 'data'), ('1415884_at Cela3b', 'zscore')]
indices = pd.MultiIndex.from_tuples(indices)
df2 = pd.DataFrame(data=stats.zscore(df, axis = 0), index=df.index, columns=df.columns)
df3 = pd.concat([df, df2], axis=1).to_numpy().reshape(-1, 3)
result = pd.DataFrame(data=np.round(df3, 3), index=indices, columns=df.columns)



PY
