#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
def g(corr):
    corr_triu = corr.where(~np.tril(np.ones(corr.shape)).astype(bool))
    corr_triu = corr_triu.stack()
    corr_triu.name = 'Pearson Correlation Coefficient'
    corr_triu.index.names = ['Col1', 'Col2']
    return corr_triu[corr_triu > 0.3].to_frame()

result = g(corr.copy())

PY
