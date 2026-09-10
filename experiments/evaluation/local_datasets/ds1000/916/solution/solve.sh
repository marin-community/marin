#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
from sklearn.feature_extraction import DictVectorizer

X = [dict(enumerate(x)) for x in X]
vect = DictVectorizer(sparse=False)
new_X = vect.fit_transform(X)
PY
