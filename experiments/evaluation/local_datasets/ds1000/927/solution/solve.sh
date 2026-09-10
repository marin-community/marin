#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
count = CountVectorizer(lowercase=False, token_pattern='[a-zA-Z0-9$&+:;=@#|<>^*()%-]+')
vocabulary = count.fit_transform([words])
feature_names = count.get_feature_names_out()
PY
