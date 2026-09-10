#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
from sklearn.metrics.pairwise import cosine_similarity

response = tfidf.fit_transform(df['description']).toarray()
tf_idf = response
cosine_similarity_matrix = np.zeros((len(df), len(df)))
for i in range(len(df)):
    for j in range(len(df)):
        cosine_similarity_matrix[i, j] = cosine_similarity([tf_idf[i, :]], [tf_idf[j, :]])
PY
