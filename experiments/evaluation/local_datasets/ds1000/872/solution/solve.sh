#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
from sklearn.metrics.pairwise import cosine_similarity

cosine_similarities_of_queries = []
for query in queries:
    query_tfidf = tfidf.transform([query])
    cosine_similarities_of_queries.append(cosine_similarity(query_tfidf, tfidf.transform(documents)).flatten())

PY
