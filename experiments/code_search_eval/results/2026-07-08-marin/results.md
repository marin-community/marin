# Code search evaluation — 150 queries

Recall@k (gold file in top-k) / judge-hit@k (agent says a snippet answers the need):

| engine | R@1 | R@3 | R@5 | R@10 | J@1 | J@3 | J@5 | J@10 | MRR | tok@5 | build s | idx MB | ms/q |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense | 0.31 | 0.49 | 0.53 | 0.63 | 0.42 | 0.67 | 0.77 | 0.91 | 0.41 | 1882 | 2524.4 | 30.1 | 12.6 |
| bm25 | 0.22 | 0.33 | 0.37 | 0.52 | 0.27 | 0.50 | 0.63 | 0.81 | 0.30 | 2012 | 4.28 | 15.5 | 2.3 |
| ripgrep | 0.15 | 0.28 | 0.36 | 0.50 | 0.19 | 0.42 | 0.56 | 0.79 | 0.25 | 5865 | - | 0.0 | 222.7 |
