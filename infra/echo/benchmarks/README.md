# Echo search benchmark

`search_queries.jsonl` is a manually judged, source-grounded evaluation set for
Echo federated search. Each line is one independent query. The set deliberately
mixes natural-language questions, identifiers, paths, current GitHub work, and
queries with no useful answer so a ranking change cannot optimize only one
surface.

Each record contains:

- `id`: stable human-readable case name.
- `split`: deterministic `dev` or held-out `test` assignment. Every fifth
  source-ordered record is test, which keeps each broad block represented in
  both splits without examining system outputs.
- `query`: text sent to search.
- `domains`: domains expected to contain useful results; an empty list means no
  result should be promoted.
- `intent`: broad query class (`how_to`, `concept`, `identifier`, `navigation`,
  `issue_recall`, `pr_recall`, `incident`, `cross_domain`, or `no_answer`).
- `source`: the repository path or GitHub URL used to write the judgment.
- `relevant`: independently useful artifacts. `target` is a repository-relative
  path for files, a canonical GitHub URL for issues and pull requests, or a
  `wiki:<id>` identifier. `grade` is 3 for the answer a user should see first,
  2 for a useful supporting result, and 1 for related background. File entries
  include approximate inclusive `lines` where a reviewer can inspect the gold
  passage quickly.

Construction uses repository files and narrow GitHub metadata/body inspection,
not Echo result rankings. It is a regression corpus, not a statistically random
sample: current operational and repository vocabulary is intentionally
overrepresented. GitHub and wiki records may age out of an Echo deployment, so
evaluations should report missing-corpus cases separately rather than scoring
them as ranking failures. A result is credited when its domain and stable target
match a judged artifact; line hints guide human review and sub-file evaluation.

Tune only on the `dev` split. Freeze the model, candidate budget, fusion weights,
and quality threshold before collecting the `test` split:

```bash
uv run infra/echo/search_benchmark.py collect \
  infra/echo/benchmarks/search_queries.jsonl /tmp/echo-search-dev.jsonl \
  --split dev --workers 4
uv run infra/echo/search_benchmark.py evaluate \
  infra/echo/benchmarks/search_queries.jsonl /tmp/echo-search-dev.jsonl \
  --split dev
```

The 2026-07-29 dev comparison used 59 queries. The previously deployed search
scored MRR@10 0.804, nDCG@10 0.804, hit rate 0.923, judgment recall 0.812, and
no-answer accuracy 0.000. Its mean, p50, and p95 latency were 2.35, 3.27, and
3.98 seconds. Bounded full-chunk reranking with the -2 quality floor scored MRR
0.858, nDCG 0.846, hit rate 0.923, judgment recall 0.841, and no-answer accuracy
0.714. Its mean, p50, and p95 latency were 2.14, 2.15, and 2.71 seconds. These
figures are a regression reference, not a service-level objective: corpus
freshness, Cloud SQL load, and model cold starts affect live latency.

A 2026-07-30 same-candidate ablation compared the float and 23 MB INT8 ONNX
MiniLM models over complete chunks. On dev, INT8 kept hit rates of 0.923 at
rank 5 and 0.942 at rank 10; MRR@10 changed from 0.874 to 0.862 and nDCG@10
from 0.858 to 0.849. Mean reranker time fell from 1.69 to 1.21 seconds. On the
held-out split, hit rates were unchanged at 0.692 at rank 1 and 0.846 at ranks
3, 5, and 10. MRR@10 changed from 0.744 to 0.756, nDCG@10 from 0.736 to 0.738,
and mean reranker time from 1.60 to 1.17 seconds. Judgment recall fell from
0.722 to 0.667 because one secondary artifact moved from rank 9 to rank 12.

An evidence-first 1,000-character input improved dev MRR@10 to 0.899, but was
rejected because held-out pass@5 fell to 0.769 with both float and INT8 weights.
Preserving complete chunks mattered more than the dev-only gain.
