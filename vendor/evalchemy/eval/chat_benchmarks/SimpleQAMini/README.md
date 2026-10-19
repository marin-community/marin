# SimpleQAMini

`SimpleQAMini` is a separately registered 500-question subset of `SimpleQA`.
It applies the OpenAI reference evaluator's
`random.Random(0).sample(full_dataset, 500)` selection to the pinned 4,326-row
dataset shipped under `SimpleQA/data/`. Generation, grading, metrics, and judge
credentials match the full benchmark.

See [`../SimpleQA/README.md`](../SimpleQA/README.md) for provenance and usage.
