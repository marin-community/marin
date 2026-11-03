# NUPA5K-Loose

`NUPA5K-Loose` is a separately registered, fixed 5,000-record convenience panel from
the canonical NUPA test source. It uses the same prompts, generation settings,
scorer, and metrics as `NUPA-Loose`.

The checked-in identity manifest covers all 44 tasks and all 2,391 nonempty
`(task_name, digit)` strata. It allocates records by round-robin over strata
sorted by task name and numeric digit length, skipping an exhausted stratum on
later passes. Within each stratum, unique source texts are ordered by SHA-256.
The manifest pins membership and order independently of source JSON ordering.

NUPA5K-Loose reports an unweighted mean over its panel. It is not an unbiased
estimator of the 238,926-record NUPA publication protocol. Always identify the
variant when reporting a result. Its permissive answer extraction also means
scores are not directly comparable to the paper's direct-answer results.

Run the fixed panel against an OpenAI-compatible endpoint with:

```bash
uv sync --extra nupa5k-loose

eval --model local-completions \
  --tasks NUPA5K-Loose \
  --model_args model=served,base_url=http://localhost:8000/v1/completions
```

Rebuild and verify the manifest from the pinned source with:

```bash
uv run --extra nupa5k-loose python -m eval.chat_benchmarks.NUPA5K-Loose.data_prep.build_manifest
```

See [`../NUPA-Loose/README.md`](../NUPA-Loose/README.md) for source provenance, scoring, and
the full benchmark protocol.
