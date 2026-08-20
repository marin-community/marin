---
name: change-grug
description: Modify or upstream a Grug/Grugformer experiment variant.
---

# Change Grug

Grug is template-first. The canonical edit surface is
`experiments/grug/base/{model.py,train.py,launch.py}`; variants live under
`experiments/grug/<variant>/` and one-off speedruns are exploratory only. For
array-stacked wiring/perf experiments, use the reference branch
`https://github.com/marin-community/marin/tree/codex/array-stacked-grug-variant-pointer`.

Keep one change bucket per pass: attention/masking, block wiring/norm order,
MLP/activation, loss behavior, or optimizer/training loop.

1. Copy `experiments/grug/base` to a new variant and make explicit local edits.
   Avoid reusable framework abstractions until repeated use justifies them.
2. Update `docs/reports/grug-archive.md` with path, origin, commit SHA when
   known, purpose, status, and a visual diff link. For a local fallback:

   ```bash
   uv run python scripts/ci/grug_dir_diff.py \
     experiments/grug/base experiments/grug/<variant> --out /tmp/grug-diff
   ```
3. Run the smallest representative experiment and record evidence. If it wins,
   port the change into the base template and delete stale variant paths.

Validate every change with:

```bash
./infra/pre-commit.py --all-files
uv run pytest tests/test_grug_variant_contracts.py
```

Add focused tests for changed behavior.

Keep Grug style: plain JAX arrays, explicit sharding, Equinox `init` + `__call__`,
few config knobs, and readable local helpers. When HBM or compile time is the
bottleneck, read `docs/references/hbm-optimization.md` and evaluate the
array-stacked variant reference branch before bespoke hacks.

The archive entry is required for every variant; preserve CI visual-diff links
when available.
