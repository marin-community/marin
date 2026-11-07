# marin-mariner

MARINER's mixture-selection package. The repository's [root README](../../README.md) covers installation, the
`mariner` commands, and the reproductions of the paper's mixtures, figures and main table.

## Tests

From the repository root:

```bash
uv run --package marin-mariner --extra olmix --extra figures --group test pytest lib/mariner/tests
uv run --package marin-mariner --extra olmix --extra figures --group test pytest lib/mariner/tests \
  -m data_integration -n auto --mariner-data DIR --session-timeout=3600
```

The first command runs the tests that need no dataset. The second runs the `data_integration` tests against a
mariner-data snapshot in `DIR` (`mariner download --out DIR`). They refit the three swarms and the Olmix laws and
compare them with the released tables, and they draw each of the paper's 27 figures from only the dataset
configurations that the snapshot's `figures_tables.csv` lists for it, checking each against a number or claim that
the paper states. They take about 10 minutes, most of it the OlmoBaseEval Easy Olmix refit.

The committed PDFs in [`paper_figures/`](paper_figures/) were drawn on macOS arm64 with the lockfile's versions.
PDF bytes depend on the platform and the BLAS build, so the tests do not compare them with the committed files.
