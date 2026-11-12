# Model Scorecard

A static Marina app that shows selected models' scores on the eval-policy chat benchmarks
([#10197](https://github.com/marin-community/marin/issues/10197)) on a 0–100 scale. Each score links
to its run in EvalDash, or to the raw results in Filer when the result is not an eval run (for example
a regrade). Model names link to the Hugging Face revision that was evaluated.

[scorecard.yaml](scorecard.yaml) is the source of truth for the rows, the columns and the run behind
each cell. Each cell is labeled `official` (run under the eval-policy pin) or `unofficial` (run from
Marin main outside the pin). [scorecard.json](scorecard.json) is generated from it and checked in, so
a release never reads object storage.

## Add a model or update a cell

1. Edit `scorecard.yaml`: add the model with its `hf_repo` and `revision`, then give each eval a
   `run_id`, a `source` URI with a `score`, or a `missing` reason.
2. Regenerate the data from the repository root. This reads each run's `record.json` and needs
   CoreWeave credentials:

   ```bash
   uv run python infra/marina/apps/model_scorecard/build_data.py
   ```

   The script rejects runs that did not succeed and runs whose eval does not match the column.
3. Run the tests and open a PR with both files:

   ```bash
   cd infra/marina && uv run pytest apps/model_scorecard
   ```

## Release

After the change merges, release from a clean checkout at current `main`:

```bash
uv run marina release-status infra/marina/apps/model_scorecard --json
uv run marina apps versions 58367197-a444-4cbd-b080-5ffd8d0b8034 --json
uv run marina release infra/marina/apps/model_scorecard --base-version CURRENT_VERSION --json
```

Preview a change locally with `uv run marina release infra/marina/apps/model_scorecard --local --json`.
