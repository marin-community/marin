# Model Scorecard

A static Marina app that shows selected models' scores on the eval-policy chat and agentic benchmarks
([#10197](https://github.com/marin-community/marin/issues/10197)) on a 0–100 scale. Each score links
to its run in EvalDash, or to the raw results in Filer when the result is not an eval run (for example
a regrade). Model names link to the Hugging Face revision that was evaluated, or to an unreleased
checkpoint in Filer.

[scorecard.yaml](scorecard.yaml) is the source of truth for the rows, the columns and the run behind
each cell. Each cell is labeled `official` (run under the eval-policy pin) or `unofficial` (run outside
the pin, from Marin main or a branch ahead of it). A cell marked `suspect` has a known grading or
protocol problem, so its score may not reflect the model; the page shows it struck through with a red
`!` and the reason on hover. An agentic cell can carry its Harbor trial counts (`trials`), shown under
the score as the share of trials scored and the share that hit the agent time limit; completion under
Marin's 90% bar shows in red. A model is a Hugging Face revision or, for an unreleased checkpoint, its
object-store location. [scorecard.json](scorecard.json) is generated from it and checked in, so
a release never reads object storage.

## Add a model or update a cell

1. Edit `scorecard.yaml`: add the model with its `hf_repo` and `revision` (or its checkpoint
   `location`), then give each eval a `run_id`, a `source` URI with a `score`, or a `missing` reason.
   Add `suspect: <reason>` to a scored cell with a known grading or protocol problem, and
   `trials: {attempted, scored, timeouts}` to an agentic cell.
2. Regenerate the data from the repository root. This reads each run's `record.json` and needs
   CoreWeave credentials:

   ```bash
   uv run python infra/marina/apps/model_scorecard/build_data.py
   ```

   The script uses the published EvalDash measurement, including the eval-policy metric selection.
   It rejects incomplete runs, runs whose eval does not match the column, and runs whose model or
   revision does not match the row.
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
