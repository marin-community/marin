# Debugging log for codehealth review aggregate

Fix `infra/codehealth/review.py aggregate --days 30` so the default PR scan
does not stop at 100 merged pull requests.

## Initial status

The command logged `Found 100 merged PRs` for a 30-day window even though the
repository had more merged PRs in that period.

## Hypothesis 1

`aggregate` passed its default `--limit 100` directly into `gh pr list`, so the
listing stopped at 100 before comment aggregation began.

## Changes to make

Replace the capped `gh pr list` call with a REST pulls scan that pages closed
PRs by `updated_at`, filters by `merged_at`, and stops only once `updated_at`
falls before the requested window. Keep `--limit` as an explicit optional cap.

## Results

Focused validation with `list_merged_prs("marin-community/marin", 30, None)`
returned 335 PRs, confirming the aggregate default is no longer capped at 100.
