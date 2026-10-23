# Terminal-Bench verifier review

The `qemu-alpine-ssh` task in the campaign's Terminal-Bench 2.1 dataset can fail verifier package installation, leave `curl` and `uvx` unavailable, and report a zero reward without running tests. `review_terminal_bench.py` requires a terminal result record for each of the task's three agent trials, including infrastructure failures, and checks them separately. It excludes an originally scored reward only when the result contains this package-installation failure and no pytest execution summary. Outcomes already excluded by the native infrastructure taxonomy are not subtracted again. Genuine test failures remain scored. Model generations and benchmark configuration are unchanged.

Each model attempted 89 tasks with three trials per task, totaling 267 trials. A score remains sealable, meaning eligible for a completed tracker cell, only with at least 241 valid outcomes (90%). A failed verifier cannot establish the score of an agent's retained files and commands, even if that agent previously timed out; the release requires running the verifier rather than automatically assigning Harbor timeouts zero.

Run from the Marin repository root with `CW_KEY_ID` and `CW_KEY_SECRET` set for CoreWeave S3. Credentials are required for reading the source results and publishing reviews. This example uses the current local campaign:

```bash
CAMPAIGN=experiments/evaluation/campaigns/eval-policy
CAMPAIGN_ROOT=/Users/benfeuer/Documents/experiments/active/eval-policy-09-25
TRACKER="$CAMPAIGN_ROOT/TRACKER.md"
RECOVERED_METRICS="$CAMPAIGN_ROOT/artifacts/recovered-record-metrics.json"
REVIEW_MANIFEST="$CAMPAIGN_ROOT/artifacts/tb2-verifier-review.json"
uv run python "$CAMPAIGN/analysis/harbor_verifier_audit/review_terminal_bench.py" \
  --tracker "$TRACKER" --recovered-metrics "$RECOVERED_METRICS" \
  --output "$REVIEW_MANIFEST"
```

`TRACKER` is Markdown; the other two paths are JSON. The output manifest records per-model original and corrected means, SEMs, valid trial counts, excluded trial IDs, source object SHA-256 hashes, and the review code commit. Add `--publish-prefix` with an unused S3 prefix to save immutable per-model copies and record their URIs in the manifest.

Scoring categories come from the Harbor dependency pinned by Marin. Statistics come from each source archive's `harbor_result.json`; if it is missing, Marin's aggregate helpers recompute them from individual `result.json` trial files. This avoids combining a stale operator trial count with a newer mean.

The tool does not change the tracker or recovered-metrics registry. Promotion updates those files only after reviewing the manifest. It must hold an exclusive `fcntl.flock` on `$CAMPAIGN_ROOT/artifacts/.auto-refill.lock` to serialize writes with the refill supervisor. A below-threshold result must be restored through the queue rather than promoted as complete.
