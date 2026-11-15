# Verified SWE intake

`swe.py` declares the pinned Scale-SWE-Verified and SWE-Lego-Real-Data-Verified
populations. These sources remain in `PENDING_MEASUREMENT_SOURCES` until their
current release has all Atlas admission receipts. Select them by name for intake:

```bash
uv run python -m experiments.post_training.task_curation.driver \
  --source scale-swe-verified --source swe-lego-real-data-verified \
  --mode quick --output-root /private/new-swe-version --run
```

The checked-in license receipts record the dataset-level user decision and
named teacher licenses. QUICK conversion supplies no admission. Follow
[task curation](../../../../../docs/references/task-curation.md) for lexical
decontamination, full-population oracle and no-op trials, SAMPLE and final-pin
qualification.

Each row becomes a native Harbor package. Its Dockerfile uses the original
image with the source work directory; public setup runs before the actor.
The golden patch stays under `solution/`, and test material is supplied only
at verification. Source-declared 4 CPU, 4 GiB memory and 10 GiB disk limits
are retained. For missing phase deadlines, stage a versioned package with a
five-minute agent timeout and a one-minute verifier timeout, then run the
[full-population oracle filter](../../../../../docs/references/task-curation.md#full-population-oracle-filter).
Its calibration receipt records each attempt and the final per-task deadlines;
retain the source's explicit CPU, memory and disk limits.

Scoring preserves the source implementation pinned at
`PrimeIntellect-ai/research-environments@b41d8759070715dc102000e06fba0d73ebce9955`:

- `scale_score.py` retains the Scale-SWE JUnit scorer, with context-managed
  UTF-8 file reads.
- `native_scripts.py` contains the source setup, restoration and evaluation
  functions. The packaged `tests/native_scripts.py` postpones annotation
  evaluation because it executes with Python inside the original source image.
- `parse_pytest_outcomes` is from
  `PrimeIntellect-ai/verifiers@672d020c4dcabf5127a76f51aeb4adaf5b89d3fc`.
- The grade entry points adapt source runtime calls to the native verifier
  process and preserve its patch fallback order and test-result interpretation.

Both upstream projects license their code under Apache-2.0. Their license
notices are retained beside the adapted code.
