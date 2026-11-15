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

## Multi-SWE private image inputs

Multi-SWE stages the original image's top-level `/home/*.sh` and `test.patch`
on the trusted host before removing them from the actor. Its converter needs a
frozen private archive from each original image. Resolve every pinned source
row's image tag to a digest before capture. The rows are
[`PrimeIntellect/Multi-SWE-RL-Verified@64b3ebf62750b1373e26106cfc658a1861809d0f`](https://huggingface.co/datasets/PrimeIntellect/Multi-SWE-RL-Verified/tree/64b3ebf62750b1373e26106cfc658a1861809d0f),
in `data/train-00000-of-00003.parquet` through `data/train-00002-of-00003.parquet`.
Retain those files' SHA-256 hashes under `raw_files` in the
[`CaptureInput` schema](capture_artifacts.py). Each `ImageBinding` contains the
original references, their pinned digest reference and source `instance_id`
values. Retain registry descriptors beside the index. The stage requires all
2,232 source tasks and one record per distinct image digest.

First generate an exact source-ID/pin receipt with the documented
[stage-0 gate](../../../../../docs/references/task-curation.md#stage-0-binding-license-gate).
Supply its JSON registry with `--license-receipts RECEIPTS.json`, or use the
canonical checked-in registry. `PRIVATE_REGIONAL_ROOT` is a private `s3://` prefix
in the acquisition region; for CW RNO2A, use a new prefix under
`s3://marin-rno2a/`. Preserve its artifact version when resuming identical inputs.

```bash
uv run python -m experiments.post_training.task_curation.datasets.prime_intellect.capture_artifacts \
  --images FROZEN_IMAGE_INDEX.json --images-sha256 INDEX_SHA256 \
  --cluster cw-rno2a --concurrency 4 --artifact-root PRIVATE_REGIONAL_ROOT \
  --artifact-version YYYY.MM.DD.N --output capture-result.json --run
```

The stage enforces the exact-pin license gate before creating its graph and
again before capture. It uses secret-free SANDBOX machines with network denied,
one CPU, 2 GiB memory and a requested 10 GiB job disk budget. This auxiliary
metadata operation supplies no source execution or storage-enforcement
qualification; original task allocations remain in the task contract.
Each archive transfer must match the source-machine SHA-256, and its receipt
retains member bytes, modes and mtimes. Completed captures resume only when
their image/task binding and stored hashes still match. Failed attempts and
partial archives stay preserved; a failure prevents a complete-source result.
`capture-result.json` references the complete `captures.json` manifest and its
hash. Preserve both with the intake evidence. The Multi-SWE TaskSpec converter
and its private Hugging Face auxiliary-input binding remain pending.

Docker Hub resolution must be single-threaded and honor `Retry-After`, with
exponential backoff capped at ten minutes. Anonymous registry `HEAD` version
checks avoid image downloads; retain response digests and rate-limit evidence.
Do not replace unavailable images with another environment.
The documented endpoint is
`https://registry-1.docker.io/v2/REPOSITORY/manifests/TAG`; use Docker's normal
anonymous registry negotiation or an already authorized Docker CLI sign-in.
Keep safe response headers and retry timestamps beside the index, excluding
authorization tokens. See [Docker's version-check procedure](https://docs.docker.com/docker-hub/usage/pulls/#view-pull-rate-and-limit).
