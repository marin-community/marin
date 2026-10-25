# Tied-drafter pretraining on Belay

All tested strategies were negative for demonstrated cheaper progress to the same held-out loss.
The final four-expert draft used about 13% less execution time for the same update count,
but native training reached lower loss at lower cost after 12,288 updates than the draft
reached after 16,384 updates in all twelve paired comparisons.

[Complete report](https://storage.googleapis.com/marin-public/benfeuer/tied-drafter-belay/2026.10.24/index.html)
and [public reproduction index](https://gist.github.com/penfever/c2ff778a830e63c356e5efdf8cd590c1).
The index links four public Gists containing exact source snapshots, frozen protocols,
raw measurements, reports and checksums. Its downloader verifies all 653 files.

“Tied” means copied from full-model coordinates, trained independently, and mapped back;
the draft does not share mutable weight storage with a concurrently training hero.

The experiment uses ordinary backprop on a cheaper tied model. It includes both an early
cost-aware acceptance rule and later endpoint checks that require only a measurable loss
improvement. The latter accepts useful endpoints without proving that they match the
progress of full-model training or reduce cost to a common loss.

## Code and historical runtime

The adjacent experiment modules contain the readable drivers, acceptance logic,
coordinate restriction, transaction/state handling and architecture variants. Runtime-dependent
tests stay in the exact capsules and run in the restored pinned checkout; current main
has removed private APIs that they exercised. Pure acceptance and exchange tests remain
in the current checkout.
This directory contains analysis entrypoints and frozen protocols. Historical report
generators, diagnostic/collection programs and earlier variants are preserved byte-for-byte
in `reproduction/support-source.tar.gz` and the phase-specific source capsules. These
archives are part of the checked-in code, not generated model artifacts.

Measurements used Marin commit `8983bec6fdaeb1cd6d0acff041194d2bf53ab679` plus the recorded
runtime overlay. Shared model, MoE-combine and optimizer changes remain inside the source
capsules and `reproduction/final-runtime.patch`; this PR does not alter the current shared
training runtime. Readable copies have mechanical formatting changes. Use each phase's
exact capsule to reproduce its source hashes rather than running its driver against
current main.

Restore the final measured runtime in a new directory without launching jobs:

```bash
uv run --no-project --python 3.12 python experiments/grug/tied_drafter_record/prepare_runtime.py \
  --destination /tmp/tied-drafter-runtime
```

The script checks the archive and every restored member against the source manifest,
then restores supporting analysis and collection code. Earlier phases have their own
source manifests and capsules; the final overlay is not a substitute for those variants.
Original launchers preserve the historical machine paths. Substitute local checkout
paths and supply your own credentials before using them.

## Reproduce the analysis

Clone the public index Gist, run its downloader with Python 3.12+, then follow the commands
in its README. The public record can regenerate the 4,096/16,384-update equal-cost curves,
common-loss comparisons and direct measured endpoint comparison. It includes the original
analysis code and raw per-sequence loss/count measurements; no W&B account is required.

GPU reproduction additionally requires the referenced RNO2A data caches, tokenizer and
three full optimizer checkpoints, or regeneration using the recorded mixture, seeds and
warmup code. The original checkpoint bytes, corpus tokens and credentials are not public
artifacts. The protocols record their identifiers and hashes where available.

Cost comparisons use synchronized execution on matched eight-H100 allocations, adding
one initial copy and baseline/candidate check to draft execution. Research overhead is
reported separately. The experiments do not demonstrate production streaming throughput,
concurrent long-block hero/drafter training or a total-FLOP speedup.
