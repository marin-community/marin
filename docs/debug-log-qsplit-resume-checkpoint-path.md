# Debug Log: QSplit Resume Checkpoint Path

## Context

The fixed stratified 520M relaunch:

```text
/calvinxu/dm-stratified-520m-10p4b-20260414-224338
```

got past the JAX distributed initialization bug, then failed while resuming:

```text
ValueError: initialize_from must be a checkpoint path, got mirror://checkpoints/pinlin_calvin_xu/data_mixture/ngd3dm2_stratified_520m_10p4b/baseline_stratified-263bc9
```

Levanter also logged that the path existed but did not contain `metadata.json` or direct checkpoint subdirectories.

## Root Cause

The qsplit resume resolver returned the run root:

```text
.../baseline_stratified-263bc9
```

but `TrainerConfig.initialize_from` requires a concrete Levanter checkpoint directory:

```text
.../baseline_stratified-263bc9/checkpoints/step-<N>
```

The run root is still useful for eval launchers that discover HF exports, so changing the existing root resolver would break that path.

## Fix

Added `resolve_latest_checkpoint_path(...)` for training resume paths. QSplit replay and stratified scaling launchers now use the concrete checkpoint path when `resume_latest_checkpoints` is enabled.

## Follow-up: Temporary Checkpoint Metadata

The next stratified 520M relaunch selected:

```text
mirror://checkpoints/pinlin_calvin_xu/data_mixture/ngd3dm2_stratified_520m_10p4b/baseline_stratified-263bc9/checkpoints/step-9756
```

That directory had `metadata.json`, but its metadata said `"is_temporary": true`. The east5 copy only had
the metadata file, and the central1 copy still failed restore with missing OCDBT arrays. The resolver was
using metadata presence as proof that the checkpoint was resumable.

Fix: both latest-checkpoint resolvers now skip metadata with `is_temporary=true`. If no committed
checkpoint exists, resume resolution returns `None` and the launcher starts from scratch instead of
trying to restore an incomplete checkpoint.
