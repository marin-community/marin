# 520M Data Point Recovery Instructions

Reference: `.agents/logbooks/520m-1.2b-babysit-20260421.md`

## Context

All 12 qsplit-520M-0.5x runs completed training to step 9917 and wrote
perplexity eval (uncheatable + paloma) to `eval_metrics.jsonl`. However:
- 0/12 have lm-eval harness results
- 0/12 have SUCCESS executor status
- Only 3/12 still have checkpoints at step 9917

The perplexity evals are the primary objective metric for the scaling study.
The lm-eval harness (piqa, arc, hellaswag, sciq, MMLU) is secondary.

## Recoverable Data Points

### Tier 1: Have checkpoint at 9917 — can complete lm-eval

These 3 runs have both perplexity eval AND a surviving checkpoint at step 9917.
A new child can load from the checkpoint, skip training (already at target),
and run the lm-eval harness to get SUCCESS.

GCS base: `gs://marin-us-east5/checkpoints/pinlin_calvin_xu/data_mixture/ngd3dm2_qsplit240_representative12_520m_5p2b_target0p5x/`

1. `baseline_unimax-3155cd` — checkpoint at step 9917, eval at 9917
2. `run_00090-5f5169` — checkpoint at step 9917, eval at 9917
3. `run_00213-58954c` — checkpoint at step 9917, eval at 9917

**Recovery**: The existing parent job (`relaunch_strong_tier_cell.py` with
`--resume-latest-checkpoints`) should dispatch children that load from step
9917, find training complete, run eval hooks (including lm-eval harness),
and write SUCCESS. The checkpoint-wait fix in the workspace bundle ensures
the checkpoint save completes before the process exits.

The bottleneck is surviving the ~2h lm-eval window without preemption.

### Tier 2: Have perplexity eval but lost checkpoint — consider done for analysis

These 9 runs have valid perplexity eval at step 9917 in their
`checkpoints/eval_metrics.jsonl` file, but lost their step-9917 checkpoint
to Levanter's checkpoint rotation bug.

1. `baseline_proportional-8057c8`
2. `run_00018-830517`
3. `run_00021-f23fc9`
4. `run_00050-c1a3ed`
5. `run_00056-7cd4a7`
6. `run_00125-7aff4f`
7. `run_00152-2e1de5`
8. `run_00155-40d4cb`
9. `run_00180-be99ab`

**For analysis purposes**: The `eval/uncheatable_eval/bpb` values in
`eval_metrics.jsonl` at step 9917 are valid and can be used directly.
The perplexity eval ran successfully at the target step — the checkpoint
loss happened later when a new child rotated it out.

**To extract these values**:
```python
import fsspec, json
fs = fsspec.filesystem('gcs')
prefix = 'marin-us-east5/checkpoints/pinlin_calvin_xu/data_mixture/ngd3dm2_qsplit240_representative12_520m_5p2b_target0p5x'
for run_name in ['baseline_proportional-8057c8', 'run_00018-830517', ...]:
    content = fs.cat_file(f'{prefix}/{run_name}/checkpoints/eval_metrics.jsonl').decode()
    for line in content.strip().split('\n'):
        entry = json.loads(line)
        if entry.get('step') == 9917:
            print(f"{run_name}: bpb={entry['eval/uncheatable_eval/bpb']}")
            break
```

**To get lm-eval harness results**: These runs need to retrain to step 9917,
save a checkpoint, and then survive the eval harness window. The babysit
loop will eventually get them there. With the checkpoint-wait fix, the
checkpoint will be properly saved, and checkpoint rotation won't delete it
(the force-save at the final step marks it as permanent).

### Tier 3: Unrecoverable

- `baseline_stratified 520M 0.5x` (`baseline_stratified-a2aad9`):
  Overshot to step 19835. No checkpoint at 9917, no eval at 9917.
  Eval schedule was every 1000 steps, so step 9917 was never evaluated.
  Cannot recover a valid 0.5x data point from this run.

## Registry Update

The run registry builder (`build_run_registry.py`) should be patched to:
1. Accept Tier 2 runs as perplexity-ready even without a checkpoint, since
   `eval_metrics.jsonl` contains the valid eval
2. Require the eval record to be at the `target_final_checkpoint_step`,
   not just at any step past it (to catch overshoot cases like the stratified bug)

## Current Source Of Truth

The canonical provenance layer is:

- `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/run_registry/build_run_registry.py`
- `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/run_registry/logical_runs.csv`
- `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/run_registry/run_attempts.csv`
- `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/run_registry/strong_tier_perplexity_ready.csv`
- `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/run_registry/summary.json`

Important:

- `run_registry/` is the canonical local provenance layer.
- `build_run_registry.py` is the only supported writer for that layer.
- packet-local copies of `strong_tier_perplexity_ready.csv`, packet-local NPZ
  files, W&B mirrors, and ad hoc benchmark artifacts are **derived** and must
  never be treated as the source of truth.
- The real authoritative metric source is the exact-step record in
  `checkpoint_root/checkpoints/eval_metrics.jsonl` at
  `target_final_checkpoint_step`.

Operationally, the source-of-truth hierarchy is:

1. Exact-step `eval_metrics.jsonl` record on GCS for the attempt
2. Attempt metadata discovered by the registry builder
3. Generated `run_registry/*.csv` / `summary.json`
4. Derived packet-local snapshots

## Safe CC Registry Refresh Contract

Yes, I would trust CC to update the registry **if it follows this contract**:

1. Do not hand-edit `run_registry/*.csv` or `summary.json`.
2. Treat `build_run_registry.py` as the only writer.
3. Use the exact-step eval rule only:
   - a run is perplexity-ready iff `eval_metrics.jsonl` contains the objective
     metric at `target_final_checkpoint_step`
   - checkpoint existence is optional
   - evals that appear only at steps *past* the target step do **not** count
4. Refresh the registry before refreshing any packet-local copies.
5. After refresh, spot-check the `520M` rows that changed.

## Safe Refresh Command

Use the deterministic refresh first:

```bash
uv run --with torch python \
  experiments/domain_phase_mix/exploratory/two_phase_many/run_registry/build_run_registry.py \
  --no-include-live-status
```

Only use live status when needed for operational babysitting state, not for the
canonical analysis refresh:

```bash
uv run --with torch python \
  experiments/domain_phase_mix/exploratory/two_phase_many/run_registry/build_run_registry.py
```

## Required Post-Refresh Checks

After every refresh, CC should verify all of:

1. `summary.json["strong_tier_perplexity_ready_rows"]`
2. `strong_tier_perplexity_ready.csv` fixed-`520M` row count
3. No stratified overshoot row is counted as `is_perplexity_ready`
4. Newly added `520M` rows have monotone same-run losses across available
   smaller scales (`130M -> 300M -> 520M`)

For the current canonical basis, the fixed-`520M` analysis slice should be:

- qsplit-only
- `16` rows
- no stratified row

If a refresh changes that, CC should assume the basis changed materially and
log the exact delta before running any downstream benchmark refresh.

## When CC Should Update The Registry

CC should update the registry when:

- new strong-tier runs finish target-step perplexity evals
- a babysat recovery run writes a new exact-step eval record
- a bug fix changes readiness logic in the builder

CC should **not** update packet-local benchmark snapshots first and then try to
reconcile the registry after the fact. The registry must move first.

## Ongoing Babysit

The babysit loop continues to:
- Resubmit dead parents every 10 minutes
- All parents include the checkpoint-wait fix in their workspace bundle
- Children from new parents properly wait for checkpoint save before eval
- The qsplit 2.0x runs are at 50-76% and approaching target

PR for the checkpoint fix: marin-community/marin#5022
