# StarCoder WSD80 gradient plot completion v5 provenance diff audit

This supplement closes the source-diff contingency from the v5 CC review.

## Commit relation

- Recorded clean commit: `7efb96842624a2e8cbab36c9a9aa6b1cb68c4922`.
- The v10 jobs recorded `code_dirty=true` at that commit.
- Later commit containing the relevant dirty root/library state: `377ad16d816a1726cc97396355607594910e9f0a`, the immediate child of `7efb`.
- There are no `pyproject.toml`, `uv.lock`, or `lib/**` changes from `377ad` through recorded experiment commit `1fe1358`.

## Root/library changes in `7efb..377ad`

The commit changes Iris/Marin retry and workspace-upload plumbing, checkpoint-save correctness, deterministic finite-support/holdout dataset views, and continuation configuration. It does not change model forward computation, loss computation, gradient accumulation, gradient transforms, optimizer implementation, Haliax reductions, or trainer-state reduction.

The only optimizer-adjacent source change is in `levanter/main/train_lm.py`:

- `TrainLmConfig.optimizer_schedule_num_train_steps` can build the existing optimizer schedule against the full training horizon while a resumed continuation has a shorter `trainer.num_train_steps`.
- The default remains `trainer.num_train_steps`.
- No optimizer transform or update equation changes.

Other Levanter changes are:

- `checkpoint.py`: avoid duplicate forced-final writes at a step already saved permanently.
- `data/dataset.py` and `data/text/datasets.py`: deterministic random holdouts, deterministic support-window selection, vectorized but order-restoring dataset reads, and per-dataset shuffle caches.
- `train_lm.py`: optional create-only initial-state evidence after restore.

No Haliax files and no files under `levanter/optim`, `trainer.py`, or `trainer_state.py` changed.

## Consequence for recovery

`377ad` is required to reconstruct the frozen v10 configurations: the clean `7efb` API rejects the pinned `artifact_cache` argument, while `377ad` reconstructs all 256 configurations and matches all 5,888 frozen fields.

Source inspection cannot prove that the later committed tree is byte-for-byte identical to the uncommitted tree used by the historical jobs. Therefore the v5 launch remains fail-closed on Stage 1 numerical reproduction:

- approximately 9,000 parent-vs-recovery scalar comparisons at tolerance `5e-6 * max(|observed|, |expected|, 1)`;
- 88 final source-gradient norm comparisons;
- exact Python, JAX, JAXLIB, libtpu, and installed-distribution inventory checks;
- no Stage 2-4 launch unless Stage 1 passes.

V3 already demonstrated that this gate detects a plausible but numerically wrong runtime: 2,471 of 9,000 scalar comparisons failed even though losses matched exactly. V4 was never launched because its historical worktree failed configuration reconstruction before TPU spend.
