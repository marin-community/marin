# A3 XLA flag documentation report

## Built

`experiments/grug/moe/README.md` now records the per-job environment required
for multi-host GB200 MoE runs on JAX/JAXlib 0.11:

- the mandatory ragged all-to-all NCCL barrier disable;
- collective overlap limit 4 and the non-monotone limit result;
- the JAX CUDA async allocator and the ineffective TensorFlow allocator variable;
- the two unsupported fusion flags;
- the multi-host auto-PGLE failure and the narrower PGLE result;
- the JAX 0.11 baseline discontinuity.

The existing Grug MoE README absorbed the guidance. No standalone page or
MkDocs navigation entry was added. Every result links to an issue comment or an
immutable commit artifact, including the fresh d6144 schedule census.

## Diff size

`sequence.md` estimates A3 as "docs only" without a line count. The functional
diff is one existing documentation file at +62/−0. This report adds
79 lines of required task record and is excluded from the
functional count.

## Verification

- Read all of `sequence.md`, `evidence.md`, and the omnibus `README.md`.
- Checked each documented claim against the cited issue comment, source-branch
  artifact, or census report.
- Confirmed the documentation lives in the existing Grug MoE README and uses
  relative repository links plus absolute evidence links.
- `git diff --check`: passed.
- Shell syntax check for the copied environment block: passed.
- `uv run python infra/check_docs_source_links.py`: passed.
- `uv run mkdocs build --strict`: passed after setting `SSL_CERT_FILE` to the
  environment's certifi bundle. Without it, six external object inventories
  failed certificate verification.
- `./infra/pre-commit.py --all-files --fix`: passed, including the Pyrefly and
  Markdown checks. The first run had no `.venv` and reported missing imports;
  it passed after `uv sync` created the repository environment.
- `uv run pyrefly` exits with usage status 2 because the installed Pyrefly 1.0
  CLI requires a subcommand. The equivalent
  `uv run --with 'pyrefly>=1.0.0,<1.1.0' pyrefly check --baseline
  .pyrefly-baseline.json` passed with zero errors.
- Default `uv run pytest`: 1 failed, 1,252 passed, 17 skipped, 47 deselected,
  and 5 xfailed. The failure is
  `test_grug_base_run_emits_expected_metrics_with_json_tracker`, where unchanged
  `experiments/grug/base/model.py` raises `ShardingTypeError` while concatenating
  differently sharded label operands. The failing model, trainer, dispatcher,
  and test blobs are identical to `origin/main`; A3 changes no Python.

No GPU or cluster job was submitted. This slice does not independently reproduce
the cited runtime failures or measurements.

## Dropped

- No launcher or repository default was changed. These settings remain job
  environment because A3 is documentation-only.
- The older EP4 schedule census (`10, 3, 0, 1`) was not copied into the user
  guidance. The d6144 census supersedes it for the documented shape.
- No new documentation page was created because the Grug MoE README is the
  nearest discoverable home.

## Plan discrepancies and uncertainty

- `evidence.md` contains no `XLA_PYTHON_CLIENT_ALLOCATOR` or
  `TF_GPU_ALLOCATOR` entry, despite the assignment saying every required claim
  was recorded there. The documentation instead cites #7012 comment 4997024240
  and the allocator deadlock record at commit `8d48fb7fb`.
- The assignment dates the d6144 census 2026-07-29. The report and its immutable
  commit `3afd517d3` are dated 2026-07-28, so the documentation uses 2026-07-28.
- `sequence.md` was compiled against an older `origin/main`; this worktree and
  assignment use `6ce4a7e68`. A3 is additive documentation, so no extraction
  premise depends on that older base.
- The current Grug dispatcher forwards `XLA_FLAGS` and `JAX_` variables but not
  `XLA_PYTHON_CLIENT_ALLOCATOR`. The README warns that the allocator must reach
  the final accelerator process. Fixing propagation would be a behavior change
  outside A3.
