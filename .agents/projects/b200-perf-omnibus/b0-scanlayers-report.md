# B0 scan-layers port

## Result

Ported `97b53fe0e` onto `origin/main` at `6ce4a7e6874e10a6013949000dbb0f7e0a92bcf2`.
`SCALE_SCAN_LAYERS=1` now reaches `GrugModelConfig.use_array_stacked_blocks` from
`build_scale_model()`. The model stores the layer parameters in
`ArrayStacked[Block]` and executes one `jax.lax.scan`; the default remains the
unrolled tuple of blocks.

Scan mode rejects `disable_pko=False`. PKO chooses behavior from a Python
per-layer flag at trace time, which cannot vary inside the homogeneous scan
body. No debug side effects or other instrumentation were added to the
rematerialized body.

The source commit was not self-contained against current `main`.
`Transformer.blocks` becomes `None` in scan mode, but
`experiments/grug/moe/train.py` iterated it while initializing QB state and while
applying pending QB betas. The source branch fixed this later in `a796ff99a`.
This port includes that 10-line behavior so the launcher configuration can
lower a training step. It also unpacks `stacked_blocks` for inference state-dict
export; without that update, `Transformer.to_state_dict()` failed in scan mode.

## Diff size

The source estimate was `+92/-22` across the launcher and model. The functional
port is `+101/-24` across the launcher, model, and training loop: 1.10 times the
estimated additions and 1.09 times the estimated deletions. The regression
coverage is another `+109` lines. Before this report, the total diff was
`+210/-24`.

The extra functional lines are the QB training fix from `a796ff99a` and the
state-dict call-site update. No unrelated source hunk, vendored file, smoke
script, or instrumentation was carried.

## Verification

- Scanned and unscanned models initialized from the same key match at
  `rtol=atol=1e-5` for hidden values, every per-layer router metric, and every
  exported state-dict tensor.
- The `SCALE_SCAN_LAYERS=1` launcher path lowers to one JAXPR `scan` with
  `length=5` and `unroll=1`. StableHLO contains one `stablehlo.while`.
- A full MoE training step lowers with stacked blocks, exercising QB state
  initialization and the pending-beta update path.
- `uv run pytest` on the three B0 tests passed: 3 passed.
- `tests/test_grug_variant_contracts.py` reached 14 passed and 1 GPU skip. Its
  remaining test fails in the untouched `experiments/grug/base/model.py` label
  concatenation because JAX sees `P(("replica_dcn", "data"), None)` and
  `P(None, None)` operands. The failure reproduces when run alone.
- The full default `uv run pytest` selection completed with 1,255 passed,
  17 skipped, 5 xfailed, 47 deselected, and the same one pre-existing
  Grug-base failure. The failing base model and training files are unchanged
  from `origin/main`.
- `uv run pyrefly check` passed with 0 errors. The documented
  `uv run pyrefly` command is stale for Pyrefly 1.0 and exits with usage because
  the CLI now requires the `check` subcommand.
- `./infra/pre-commit.py --all-files --fix` passed after applying Black's
  formatting change.

No GPU or cluster job was run. The CPU tests verify numerical parity and program
structure, but they do not reproduce the production-shape compile-time or HBM
failure.

## B4 applicability

The complete `a33e16ced` patch now passes:

```text
git show --format= --binary a33e16ced | git apply --check --verbose -
```

All six `experiments/grug/moe/model.py` hunks apply, as do
`attention/__init__.py`, `attention/_core.py`, and `attention/_fa4_cute.py`.
Some model and `_core.py` hunks apply with line offsets, but none require a
conflict resolution. This was an apply check only; B4 was not added to this
branch.

## Plan discrepancies and uncertainty

- `sequence.md` omits B0, as the assignment states. Its header also targets an
  older `origin/main` than this worktree.
- `97b53fe0e` alone does not produce a runnable scan-mode training path because
  it predates `a796ff99a`.
- The assignment says the scan OOM and PKO interaction is already recorded in
  the config docstring on `main`. At `6ce4a7e68`, `main` documents PKO but has no
  scan field or scan OOM text. This port adds the combined constraint to the new
  field's docstring.
- The default suite is not fully green because of the isolated Grug-base
  explicit-sharding failure described above. That failure is outside B0 and was
  not changed here.
