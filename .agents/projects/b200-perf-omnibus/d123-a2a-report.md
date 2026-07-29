# EP64 dispatch core extraction report

## TL;DR

D1, D2, and D3 are committed in order on `agent/impl-d123-a2a`. The series adds
fixed-capacity `lax.all_to_all`, constructs its send buffer with an int32 index
scatter plus activation gather, and replaces both gather transposes with
structured VJPs. At the CPU test shape, the backward-only StableHLO changed from
4 scatter operations to 0 while values and gradients matched at
`rtol = atol = 1e-5`.

No GPU, multi-device collective, or throughput measurement was run.

## Commits and diff size

| Item | Commit | Estimate | Actual |
|---|---|---:|---:|
| D1 | `67121ea3c` | +164 code, +117 tests | +164 code, +75/−2 tests |
| D2 | `89b9ede2d` | +17/−2 code | +19/−2 code |
| D3 | `38ab51f25` | +117 code, +117 tests | +105/−12 code, +213 tests |

D1's smaller test diff excludes `experiments/grug/moe/test_model.py`. Its 42
lines assert C3's non-expert `("data", "expert")` weight sharding and fail on a
D1-only branch; they do not exercise fixed all-to-all.

D2's two extra lines are Black's required formatting of the chained indexed
scatter. D3's implementation is smaller than the estimate because the
import-time research sentinel and redundant comments were removed. Its test
diff is larger because source `c9e30f848` had gradient parity but no HLO
assertion; the added backward-only lowering test is required by the sequence.
All three functional diffs remain within the requested factor-of-two bound.

The cumulative branch diff is +274 code and +288/−2 tests against
`origin/main` at `6ce4a7e68`.

## Behavior

D1 adds a `SCALE_A2A_FIXED=1` path that dispatches and combines static-capacity
expert buckets through `jax.lax.all_to_all`. `SCALE_A2A_CHUNKS` defaults to 1;
the measured two-chunk setting was worse.

D2 gates the index-scatter and activation-gather send-buffer construction behind
`SCALE_A2A_GATHER_DISPATCH=1`. The original repeated-bf16 scatter remains the
off path.

D3 gates the structured dispatch and combine VJPs behind
`SCALE_A2A_CUSTOM_ADJOINT=1` and requires gather dispatch. The dispatch
transpose gathers each kept slot's cotangent and sums over top-k assignments.
The injective combine transpose gathers through the slot-to-assignment inverse.

## Verification

- `uv run pytest lib/levanter/tests/grug/test_grugformer_moe.py`: 18 passed and
  6 existing accelerator-dependent cases skipped.
- Value, dropped-assignment count, input gradient, combine-weight gradient, and
  both expert-weight gradients match ordinary autodiff at
  `rtol = atol = 1e-5`. The test uses capacity factor 0.5 so it exercises
  overflow.
- Backward-only StableHLO at tokens 8, hidden size 6, intermediate size 8,
  4 experts, top-2, and capacity factor 0.5 contains 4 `stablehlo.scatter`
  operations with ordinary autodiff and 0 with the custom VJPs.
- The fixed path lowers on the abstract 2-way expert mesh, including
  `SCALE_A2A_CHUNKS=2`, and matches a dense reference on a one-shard CPU mesh.
- `./infra/pre-commit.py --all-files --fix`: clean.
- `uv run pyrefly check`: 0 errors. The literal command named in the plan,
  `uv run pyrefly`, prints usage with the installed Pyrefly because this version
  requires a subcommand.
- The repository-wide default marker selection ran with the root package's
  declared `math` extra and `pytest-timeout`: 1,252 passed, 17 skipped,
  5 xfailed, and 1 failed. The failure is
  `test_grug_base_run_emits_expected_metrics_with_json_tracker`, where untouched
  `experiments/grug/base/model.py` concatenates explicitly sharded and replicated
  token slices on CPU. Neither changed file is present in its stack.

The CPU environment cannot execute the real multi-device `all_to_all`, so only
abstract lowering and one-shard numerics were checked. No B200 performance
claim was remeasured.

## Dropped scope

- From `fe21ea495`: C2 Muon changes, C3 model and sharding changes, the CUTLASS
  revert, dispatch environment forwarding, and the C3-only
  `experiments/grug/moe/test_model.py`.
- From `c9e30f848`: the import-time log sentinel used to identify a research
  bundle.
- E2 same-step spill and all receiver-ECHO, FP8, optimizer, launcher, and
  unrelated dispatch work.

## Plan errors and uncertainty

The planning documents cite older `origin/main` snapshots than this worktree's
`6ce4a7e68` base. The D1 source hunks still apply, but the statement that both
test files belong to D1 is wrong: one is exclusively a C3 sharding test. The D3
source test also lacked the binding backward-HLO assertion.

The custom VJPs were verified at one small CPU shape. The test proves the
scatter-removal mechanism is engaged at that shape, but it does not establish
the recorded 544-to-0 count at the production shape or validate GPU code
generation.
