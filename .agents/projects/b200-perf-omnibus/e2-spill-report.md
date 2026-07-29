# E2 same-step spill implementation report

## TL;DR

E2 re-offers fixed-capacity overflow assignments to the next-ranked expert that
the token selected. Each accepted spill uses the candidate expert's router
weight, the number of attempts is capped at `top_k - 1`, and the fixed
`num_experts * capacity` sender envelope does not change.

The functional diff is +174/−13 across two files, compared with the planned
+147/−12. The additions are 18% above the estimate and the deletions are 8%
above it. The report is excluded from that count.

## Implementation

`SCALE_A2A_SPILL=m` controls a static sequence of same-step placement attempts.
Round zero retains the existing stable first-assignment-wins placement. Each
later round offers only unplaced assignments to another expert from the same
token's top-k selection. Occupancy advances only for successful placements, so
no two accepted assignments share an `(expert, slot)` pair.

Experts and combine weights roll together during each attempt. This matters for
an overflow assignment displaced from expert 0 to expert 2: the expert 2 output
uses expert 2's router weight. The implementation uses static `roll` and
`where` operations instead of a dynamic combine-weight gather, which avoids
introducing a gather transpose into the combine-weight gradient.

Attempts are clamped to `top_k - 1`. The recorded `m=3`, capacity-factor
1.0625 operating point therefore fits both top-4 and top-8 routing, but only the
top-8 architecture can expose attempts 4 through 7. This ceiling is an
architecture-selection input, not a kernel tuning knob.

E2 contains no cross-step state or controller update.

## Verification

The allocation regression starts with five accepted assignments in a
three-expert, capacity-two layout. One spill fills the sixth existing slot.
Every accepted slot remains unique and below the original per-expert capacity,
and requesting 20 attempts produces the same result as the top-k cap of one.

The end-to-end fixed-all-to-all regression gives experts output scales 1, 10,
and 100 and assigns the spilling token weights 0.7 for expert 0 and 0.3 for
expert 2. The result is 60 because expert 2 executes twice at weight 0.3. The
source behavior would produce 100 by reusing displaced expert 0's weight.

Commands and results:

- `uv run --package marin-levanter --group test pytest
  lib/levanter/tests/grug/test_grugformer_moe.py`: 17 passed, 6 skipped.
- `uv run --package marin-levanter --group test pytest
  lib/levanter/tests/grug/`: 34 passed, 13 skipped.
- `uv run --package marin-levanter --group test pytest
  lib/levanter/tests/`: 1,134 passed, 201 skipped.
- `./infra/pre-commit.py --all-files --fix`: clean, including Pyrefly.
- `uv run --package marin-core --group lint pyrefly check`: 0 errors, 415
  baseline suppressions, 506 warnings not shown.
- `uv run pytest`: 1,252 passed, 17 skipped, 47 deselected, 5 xfailed, and one
  failure. The failure is
  `test_grug_base_run_emits_expected_metrics_with_json_tracker`; it reproduces
  alone in unchanged `experiments/grug/base/model.py` before MoE dispatch,
  where JAX rejects concatenating two differently sharded label slices.

Plain `uv run pytest` initially could not find pytest in the fresh virtual
environment. The Levanter test dependency group installed it, after which the
literal command ran. The installed Pyrefly CLI requires the `check` subcommand;
plain `uv run pyrefly` prints help rather than checking the project.

No GPU or cluster job was used. The recorded 20.708% MFU and 1.44% tail-100 drop
result was not remeasured, and this work does not validate rack-scale compile
cost or throughput.

## Extraction decisions

The source commit `1224ccb02` is based on D1 through D3 plus unrelated
batch-expert experiments. I did not cherry-pick it. I extracted the placement
mechanism onto D1 commit `67121ea3c` and did not carry batch-expert code,
research logs, launcher changes, or cross-step router controllers.

I also did not carry the source custom-adjoint parity test because it depends on
D2 and D3, which are owned by the sibling Phase D slice and must remain
orderable after E2. The binding E2 tests exercise the D1 fixed-capacity core.
After assembly, D3's gradient and HLO tests should also run with spill enabled.

## Plan discrepancies and remaining uncertainty

`origin/main` at `6ce4a7e68` has no fixed-capacity all-to-all core, so E2 cannot
be a functional commit directly on main. This branch includes the sibling D1
commit as an unchanged prerequisite. The assembler should cherry-pick the E2
commit after D1 and before D2 and D3.

The source implementation does not meet the binding combine-weight clause. It
changes the target expert but combines in the original assignment order, so a
spill inherits the displaced expert's weight. The extracted implementation
corrects that behavior and the end-to-end regression distinguishes the two
results.

The source also does not cap the environment value at `top_k - 1`; it repeatedly
wraps over the selected experts. The extracted implementation enforces the
documented architecture-dependent ceiling.

The default repository test selection is not fully green because of the
unchanged Grug base label-sharding failure described above. I did not modify
that out-of-scope training path.
