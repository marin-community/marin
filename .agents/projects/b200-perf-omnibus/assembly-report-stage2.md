# Stage 2 assembly report

Stage 2 adds the late Phase B item, the ordered D1–E2–D2–D4 throughput and
fidelity stack, and F1 on top of stage 1 at `f81a0b32f`. The implementation
series contains seven commits. No stage 1 commit was rewritten, and no branch
push, pull request, accelerator job, or cluster job was made.

## Commit order

| Order | Item | Source | Assembled commit |
|---:|---|---|---|
| 1 | B4 FA4 segment-bounds hoist | `a5017ce5f` | `00e9fbf97110da22bf8ead11ec37845e1afc0a79` |
| 2 | D1 fixed-capacity `lax.all_to_all` | `67121ea3c` | `2b81614afd5c1d39e34a1032728dbde789aa0e95` |
| 3 | E2 same-step spill | `2e5976720` | `58a16d9f5b9f8068496045b0c3c5d8d0fa620dd4` |
| 4 | D2 gather dispatch | `89b9ede2d` | `2158a300b061954609f1ae2dba15eb9c590297da` |
| 5 | D3 custom scatter-add adjoint | `38ab51f25` | `c7b16c0fae49c6622eab5f3a02a55242efbad23b` |
| 6 | D4 padded stack-sharded Muon | `63eb97c7d` | `b2fc033decbf9f1190b232d92bb938b66f975467` |
| 7 | F1 optimizer-state host offload | `662fc6db4` | `a58ebb8bb679537cfba001c2525e697a09f012f7` |

The B4, E2, and F1 source reports arrived in their implementation commits.
`d123-a2a-report.md` from `70d25fef3` was folded into D3, the commit that
completes the D1–D3 series, instead of landing as a trailing documentation
commit. `c12d4-muon-report.md` was already present from stage 1's C2 commit and
was byte-identical to the D4 source report, so D4 added only its code and tests.
This assembly report is the final stage 2 commit.

## Conflict resolutions

Four cherry-picks required manual resolution across seven files, grouped into
the five resolutions below. Every resolved file was read at the target symbol
after its conflict markers were removed.

1. D1 conflicted in
   `lib/levanter/tests/grug/test_grugformer_moe.py`. Stage 1's C3/C4 side added
   the ring dispatch-buffer size and expert-boundary sharding tests at the same
   insertion point where D1 added the fixed-all-to-all dense-reference test.
   The resolution keeps both test groups and D1's `_fixed_a2a_core` import. A
   formatter-required blank line was folded into the new D1 commit before any
   later stage 2 commit; stage 1 was not rewritten.
2. D2 conflicted in
   `lib/levanter/src/levanter/grug/_moe/ep_ragged_all_to_all.py` inside
   `_fixed_a2a_core`. E2 had replaced D1's `flat_experts` placement with
   `_assign_with_spill`; D2 added the gather-dispatch gate against the original
   placement. The resolution keeps E2's `target_experts`, `slot`,
   `routed_weights`, and `keep`, adds D2's `gather_dispatch` gate before them,
   and builds D2's send buffer from the spill-adjusted `linear_indices`.
3. D3 conflicted twice in the same production file. E2's spill helpers and
   D3's custom-VJP helpers occupied the same module-level insertion point, so
   both helper groups were retained. Inside `_fixed_a2a_core`, D3's
   `_gather_dispatch_enabled`, `_custom_adjoint_enabled`, and dependency check
   were combined with E2's `_assign_with_spill` placement. The dispatch and
   combine VJPs consume the resulting `linear_indices` and `keep`; the final
   combine still uses E2's spill-adjusted `routed_weights`.
4. D3 also conflicted in
   `lib/levanter/tests/grug/test_grugformer_moe.py`. E2's spill regressions and
   D3's value, gradient, and StableHLO regressions shared an insertion point.
   Both groups were retained. The resolved D3 tests still exercise
   `_fixed_a2a_core`.
5. F1 conflicted in three files:
   - `experiments/grug/moe/launch_cw_scale.py`: the Phase A side documented
     `SCALE_SCAN_LAYERS` and `SCALE_REPORT_DROPS`; F1 documented
     `SCALE_OFFLOAD_OPT_STATE`. All three controls remain.
   - `experiments/grug/moe/train.py`: A0 made `initial_state` handle scanned
     models where `params.blocks is None`; F1 initialized and optionally
     offloaded `opt_state`. The resolution preserves A0's layer count and then
     applies F1's optimizer-state initialization and pinned-host transfer.
   - `tests/test_grug_variant_contracts.py`: B4 required the `fa4_cute` import;
     F1 required NumPy and `set_mesh`. All imports and both test families
     remain.

B4 and E2 applied without conflicts. D4 also applied without a conflict: its
expected preimage was exactly stage 1's C2 blob
`3fa61d24f1dec7ee9cf0d605b176d1a4e6475083`. F1's non-conflicting hunks were
checked in `GrugTrainerConfig`, `_make_train_step`, `_run_grug_local`, the scale
launcher constructor, and the optimizer-state offload regression.

## Verification

Full repository checks ran at each stage 2 phase boundary. E2 was checked
directly in a detached temporary worktree after the final code series was
assembled.

| Prefix checked | Pre-commit | Pyrefly | Default pytest |
|---|---|---|---|
| Phase B through B4, `00e9fbf97` | clean | 0 errors | 1 failed, 1,262 passed, 18 skipped, 47 deselected, 5 xfailed |
| Phase E through E2, `58a16d9f5` | clean | 0 errors | 1 failed, 1,262 passed, 18 skipped, 47 deselected, 5 xfailed |
| Phase D through D4, `b2fc033de` | clean | 0 errors | 1 failed, 1,262 passed, 18 skipped, 47 deselected, 5 xfailed |
| Phase F through F1, `a58ebb8bb` | clean | 0 errors | 1 failed, 1,263 passed, 18 skipped, 47 deselected, 5 xfailed |

Every default run had the allowed pre-existing failure:
`tests/test_grug_variant_contracts.py::test_grug_base_run_emits_expected_metrics_with_json_tracker`.
The failure remains in untouched dense Grug code at
`experiments/grug/base/model.py:229`, where explicit CPU sharding gives the two
`concatenate` operands `P(("replica_dcn", "data"), None)` and `P(None, None)`.
The exception type and operand shardings match the stage 1 and bare-main
failure. No other default test failed.

The installed Pyrefly CLI requires a subcommand. Literal `uv run pyrefly` exits
with usage code 2; `uv run pyrefly check` reports zero errors, and the Pyrefly
lane inside `./infra/pre-commit.py --all-files --fix` passes. This command
discrepancy is also recorded in the D1–D3 and E2 source reports.

The boundary commits above received the full check directly. D1, D2, and D3 did
not each receive a separate full repository run:

- D1's one-shard fixed-all-to-all numerical test passed, and its resolved diff
  passed the changed-files pre-commit check.
- E2's candidate-weight and attempt-cap regressions passed, followed by the
  direct full E2 boundary check.
- D2's fixed and spill numerical tests passed with
  `SCALE_A2A_GATHER_DISPATCH=1`.
- D3's value, gradient, and backward-StableHLO tests passed both with spill
  disabled and with `SCALE_A2A_SPILL=1`: three tests in each configuration.
- D4's Muon and optimizer selection passed 12 tests before the full Phase D
  boundary check.
- F1's optimizer-state offload and one-step contract selection passed three
  tests before the full Phase F boundary check.

The final assembly-report commit is documentation-only. It receives the
all-files pre-commit check; its F1 parent received the full code gate above.

## SYRK and sharding state after D4

After D4, `lib/levanter/src/levanter/optim/grugmuon.py` has blob
`5aefa83f98b036e6d321239f95056955941de21c`, byte-identical to the audited D4
source. The 4D expert path remains reachable from `_grug_scale_with_muon` when
`x.ndim == 4`. With an expert mesh wider than one, it:

- checks `SCALE_MUON_SYRK`;
- constructs the local `local_syrk` function;
- reads `(local_layers, local_experts, local_d, local_last)` from the
  shard-local tile and flattens only `local_layers * local_experts`;
- calls `_newtonschulz_batched_syrk` inside `shard_map` with
  `distributed_4d_spec` as both input and output spec;
- reshapes locally and reshards the distributed result back to
  `orig_4d_spec`.

The nested-`vmap` alternative remains the reachable off branch. D4 changes the
separate 3D padded non-expert path by passing `_target_sharding(param)` and
resharding before slicing; it does not bypass or replace the 4D SYRK path.

The C3 sharding split is unchanged. `Plm_head_dense` remains in dense Grug,
`Plm_head_ep` remains in Grug MoE and June TPU MoE, and Snowball continues to
import the Grug MoE model. The overloaded `Plm_head` remains absent, and
`Pfsdp` remains `("data", "expert")`. The Snowball parity suite remains green.

These checks establish local control flow, sharding structure, CPU numerics,
and source identity. They do not establish a Blackwell SYRK compile or a
full-rack zero-warning result.

## E2 correction and evidence state

Both corrections missing from source `1224ccb02` are present:

- each attempt rolls selected experts and combine weights together, and an
  accepted spill updates `routed_weights` with the candidate expert's router
  weight;
- `attempts` is clamped with
  `min(max(attempts, 0), top_k - 1)`.

The allocation test verifies that a request for 20 attempts is equivalent to
the top-k cap, and the end-to-end regression distinguishes the correct result
of 60 from the buggy displaced-weight result of 100. The D3 gradient and HLO
tests also pass with spill enabled.

The recorded E2 fidelity result was measured on the buggy source path, not on
this corrected implementation. The corrected route has not been measured on a
rack. The reported 20.708% MFU at 1.44% drops therefore remains evidence about
the source experiment, not a qualification of this assembled tree.

## Plan differences and measurement limits

- `sequence.md` cites older main snapshots. Stage 2 uses the assigned stage 1
  tip `f81a0b32f`, whose base is `origin/main@6ce4a7e68`.
- The sequence attributes two test files to D1. The extracted D1 report shows
  that `experiments/grug/moe/test_model.py` is a C3-only sharding test and fails
  on a D1-only branch, so it was not carried again with D1.
- The current Pyrefly package contradicts the literal `uv run pyrefly` command
  in the governing requirement by requiring `pyrefly check`; both effective
  type-check paths are clean.
- Stage 1's report said `c12d4-muon-report.md` included deferred D4 sections.
  D4 is now present, and the report remained byte-identical. No stage 1 report
  claim was contradicted.
- The assigned source commits contained the mechanisms and reports described
  in the stage 2 instructions. No additional source-report contradiction was
  found.

The D-2 median of 22.398% MFU and 346,950 tok/s at 1.444% drops came from
`agent/deri-d2-build` artifact `c24ccfcc2`, not this branch. The concurrent
stage-1 EP64 functional gate cannot reproduce that composed configuration
because it lacks the stage 2 changes. Quoting the D-2 figure for shippable code
requires a rack rerun of the exact assembled series. That job remains in the
measurement queue and was not submitted here.
