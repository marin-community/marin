# Stage 1 assembly report

Stage 1 rebases the existing 14 documentation commits onto
`origin/main@6ce4a7e68` and adds the 15 requested implementation commits through
A2. The documentation rebase was conflict-free. The assembled branch stops at
A2; no stage 2 change, cluster job, push, or pull request is included.

## Commit order

The rebased documentation prefix runs from `343bcdeecc8f90b62f9f0c42705bbc4560c88ca3`
through `970df8b141b77cd27c173e09b5d7a9dd6fcb6a91`. The implementation commits
follow in this order:

| Order | Item | New commit |
|---:|---|---|
| 1 | A0 scan-layers | `c9adea6713f9907fe32dcc95cf02a130d97566a2` |
| 2 | A1 drop-report gate | `54af976f481d2ed1a87c43e020771c19d5c03b0c` |
| 3 | Child job priority | `5ba314b0f4f3c9fed2822d659c4941ef994ab8ee` |
| 4 | A3 flag documentation | `1e2f441f8ac6f282a0fe5ff00922aee29be59c38` |
| 5 | A4 dispatcher environment prefix | `b4302d3aa49cd7fcb0ebfc6ec367baea1191d777` |
| 6 | B1 QuACK dependency | `d8d08ee034ab32ab42fbcbb99c1b4c7c0d953767` |
| 7 | B2 `sonic_cute` backend | `ee41089225473e8e3ae1751371b72bcbfb5b0afa` |
| 8 | B2 chunk-drop fix | `a7ee6b171540567b3c267d48906880ac9f1c3522` |
| 9 | FA4 CUTLASS 4.6 migration | `c2927901c341b110ab9f2249be8bf13fd8e39267` |
| 10 | B3 replica-local embedding gather | `6802bbffa08642ed6df92a0477ca2c67312670e0` |
| 11 | C1 4D stacks through Newton–Schulz | `6c8ef26fc3c7b69402c26f98b7d7f8387bc26888` |
| 12 | C2 preserve expert sharding | `bd636024c0e0531c2bb8c75575eec9d206f588de` |
| 13 | C3 non-expert FSDP over `(data, expert)` | `aea5c58cf2b6de49d2a6550856a9aa8644f2d763` |
| 14 | C4 expert-axis batch spec | `f8b4f7aa337e63d77b0d4894cf079d41224b94c7` |
| 15 | A2 capacity-factor default | `bc8ca819b1fddce71683c17291457823cae9d05b` |

## Diff size

These counts exclude tests and the required reports. The item reports contain
the file-level accounting and explain the differences from the estimates.

| Item | Plan estimate | Actual functional diff |
|---|---:|---:|
| A0 | +92 / -22 | +101 / -24 |
| A1 gate | shared approximately +30 with the chunk fix | +11 / -2 |
| Child priority | not estimated | +18 / -0 |
| A3 | documentation only | +62 / -0 |
| A4 | approximately one line | +1 / -1 |
| B1 | approximately +10 | +7, plus generated lockfile +10 / -3 |
| B2 backend | approximately +560 | +584 / -0 |
| B2 chunk fix | shared approximately +30 with A1 | +12 / -2 |
| FA4 CUTLASS migration | +5 / -5 | +5 / -5 |
| B3 | +32 / -6 | +31 / -3 |
| C1 | +222 / -20 | +155 / -18 |
| C2 | +54 / -11 | +103 / -13 |
| C3 | approximately +23 / -8 | +27 / -17 |
| C4 | +11 / -0 | +11 / -0 |
| A2 | approximately +8 | +14 / -7 |

C2 is the largest ratio to its estimate at 1.91 times the estimated additions.
Its source report attributes the extra lines to preserving the SYRK helper and
call path absent from the standalone base. No assembled item exceeded twice its
estimated functional additions.

## Conflict resolutions

Four cherry-picks required manual resolution:

1. A1 conflicted with A0 in `experiments/grug/moe/launch_cw_scale.py`. A0 had
   added `SCALE_SCAN_LAYERS` and `use_array_stacked_blocks`; A1 added
   `SCALE_REPORT_DROPS` and `report_capacity_overflow`. The resolution keeps
   both environment variables in the launcher documentation and both fields in
   `GrugModelConfig`.
2. B2 conflicted with the Phase A launcher changes in the same file. The Phase A
   side carried `use_array_stacked_blocks` and `report_capacity_overflow`; B2
   added `moe_implementation`, `attention_implementation`, and their imports and
   environment resolution. The resolution keeps all four model fields and both
   B2 resolution paths.
3. C3 conflicted in two files:
   - In `experiments/grug/moe/model.py`, A0 had made block construction
     scan-aware, while C3 changed the output projection to `Plm_head_ep` against
     the former tuple-only block construction. The resolution uses
     `Plm_head_ep` and retains A0's conditional `ArrayStacked` construction,
     `stacked_blocks`, and return value.
   - In `lib/levanter/src/levanter/grug/sharding.py`, B3 had made
     `Pembed_vocab` replica-local while C3 introduced `Plm_head_dense`,
     `Pfsdp=("data", "expert")`, and `Plm_head_ep` against the old embedding
     spec. The resolution keeps B3's replicated `Pembed_vocab`, adds both C3
     lm-head layouts and the two-axis FSDP spec, and removes the overloaded
     `Plm_head`. Dense Grug uses `Plm_head_dense`; Grug MoE, June TPU MoE, and
     Snowball use `Plm_head_ep`.
4. A2 conflicted with B2's launcher import block. B2 needed
   `resolve_moe_implementation` and `GrugAttentionImplementation`; A2 needed
   `DEFAULT_EP_CAPACITY_FACTOR`. The resolution keeps all three imports.
   `capacity_factor` now resolves `SCALE_CAPACITY_FACTOR` before falling back to
   the canonical default, without removing any Phase A or B2 controls.

The manual C3 resolution initially wrapped one `output_proj` assignment
incorrectly. C3 was amended during an interactive rebase and C4 and A2 were
replayed, so the formatting fix is part of the first prefix that contains the
resolution rather than a later cleanup commit.

## Verification

Full repository checks ran at the requested boundaries:

| Prefix checked | Pre-commit | Pyrefly | Default pytest |
|---|---|---|---|
| Rebased documentation, through `970df8b14` | clean | 0 errors | 1 failed, 1,252 passed, 17 skipped, 47 deselected, 5 xfailed |
| Phase A, through `b4302d3aa` | clean | 0 errors | 1 failed, 1,256 passed, 18 skipped, 47 deselected, 5 xfailed |
| Phase B, through `6802bbffa` | clean | 0 errors | 1 failed, 1,257 passed, 18 skipped, 47 deselected, 5 xfailed |
| Stage 1 code, through `bc8ca819b` | clean | 0 errors | 1 failed, 1,261 passed, 18 skipped, 47 deselected, 5 xfailed |

Every default run had the one allowed pre-existing failure:
`tests/test_grug_variant_contracts.py::test_grug_base_run_emits_expected_metrics_with_json_tracker`.
The failure remains in untouched dense Grug explicit-sharding code, where
`experiments/grug/base/model.py` concatenates operands with
`P(("replica_dcn", "data"), None)` and `P(None, None)`. No other default test
failed.

The four boundary commits above received the full check directly. The
intermediate implementation commits did not each receive a separate full
repository run; each is covered by the next boundary run and by the focused
checks recorded in its attached source report.

With eight CPU devices, 19 applicable MuonH, optimizer, model, expert-boundary,
and chunk-drop tests passed. A broader 38-test selection produced 33 passes,
3 skips, and 2 failures because XLA CPU does not implement
`ragged-all-to-all`; both failures were the expected expert-parallel MoE tests
identified in the source reports.

The chunked Sonic backend was also inspected at its target symbol:
`_moe_mlp_local_sonic_cute_chunked` returns `_chunk_capacity_drops(...)`, while
the unchunked path and intermediate chunks retain zero drop counts. The
canonical B2 QuACK blobs are
`d8b6b520be800e0570017535276169072b1093ff` and
`628f77fdb2d64a69d5ddc13b87bb0f118e03ad4c`. The FA4 migration contains the
five prescribed `cute.make_rmem_tensor` substitutions and no remaining
top-level `cute.make_fragment` call.

## SYRK state after C2

`lib/levanter/src/levanter/optim/grugmuon.py` after C2 has blob
`3fa61d24f1dec7ee9cf0d605b176d1a4e6475083`, matching the C2 source commit.
The expert-axis branch still checks `SCALE_MUON_SYRK`, constructs a local SYRK
function, flattens only the shard-local `(local_layers, local_experts)` tile,
calls `_newtonschulz_batched_syrk` through `shard_map`, and reshapes the result
back to `orig_4d_spec`. The nested-`vmap` alternative and the non-expert SYRK
path also remain reachable.

This verifies the control flow and sharding structure, including focused CPU
tests, but not a Blackwell SYRK compile or execution. No GPU or cluster job was
submitted.

## Plan differences and remaining limits

- `sequence.md` still describes a series against
  `origin/main@1c631c4c0`; this assembly used the assigned current base
  `6ce4a7e68`.
- The C1/C2 report existed only in the source branch's later D4 commit. It was
  attached to C2 so the stage 1 evidence is visible. Its D4 sections describe
  deferred source-branch work, not code present in this stage.
- The FA4 source branch began at `ac6b03aef`, not the assembly base. Its five
  requested substitutions applied cleanly and were verified at the intended
  symbols; no dependency-range edit or wheel upgrade was carried.
- A2 was left last as requested. It caused only the documented B2 import
  conflict, and no semantic dependency justified moving it earlier.
- Local checks did not compile the SM100 QuACK kernels, FA4, or the SYRK path on
  an accelerator. They also do not reproduce rack-scale warning counts or
  memory behavior. Those measurements remain for the later rack gate.
