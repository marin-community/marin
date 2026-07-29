# B1 and B2 extraction report

## Result

B1 is commit `47c31192c`. It adds `quack-kernels[cu13]==0.6.1` to the
Levanter GPU extra, updates the lock from QuACK 0.5.0 to 0.6.1, and lets
Pyrefly ignore the optional `cutlass` and `quack` modules on CPU-only
installations.

B2 adds the QuACK SM100 grouped-GEMM backend and wires `SCALE_MOE_IMPL` and
`SCALE_ATTN_IMPL` through `launch_cw_scale.py`. `sonic_cute` is registered as
a local backend, and its import remains inside the selected dispatch branch.
The default, CPU, and non-`sonic_cute` paths do not import QuACK or CUTLASS.

The three backend files remain byte-identical to both source tips:

| file | lines | blob |
|---|---:|---|
| `sonic_cute.py` | 272 | `4d536270605b52895dac9b8932952fa41af1ed9f` |
| `quack_moe_cute.py` | 172 | `d8b6b520be800e0570017535276169072b1093ff` |
| `quack_symmetric_cute.py` | 117 | `628f77fdb2d64a69d5ddc13b87bb0f118e03ad4c` |

I resolved each blob independently at `origin/b200-300B-tune` and
`agent/ep25-d1-adjoint` before copying it. The hashes matched across the two
tips. The stale `5cf76b64a` file is blob `c747e6d2ce`, has 105 lines, and has
no `quack_symmetric_cute.py`.

## Diff size

B1 has +7 functional TOML lines against the estimate of about +10. Its
generated lockfile change is +10/-3. B2 has +584 functional lines against the
estimate of about +560: 561 lines in the three exact source blobs and 23 lines
of registration, lazy dispatch, and launcher wiring. The B2 ratio is 1.04x the
estimate.

## Extraction decisions and dropped work

I kept main's exact `nvidia-cutlass-dsl[cu13]==4.6.0` pins in both Levanter and
Marin. I dropped `538381606`'s `>=4.6.0,<4.7` changes. I did not take the
temporary smoke script from `458f647b5` or `d7decc466`.

I verified the `device_flops` premise before dropping `c81a29428` and
`7f504d9cb`. Main's `"b200"` entry uses 1.25 PFLOP/s dense TF32, 2.5 PFLOP/s
dense FP16/BF16, 5 PFLOP/s dense FP8/INT8, and 10 PFLOP/s dense FP4. These are
the same numbers as the source branch's `"gb200"` entry.

I did not wire `SCALE_MOE_EXPERT_CHUNKS`, slim Sonic residuals, Muon SYRK, the
temporary GPU smoke, or any experiment-only launcher changes. B2 includes the
canonical chunk helper functions because they are part of the required
272-line blob, but no main call site can select them.

## Canonical lineage and deferred conflicts

The canonical B2 lineage is blob `4d53627060`, shared by the FSDP and EP tips.
It contains the squashed chunk helper implementation but not the
`SCALE_MOE_EXPERT_CHUNKS` selection and weight-prefetch changes in
`grug_moe.py`. The chunk history includes four add/revert pairs across the
original and replayed histories; a later port should take the squashed end
state.

Chunking and slim residuals cannot be overlaid mechanically. Chunking adds
expert- and intermediate-dimension entry points that call `_expert_mlp` with
sorted segments, padded token indices, and one gathered weight chunk at a
time. Commit `59e5fe25f` starts from the 105-line pre-chunk blob and changes
`_expert_mlp`'s custom-VJP signature and residual from
`(x_dispatch, weights, h, routing)` to the original `x`, `token_dispatch`,
sharded weights, and recomputed activation. Applying it to the canonical file
would leave both chunk entry points calling the old signature. A combined
implementation must define how each padded or dynamically sliced chunk
reconstructs `x_dispatch`, how gathered weights are stored sharded in the
backward residual, and how padding rows receive zero cotangents. The slim
lineage also adds the `all_but_moe` rematerialization mode outside this file.

The chunk drop-accounting bug is not present on main at the assignment base
because main has no chunk backend or selector. The exact canonical blob does
contain the dormant bug: `_moe_mlp_local_sonic_cute_chunked` returns
`_zero_dropped_assignments()` even though its fixed per-chunk capacity can
drop assignments. Commit `cefc6d47b` changes that return to
`_chunk_capacity_drops(cu, bounds, caps)` and adds the supporting metric code.
That fix is not an ancestor of either source tip. It must accompany any later
chunking enablement.

## CUTLASS wheel hazard

The current lock contains both `nvidia-cutlass-dsl-libs-base` and
`nvidia-cutlass-dsl-libs-cu13`, both at 4.6.0. Commit `8f1ba5363` deliberately
removed the older `libs-base==4.5.2; sys_platform == "never"` override when
main moved to CUTLASS DSL 4.6.0, where the upstream shadowing issue was fixed.
Adding QuACK 0.6.1 kept CUTLASS at the exact 4.6.0 pin; `uv lock --check`
passed and did not produce mixed CUTLASS versions. I did not run a GPU-extra
sync on multiple pods, so I did not independently reproduce install
determinism.

## Verification

- `uv lock --check`: passed.
- CPU launcher probe with `SCALE_MOE_IMPL=sonic_cute` and
  `SCALE_ATTN_IMPL=gpu_fa4_cute`: built the config and confirmed that no
  `quack` or `cutlass` module entered `sys.modules`.
- `uv run --package marin-levanter --group test pytest
  lib/levanter/tests/grug/test_grugformer_moe.py`: 13 passed, 6 skipped.
- `./infra/pre-commit.py --all-files --fix`: passed, including Pyrefly.
- `uv run pyrefly check`: 0 errors, 408 suppressed, 505 warnings not shown.
- `uv run pytest`: 1,252 passed, 17 skipped, 47 deselected, 5 xfailed, and 1
  failed. The failure is
  `test_grug_base_run_emits_expected_metrics_with_json_tracker`, where
  `experiments/grug/base/model.py:227` concatenates arrays with different
  explicit CPU shardings. The failure reproduces in isolation. The test,
  `base/model.py`, and `base/train.py` are byte-identical to `origin/main`;
  this series does not change JAX or Equinox. I did not modify this unrelated
  main failure.

No SM100 GPU was used, so I did not compile the CuTe kernels, compare
numerics or gradients, or inspect GPU HLO. The source record reports whole-model
and gradient-parity validation, but this extraction only verifies provenance,
static checks, CPU import isolation, and the existing CPU MoE suite.

## Plan mismatches and uncertainties

Commit `538381606` contains the QuACK dependency but does not contain the
`cutlass`/`quack` Pyrefly ignore entries. Those entries first appear with the
backend commits `5cf76b64a` and `6cad4a4d7`; I included them in B1 as assigned.
The commit message for `538381606` says the dependency is added to both
Marin Core and Levanter, but its actual Marin hunk only changes the CUTLASS
range. The QuACK dependency is present only in Levanter's GPU extra.

The exact `quack_symmetric_cute.py` blob contains an import fallback between
two QuACK module locations, despite the series' no-compatibility-fallback rule.
I preserved it because the assignment requires the exact shared tip blob.
Pinned QuACK 0.6.1 contains the preferred
`quack.cute_dsl_utils.get_max_active_clusters` symbol.

The literal command `uv run pyrefly` no longer runs a check with locked Pyrefly
1.0.0; it prints the command help and exits 2. `uv run pyrefly check` is the
current equivalent and passed. The repository pre-commit wrapper also ran the
type checker successfully.
