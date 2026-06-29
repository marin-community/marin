# Hopper Pallas MGPU MoE PR Readiness Notes

This note tracks what the first stacked PR for issue
[#6597](https://github.com/marin-community/marin/issues/6597) must say or prove
before review. It is a coordination artifact for the implementation in
`lib/levanter/src/levanter/grug/_moe/pallas_mgpu.py` and the benchmark harness in
`lib/levanter/scripts/bench/bench_grug_moe_pallas_mgpu.py`.

## Current Status

- Public API: `implementation="pallas_mgpu"` is wired through the Grug MoE
  implementation selector and returns `(out, dropped_count)` through the existing
  `report_capacity_overflow` convention. Unsupported public requests fail before
  backend lowering for static shape/dtype issues, missing local GPUs,
  mixed or non-Hopper participating local GPUs, and expert axes that exceed
  visible local GPU devices.
- Forward path: the public backend executes the two intended staged MGPU forward
  kernels, `permute_up` and `down_unpermute`, with deterministic source-local
  assignment IDs and fixed route-slot combine order.
- Backward path: the public custom-VJP boundary is restored. Current default
  backward saves forward receive/hidden/W2-dispatch residuals for the
  non-chunked path rather than recomputing the full forward prerequisite stage.
- Capacity: receiver capacity is padded to satisfy Mosaic WGMMA M tiling when
  needed, and the backend warns when padding changes a default-like requested
  capacity.
- Benchmark harness: target-shape and diagnostic-stage runs emit
  machine-readable rows with compile time, steady-state time, row status,
  measurement key, shape, dtype, device, block sizes, routing, error text, git
  SHA, XLA flags, backend environment, estimated FLOPs/bytes/memory footprint,
  and allclose tolerances.
- Tuning: a small static config table now routes the public Pallas MGPU path to
  the current H100 bf16 single-node defaults. Unknown buckets fall back to
  `MoeMgpuConfig`; autotune-on-miss remains a follow-up.
- Full-run tryout: the Grug MoE scale launcher accepts
  `SCALE_MOE_IMPLEMENTATION` and `SCALE_MOE_CAPACITY_FACTOR`, and
  `.agents/projects/20260628_moe_mgpu_full_run_tryout.md` records the current
  best one-node and experimental multi-node 20-step H100 launch commands. The
  launcher also exposes `SCALE_GPUS_PER_TASK` for task decomposition experiments
  but fails fast for `pallas_mgpu` when that would hide expert-parallel ranks
  from a local process. The
  current-commit one-node target-shape trainer smoke passed 20/20 steps with
  watch disabled in
  `/dlwh/grug-moe-pallas-mgpu-20step-current-20260629-150755`; the first
  32-node attempt was externally killed before training. A later larger-shape
  B-tiled R=N scale sweep reached pre-step initialization but OOMed before any
  train-step metrics. Neither larger-scale run is scale correctness or
  performance evidence.

## Current Best Evidence

- Correctness:
  - H100 target forward matches `ragged_all_to_all` within harness tolerance on
    the target EP=8/top_k=4 shape.
  - H100 public forward and gradient parity against `ragged_all_to_all` still
    passes after the mixed-local-GPU topology validation hardening in
    `/dlwh/iris-run-test_grugformer_moe-20260629-topology-hardening-public-refresh`:
    `3 passed, 107 deselected, 1 warning in 218.99s`.
  - The broader H100 Hopper Pallas slice, including the opt-in chunked
    `permute_up` parity guard, passed after the lint-review dispatcher cleanup
    in `/dlwh/iris-run-test_grugformer_moe-20260629-lint-cleanup-broad-plus-chunked`.
  - H100 public module-boundary training-step integration through
    `MoEExpertMlp.init(..., implementation="pallas_mgpu")` passed in
    `/dlwh/iris-run-test_grugformer_moe-20260629-module-training-step`:
    `1 passed, 110 deselected, 1 warning in 63.41s`.
  - Current local Grug MoE, public validation/fallback, and benchmark-harness
    tests passed after the public device/topology fail-fast patch and latest
    readiness refresh:
    `41 passed, 11 warnings in 27.96s` for the benchmark harness,
    `38 passed, 11 warnings in 16.06s` for the public validation/fallback slice,
    and `83 passed, 28 skipped, 11 warnings in 35.88s` for the full local Grug
    MoE test file.
  - Latest focused local readiness refresh on the clean current tree:
    benchmark harness
    `uv run pytest lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py -q -o addopts=`
    passed `41 passed, 1 warning in 2.50s`; focused Grug MoE Pallas/capacity/EP
    selector
    `uv run pytest lib/levanter/tests/grug/test_grugformer_moe.py -q -k 'pallas_mgpu or capacity_overflow or expert_parallel' -o addopts=`
    passed `62 passed, 11 skipped, 38 deselected, 1 warning in 16.21s`.
  - Focused local tests cover public fallback/validation behavior, readable
    reference shape checks, capacity padding, zero/empty-group behavior, and
    benchmark row schema.
  - The focused public ordered-fallback audit in `MOE-MGPU-311` passed:
    `7 passed, 11 warnings in 26.25s`.
  - The static readiness audit (`MOE-MGPU-271`) confirmed the
    symbols and tests back the PR-readiness claims for the public backend,
    staged kernel boundaries, minimal metadata, stable assignment ordering,
    `num_sms=None` core-count behavior, benchmark row artifacts, and fallback /
    rejection coverage.
  - The post-compaction re-audit (`MOE-MGPU-277`) reconfirmed those key claims
    against the then-current code/tests, including zero checked-in
    `pl.pallas_call` sites in `pallas_mgpu.py`, Mosaic `mgpu.kernel`
    launch-wrapper usage, no literal `132` SM default, and `num_sms=None` paths
    querying `jax.devices()[0].core_count`.
  - The latest static spec-invariant audit (`MOE-MGPU-340`) reconfirmed public
    API wiring, minimal production metadata, deterministic assignment/remote-row
    evidence, no remote atomic combine, no checked-in `pl.pallas_call(...)`, no
    fake `cost_estimate`, and expected H100/test coverage hooks.
  - The cost-estimate readiness audit (`MOE-MGPU-290`) reverified that
    `pallas_mgpu.py` had zero checked-in `pl.pallas_call(...)` sites and used
    `mgpu.kernel(...)` launch wrappers. The installed
    `jax.experimental.pallas.mosaic_gpu` wrapper does not expose a consumed
    `cost_estimate=` hook for those launches.
  - The latest cost-estimate audit (`MOE-MGPU-321`) confirms the current file
    still has zero checked-in `pl.pallas_call(...)` sites and now has `18`
    `mgpu.kernel(...)` launch-wrapper sites; the installed Mosaic MGPU wrapper
    signatures still do not expose a consumed `cost_estimate=` hook for these
    launches.
  - The mixed-local-GPU public fail-fast guard added in `MOE-MGPU-274` validates
    every participating local GPU device, not only the first visible GPU. The
    local focused topology selector passed (`3 passed, 11 warnings`) and the
    broader `pallas_mgpu_rejects` selector passed (`28 passed, 11 warnings`).
- Performance:
  - Best observed target forward row: `pallas_mgpu` about `0.03853s`; comparable
    `ragged_all_to_all` baseline about `0.08204s`.
  - The forward-performance lane's best validated opt-in chunked `permute_up`
    stage row is currently `0.0092931247s`, `231.08 TFLOP/s/rank`, `23.37%`
    nominal H100 bf16 roofline per rank on the target balanced shape with
    serial fused group32, copytile512, copy_rows=1
    (`/dlwh/bench_grug_moe_pallas_mgpu-20260629-052330-copytile-nearby-sweep`).
    Split target compare
    `/dlwh/bench_grug_moe_pallas_mgpu-20260629-052800-copyrows1-copytile512-splitcompare`
    matched the staged baseline exactly (`max_abs_diff=0`, dropped=0). This is
    an opt-in benchmark path and remains below the forward roofline goal.
    Faster timing-only copytile640/1280 rows are invalid because split compares
    showed hidden mismatches. Static-workqueue experiments in a separate
    forward worktree were correct but slower at target shape, so they are not
    integrated here.
  - Current gated target full forward+backward row after the static tuned-config
    lookup and lint-review dispatcher cleanup:
    `/dlwh/iris-run-bench_grug_moe_pallas_mgpu-20260629-target-fwd-bwd-lint-cleanup`,
    `steady_state_time=0.069388s`, `139.27 TFLOP/s/rank`, `14.08%` nominal H100
    bf16 roofline per rank, no dropped routes/error.
  - Current target saved-residual backward breakdown:
    `/dlwh/iris-run-bench_grug_moe_pallas_mgpu-20260629-target-saved-bwd-breakdown`.
    The new `saved_backward_pipeline` diagnostic row is `0.035685s`; component
    rows are `combine_bwd=0.010571s`, `w2_bwd=0.005046s`,
    `w13_bwd=0.015593s`, and `dx_unpermute_vector=0.004998s`.
    This keeps the next non-forward optimization focus on W13 VJP and
    `combine_bwd`, not the direct pull/combine dx alternative.
  - Capacity sensitivity: four uniform-routing target-shape H100 seeds at
    `capacity_factor=1.125` produced no dropped routes in
    `/dlwh/iris-run-bench_grug_moe_pallas_mgpu-20260629-bwd-capacity1125-multiseed`.
    A balanced-routing target public fwd+bwd run at the same capacity factor
    `/dlwh/iris-run-bench_grug_moe_pallas_mgpu-20260629-target-fwd-bwd-cap1125-balanced`
    also produced zero dropped routes and ran in `0.065201s`, compared with the
    current default-capacity baseline around `0.069388s`. This supports a later
    capacity sensitivity study, but does not change the conservative default
    `capacity_factor=1.25` for the first PR.
- Full trainer tryout:
  - Current-commit one-node target-shape Grug MoE trainer smoke
    `/dlwh/grug-moe-pallas-mgpu-20step-current-20260629-150755` completed
    `20/20` steps on one 8xH100 node with `implementation="pallas_mgpu"`,
    `capacity_factor=1.25`, `SCALE_REMAT=save_moe`, local checkpoints, JSON
    logging, and `SCALE_WATCH_TARGETS=`. It saved checkpoint `step-20`,
    reported final `train/loss=7.496264457702637`,
    `train/cross_entropy_loss=7.468044757843018`, mean MFU
    `20.108652513850707%`, p50 MFU `20.118615659083623%`, and a final
    throughput sample of `552901.8144131473` tokens/s. This is recorded in
    `MOE-MGPU-363` and summarized on #6597.
  - One-node target-shape Grug MoE trainer smoke
    `/dlwh/grug-moe-pallas-mgpu-20step-smoke-2d87348e7` completed `20/20`
    steps on one 8xH100 node with `implementation="pallas_mgpu"`,
    `capacity_factor=1.25`, `SCALE_REMAT=save_moe`, local checkpoints, JSON
    logging, and `SCALE_WATCH_TARGETS=`. It saved checkpoint `step-20`, reported
    final `train/loss=7.496628761291504`,
    `train/cross_entropy_loss=7.468381404876709`, mean MFU
    `20.127821040818162%`, p50 MFU `20.15132069436141%`, and a final
    throughput sample of `554162.1451621387` tokens/s. This is recorded in
    `MOE-MGPU-347` and summarized on #6597.
  - A first 32-node/256-H100 20-step scale attempt
    `/dlwh/grug-moe-pallas-mgpu-20step-scale-fe788a6e5-20260629-134609` was
    killed externally while all 32 child tasks were still building, before any
    training progress, metrics, checkpoint, OOM, traceback, pending capacity, or
    Iris failure signal appeared. This is recorded in `MOE-MGPU-349` as an
    operational interruption only.
  - A later B-tiled CE R=N larger-shape scale sweep
    `/dlwh/grug-moe-pallas-mgpu-btiled-r{4,8,16,32}-scale-n{4,8,16,32}-ed130604e-20260629-140300`
    was killed after n4, n8, and n16 jobs hit pre-step GPU OOM while allocating
    576 MiB during `int(state.step)`, before any train-step metrics. This is
    recorded in `MOE-MGPU-351` as larger-model scale-recipe evidence, not a
    kernel milestone.
- Observability:
  - H100 diagnostic stage progress schema smoke
    `/dlwh/iris-run-bench_grug_moe_pallas_mgpu-20260629-stage-progress-schema`
    succeeded and showed `measurement_key` on stage start events and result rows.
  - H100 benchmark gate smoke after the benchmark capacity-default alignment
    `/dlwh/iris-run-bench_grug_moe_pallas_mgpu-20260629-default-capacity-smoke`
    succeeded with `--fail-on-error`, emitted ok `permute_metadata` and
    `permute_values` rows on 8 H100s, confirmed the deleted hook field is no
    longer present in `block_sizes`, and confirmed omitted `--capacity-factor`
    uses `1.25`.
  - The benchmark CLI default for `--capacity-factor` now comes from
    `MoeMgpuConfig().capacity_factor`, keeping harness defaults aligned with the
    spec and kernel config default of `1.25`; target H100 commands still pass the
    value explicitly.

See `.agents/logbooks/6597-moe-mgpu.md` for current exact commands and run
outputs. The 2026-06-28 entries are archived in
`.agents/logbooks/6597-moe-mgpu-20260628.md` to keep each logbook file below the
repository large-file threshold.

## Checked-In Defaults

The spec's `MoeMgpuConfig` block records an initial seed configuration. The
checked-in defaults are the current reviewed runtime defaults and are what the
benchmark harness emits when CLI flags are omitted:

```text
block_m=64
block_n=128
block_k=64
max_concurrent_steps=4
grid_block_n=2
capacity_factor=1.25
num_sms=None
deterministic=True
dispatch_copy_schedule="assignment_major"
dispatch_expert_group_size=8
dispatch_chunk_copy_tile=128
dispatch_chunk_copy_rows=1
dispatch_chunk_vectorized_copy_rows=False
dispatch_fuse_metadata=True
dispatch_chunked_permute_up=False
dispatch_split_wg_permute_up=False
dispatch_split_wg_overlap_permute_up=False
combine_bwd_block_n=512
dx_unpermute_block_n=2560
```

The two differences from the initial seed, `max_concurrent_steps=4` and
`grid_block_n=2`, are treated as the current tuned defaults for this branch.
`num_sms=None` means kernels query `jax.devices()[0].core_count`; there is no
hard-coded `132` SM default.

## Explicit Limitations

These should be stated in the PR body and any user-facing docs for this backend:

- Hopper/H100 only.
- EP must be local and `EP <= 8`.
- Single-node NVLink domain only.
- No NIC, InfiniBand, NVSHMEM, NCCL expert-parallel path, or multi-host expert
  parallelism.
- bfloat16 activations and expert weights are required by the Pallas path.
  `combine_weights` may be bfloat16 or float32.
- The target shape is top_k=4. Other K values are only supported to the extent
  covered by focused tests and benchmark rows.
- No FP8 support.
- No duplicate-destination optimization for multiple top-k routes sent to the
  same destination rank.
- No remote atomics for combine. The current production `down_unpermute` avoids
  a materialized return-slot buffer by doing deterministic source-side remote
  reads from expert-owner W2 dispatch rows and accumulating routes locally in
  fixed route-slot order.
- Capacity overflow is deterministic receiver-side clipping before kernel
  execution; there is no in-kernel dynamic overflow handling.
- Current optimized backward is still a follow-on optimization area even though
  public gradient parity is covered.

## Cost Estimate Status

The spec asks for `cost_estimate=` on checked-in `pl.pallas_call` or Mosaic
wrappers where the API supports it. The current backend uses
`mgpu.kernel(...)` launch wrappers throughout `pallas_mgpu.py`; there are no
checked-in `pl.pallas_call(...)` sites in this file.

The installed `jax.experimental.pallas.mosaic_gpu` wrapper does not expose a
consumed `cost_estimate=` argument in this usage. `inspect.signature(pl.pallas_call)`
does include `cost_estimate=`, but the current file has no checked-in
`pl.pallas_call(...)` sites. `inspect.signature(mgpu.kernel)` shows explicit
launch arguments plus `**mesh_kwargs`, and `inspect.getsource(mgpu.kernel)`
does not mention `cost_estimate`. `inspect.signature(mgpu.Mesh)` accepts only
`grid`, `grid_names`, `cluster`, `cluster_names`, `num_threads`, `thread_name`,
and `kernel_name`, so passing `cost_estimate=` would be an unsupported Mesh
kwarg rather than a Pallas cost estimate hook.

Do not add fake cost estimates that are not consumed by the actual launch API.

## Open Gaps Before First PR

- `#6597-forward` owns the next `permute_up` forward-performance changes. Main
  should coordinate in that room before touching forward scheduling or shared
  benchmark defaults. Latest coordination poll saw no handoff to main; the
  best validated opt-in chunked stage result is `0.0092931247s` /
  `23.37%` roofline with group32/copytile512/copy_rows=1. This is below the
  forward target and not promoted to the default public path. Separate
  static-workqueue experiments are correct-but-slower and live outside this
  worktree.
- If the first PR remains forward-first, it should say that backward performance
  is present but still below the final roofline target, with the current
  saved-residual custom-VJP baseline as evidence.
- Before opening the PR, rerun focused local tests and touched-file
  `./infra/pre-commit.py --fix` for any further code or test changes. The
  current active diff is covered by `MOE-MGPU-241` through `MOE-MGPU-343`.
  The latest local benchmark-harness refresh is covered by `MOE-MGPU-325`:
  benchmark harness `41 passed`. The latest public validation/fallback refresh
  is covered by `MOE-MGPU-327`: public validation/fallback slice `38 passed`.
  The latest full local Grug MoE
  file is covered by `MOE-MGPU-325`: Grug MoE file `83 passed, 28 skipped`.
  The latest deterministic all-files precommit refresh is covered by
  `MOE-MGPU-329`. The latest full active tracked and untracked file-set
  precheck is covered by `MOE-MGPU-333`. The latest H100 public parity refresh
  after mixed-local-GPU topology hardening is covered by `MOE-MGPU-275`. The
  latest capacity sensitivity evidence is covered by `MOE-MGPU-281` and
  `MOE-MGPU-291`; it is logbook-only evidence and does not change the first-PR
  default capacity. The PR readiness note and living-summary evidence refreshes
  are covered by `MOE-MGPU-282`, `MOE-MGPU-284`, `MOE-MGPU-289`,
  `MOE-MGPU-291`, `MOE-MGPU-337`, `MOE-MGPU-338`, `MOE-MGPU-339`,
  `MOE-MGPU-341`, and `MOE-MGPU-342`, and the extraction-surface audits are
  covered by `MOE-MGPU-285`, `MOE-MGPU-331`, and `MOE-MGPU-341`. The latest
  cost-estimate API audit is
  covered by `MOE-MGPU-290` and refreshed by `MOE-MGPU-321`. The stale Meitner
  babysit heartbeat was resolved in `MOE-MGPU-292` and freshly re-polled in
  `MOE-MGPU-331`/`MOE-MGPU-338`, and the latest padding/config/public-validation
  readiness audit is covered by `MOE-MGPU-295`. The latest spec-tail
  down/unpermute/backward static audit and local selector are covered by
  `MOE-MGPU-298`; the latest high-risk spec-invariant static audit is covered
  by `MOE-MGPU-340`. The latest full active tracked and untracked file-set
  precheck is covered by `MOE-MGPU-333`. The latest all-files precommit refresh
  is covered by `MOE-MGPU-329`. The latest readiness-note/logbook Markdown
  hygiene after the extraction dry-run sync is covered by `MOE-MGPU-342`.
  The latest H100 module-boundary training-step integration check is covered by
  `MOE-MGPU-323`.
  The latest broad local test refresh after that H100 test is covered by `MOE-MGPU-325`;
  the latest public validation/fallback refresh is covered by `MOE-MGPU-327`.
  The latest readiness evidence text cleanups are covered by `MOE-MGPU-304`
  and `MOE-MGPU-305`.
  The latest forward-room coordination checkpoint is covered by
  `MOE-MGPU-307`.
  Rerun at least one H100 correctness/benchmark smoke from the logbook if there
  are behavior-changing backend, public API, or benchmark-harness changes after
  those entries; documentation/logbook-only changes only need the local
  Markdown/precheck hygiene already recorded in the logbook. The latest
  full-run tryout launcher/runbook wiring and changed-file precommit are
  covered by `MOE-MGPU-343`; the successful one-node 20-step trainer smoke is
  covered by `MOE-MGPU-347`, and the externally killed pre-training 32-node
  attempt is covered by `MOE-MGPU-349`, and the later B-tiled R=N larger-shape
  pre-step OOM sweep is covered by `MOE-MGPU-351`.
- `./infra/pre-commit.py --review` previously reported actionable advisories,
  and the latest concrete findings from `/tmp/marin-linter/20260629T184150`
  were addressed in `MOE-MGPU-316`. The latest rerun could not execute any
  review lanes because the lint agent quota was exhausted; every lane reported
  `You've hit your limit · resets 4:30pm (America/Los_Angeles)` in
  `/tmp/marin-linter/20260629T185553` (`MOE-MGPU-316`). Rerun it after that
  quota reset before PR extraction.

## Extraction Surface

For the first stacked production PR, include the implementation and validation
artifacts needed by reviewers:

- `lib/levanter/src/levanter/grug/_moe/common.py`
- `lib/levanter/src/levanter/grug/_moe/pallas_mgpu.py`
- `lib/levanter/src/levanter/grug/grug_moe.py`
- `lib/levanter/tests/grug/test_grugformer_moe.py`
- `lib/levanter/scripts/bench/bench_grug_moe_pallas_mgpu.py`
- `lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py`
- `experiments/grug/moe/model.py`
- `experiments/grug/moe/launch_cw_scale.py`

Keep the research coordination artifacts on the research branch unless the
reviewer explicitly wants them in the production PR:

- `.agents/projects/20260628_moe_mgpu.md`
- `.agents/projects/20260628_moe_mgpu_full_run_tryout.md`
- `.agents/projects/20260628_moe_mgpu_pr_readiness.md`
- `.agents/logbooks/6597-moe-mgpu.md`
- `.agents/logbooks/6597-moe-mgpu-20260628.md`
- `.agents/logbooks/6597-moe-mgpu-forward.md`

The PR body should link the tracking issue and logbook path and quote the
relevant validation/benchmark rows from this note instead of requiring reviewers
to read the full logbook.

Latest extraction dry-run (`MOE-MGPU-341`) confirmed the earlier changed
production files were the six implementation/test/benchmark files listed
above. `MOE-MGPU-343` intentionally adds the two Grug MoE scale-launcher files
so the snapshot can be tried in a real 20-step run. The research coordination
files remain branch-local unless explicitly requested.

## Spec Compliance Snapshot

| Requirement | Status | Evidence / note |
| --- | --- | --- |
| Public `implementation="pallas_mgpu"` backend | Done | `grug_moe.py` dispatches to `_moe_mlp_ep_pallas_mgpu_local`; public selectors pass locally and on H100. |
| Two forward kernel boundaries | Done | `permute_up_mgpu` and `down_unpermute_mgpu` are the public staged Pallas MGPU forward path. |
| Deterministic assignment ordering | Done | Stable assignment sorts are used in `common.py` and `pallas_mgpu.py`; repeated-output H100 coverage exists. |
| Minimal return metadata | Done | Production metadata is `recv_src_rank` and `recv_src_assignment`; token/route are derived from assignment id. |
| No atomic combine | Done | Production `down_unpermute` uses deterministic source-side pull/combine in fixed route-slot order; no remote atomic combine path is used. |
| Capacity clipping and padding | Done | Receiver-side clipping is used, WGMMA M capacity padding is implemented, and default-like padding warnings are tested. |
| Explicit fail-fast for unsupported explicit backend requests | Done | Public and backend validation reject missing/non-Hopper GPU, EP > 8, non-local expert axes, dtype, static-shape, and tile mismatches; ordered fallback is separately tested. |
| Forward correctness on H100 | Done | Public H100 forward parity against `ragged_all_to_all` passed, including EP=8/top_k=4 coverage. |
| Gradient/custom-VJP correctness on H100 | Done for current implementation | Public H100 gradient parity against `ragged_all_to_all` passed; backward performance remains a follow-up optimization area. |
| Module-boundary training-step integration on H100 | Done | `MoEExpertMlp.init(..., implementation="pallas_mgpu")` passes a tiny H100 differentiable loss/update check in `/dlwh/iris-run-test_grugformer_moe-20260629-module-training-step`. |
| Target-shape performance evidence | Done | Target forward is faster than `ragged_all_to_all`; current lint-cleanup target fwd+bwd row is recorded above. |
| One-node full-trainer smoke | Done | `/dlwh/grug-moe-pallas-mgpu-20step-current-20260629-150755` completed 20/20 steps from commit `14fd25a73`, saved checkpoint `step-20`, and reported mean MFU `20.11%`. |
| 32-node full-scale smoke | Not yet proven | `/dlwh/grug-moe-pallas-mgpu-20step-scale-fe788a6e5-20260629-134609` was externally killed before training; the later B-tiled R=N larger-shape sweep hit pre-step GPU OOM before train metrics. Reduce model or optimizer-state memory before relaunching scale evidence. |
| Machine-readable benchmark rows | Done | Benchmark rows include identity, timing, status/error, capacity/padding, estimates, tolerances, and environment fields; duplicate keys and row counts are tested. |
| `cost_estimate=` | Documented API limitation | This backend uses `mgpu.kernel(...)`, not checked-in `pl.pallas_call(...)`; inspected `mgpu.kernel`/`mgpu.Mesh` signatures show no consumed `cost_estimate=` hook in this usage. |
| Static tuned config lookup | Done | `infer_moe_mgpu_config(...)` returns the current H100 bf16 single-node defaults for the reviewed shape bucket, preserves explicit capacity factors for matched and fallback buckets, and falls back to `MoeMgpuConfig` for unknown buckets. |
| Autotune-on-miss | Follow-up / not part of first PR | Current work records explicit best configs and bounded benchmark rows; no runtime autotune-on-miss path is claimed for the first PR. |
| Roofline targets | Partially met / documented | Forward improves realistic target shape but remains below final roofline targets; current limitations and next optimization ownership are documented. |

## PR Body Draft

```md
Summary:
- Add `implementation="pallas_mgpu"` for Grug expert-parallel MoE on a single
  local H100/NVLink expert-parallel group.
- Implement the two spec-aligned forward kernel boundaries: `permute_up` and
  `down_unpermute`, with deterministic assignment ordering, receiver-side
  capacity clipping, fixed route-slot combine order, and no remote atomics.
- Add the public custom-VJP boundary with saved forward residuals for the
  current non-chunked path, plus benchmark artifacts for forward, stage, and
  forward+backward timing.
- Add a static tuned-config lookup for the reviewed H100 bf16 single-node bucket
  with conservative fallback to `MoeMgpuConfig` for unknown buckets.

Validation:
- Local benchmark harness:
  `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py -q`
  passed on the clean current tree:
  `41 passed, 1 warning in 2.50s`.
- Local public validation/fallback slice:
  `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_grugformer_moe.py -q -k 'ordered_implementation or invalid_direct_entrypoint_inputs or tuned_config or pallas_mgpu_rejects'`
  passed after the module-boundary H100 test:
  `38 passed, 11 warnings in 16.06s`.
- Local focused Pallas/capacity/EP selector:
  `uv run pytest lib/levanter/tests/grug/test_grugformer_moe.py -q -k 'pallas_mgpu or capacity_overflow or expert_parallel' -o addopts=`
  passed on the clean current tree:
  `62 passed, 11 skipped, 38 deselected, 1 warning in 16.21s`.
- Full local Grug MoE test file:
  `uv run --package marin-levanter --group test pytest lib/levanter/tests/grug/test_grugformer_moe.py -q`
  passed after the module-boundary H100 test:
  `83 passed, 28 skipped, 11 warnings in 35.88s`.
- Repository deterministic prechecks:
  `./infra/pre-commit.py --all-files --fix` passed after the latest
  module-boundary H100 test and local-validation refresh. The latest
  deterministic all-files precommit refresh is recorded in `MOE-MGPU-329`:
  Ruff, Black, license headers, Pyrefly, large files, Python AST, merge
  conflicts, TOML/YAML, trailing whitespace, EOF newline, notebooks, Markdown
  pre-commit, and skill metadata all reported `ok`.
  The latest clean-tree focused local refresh is `MOE-MGPU-365`: benchmark
  harness `41 passed`, focused Pallas/capacity/EP selector
  `62 passed, 11 skipped`.
  The active tracked and untracked file-set precheck is recorded in
  `MOE-MGPU-333`, including all logbooks, readiness notes, backend files, tests,
  and the benchmark harness. The latest local benchmark harness, public
  validation/fallback, and full Grug MoE file refreshes are recorded in
  `MOE-MGPU-325`, `MOE-MGPU-327`, and `MOE-MGPU-325`. The latest
  local/static down-unpermute and backward spec-tail audit is recorded in
  `MOE-MGPU-298`; that local selector passed `2 passed, 5 skipped` with the
  H100-only kernel/gradient checks skipped locally and covered by earlier H100
  evidence above. The latest capacity sensitivity experiments are recorded in
  `MOE-MGPU-281` and `MOE-MGPU-291` and are logbook-only evidence.
  The remaining `./infra/pre-commit.py --review` gate is agent-quota blocked
  until `2026-06-29 16:30 America/Los_Angeles` in the latest retry after
  lint-review cleanup.
- Direct Pyrefly:
  `uvx --from 'pyrefly>=1.0.0,<1.1.0' pyrefly check --baseline .pyrefly-baseline.json`
  passed after the static tuned-config lookup:
  `0 errors (422 suppressed, 521 warnings not shown)`.
- H100 public forward/gradient parity:
  `/dlwh/iris-run-test_grugformer_moe-20260629-topology-hardening-public-refresh`
  on `cw-us-east-02a` passed after the mixed-local-GPU topology validation
  hardening:
  `3 passed, 107 deselected, 1 warning in 218.99s`.
- H100 broad Hopper Pallas slice:
  `/dlwh/iris-run-test_grugformer_moe-20260629-lint-cleanup-broad-plus-chunked`
  on `cw-us-east-02a` passed after the lint-review dispatcher cleanup:
  `11 passed, 98 deselected, 1 warning`.
- H100 module-boundary training-step integration:
  `/dlwh/iris-run-test_grugformer_moe-20260629-module-training-step`
  on `cw-us-east-02a` passed:
  `1 passed, 110 deselected, 1 warning in 63.41s`.
- H100 benchmark artifact smoke with `--fail-on-error`:
  `/dlwh/iris-run-bench_grug_moe_pallas_mgpu-20260629-default-capacity-smoke`
  passed on 8 H100s and emitted ok rows for `permute_metadata` and
  `permute_values`, using the default `capacity_factor=1.25`.
- H100 one-node full-trainer target-shape smoke:
  `/dlwh/grug-moe-pallas-mgpu-20step-current-20260629-150755` completed
  20/20 steps from commit `14fd25a73`, saved
  `/tmp/grug-scale-ckpt/grug-moe-pallas-mgpu-20step-current-20260629-150755/step-20`,
  and reported final MFU `20.19%`, mean MFU `20.11%`, and about
  `553k` tokens/s.

Performance evidence:
- Best observed default-path target forward row: `pallas_mgpu ~= 0.03853s`
  versus comparable `ragged_all_to_all ~= 0.08204s` at EP=8, T/rank=32768,
  D=2560, I=1280, E_local=32, K=4, bf16.
- Best validated opt-in chunked `permute_up` stage row:
  `/dlwh/bench_grug_moe_pallas_mgpu-20260629-052330-copytile-nearby-sweep`,
  group32/copytile512/copy_rows=1, `steady_state_time=0.0092931247s`,
  `231.08 TFLOP/s/rank`, `23.37%` nominal H100 bf16 roofline/rank. Exact split
  compare `/dlwh/bench_grug_moe_pallas_mgpu-20260629-052800-copyrows1-copytile512-splitcompare`
  passed with `matches_baseline=true`, `max_abs_diff=0`, `mean_abs_diff=0`, and
  no dropped routes. This remains an opt-in benchmark path, not the public
  default.
- Current gated target public forward+backward row:
  `/dlwh/iris-run-bench_grug_moe_pallas_mgpu-20260629-target-fwd-bwd-lint-cleanup`,
  `steady_state_time=0.06938801701956739s`, `139.27 TFLOP/s/rank`,
  `14.08%` nominal H100 bf16 roofline/rank, no dropped routes/error.
- Current target saved-residual backward diagnostic:
  `/dlwh/iris-run-bench_grug_moe_pallas_mgpu-20260629-target-saved-bwd-breakdown`,
  `saved_backward_pipeline=0.035684608699133s`, with component rows
  `combine_bwd=0.010570675988371173s`, `w2_bwd=0.005046184717987974s`,
  `w13_bwd=0.015592893973613778s`, and
  `dx_unpermute_vector=0.004997802975897987s`.
- Capacity sensitivity note: at the target shape with uniform routing,
  `capacity_factor=1.125` had zero dropped routes across four H100 seeds in
  `/dlwh/iris-run-bench_grug_moe_pallas_mgpu-20260629-bwd-capacity1125-multiseed`.
  The balanced-routing target public fwd+bwd row
  `/dlwh/iris-run-bench_grug_moe_pallas_mgpu-20260629-target-fwd-bwd-cap1125-balanced`
  also had zero dropped routes and ran in `0.06520126666873693s`. This does not
  change the first-PR default of `1.25`.
- Full-run scale note: a first 32-node 20-step run
  `/dlwh/grug-moe-pallas-mgpu-20step-scale-fe788a6e5-20260629-134609` was
  externally killed before training. A later larger-shape B-tiled R=N scale
  sweep hit pre-step GPU OOM before train metrics. These do not change the
  one-node full-trainer evidence or the kernel benchmark evidence above.

Limitations:
- Hopper/H100 only.
- EP must be local, single-node/NVLink, and `EP <= 8`.
- No NIC, InfiniBand, NVSHMEM, NCCL expert-parallel path, or multi-host EP.
- bfloat16 activations and expert weights; `combine_weights` may be bfloat16 or
  float32.
- top_k=4 is the target shape. Other K values are only supported where covered
  by tests and benchmark rows.
- No FP8 support.
- No duplicate-destination optimization for multiple top-k routes sent to the
  same destination rank.
- No remote atomics for combine; route slots are combined locally in fixed order.
- Capacity overflow uses deterministic receiver-side clipping before kernel
  execution; there is no in-kernel dynamic overflow handling.
- Backward is correctness-covered and benchmarked, but still below the final
  roofline target and remains an optimization area.
- Autotune-on-miss is not implemented; the checked-in tuning support is a static
  config lookup for the current H100 bf16 single-node bucket.

Cost estimate:
- `pallas_mgpu.py` uses `mgpu.kernel(...)` launch wrappers, not checked-in
  `pl.pallas_call(...)` sites. The installed Mosaic MGPU wrapper forwards
  unknown `mgpu.kernel(...)` kwargs into `mgpu.Mesh(...)`; `mgpu.Mesh` does not
  accept `cost_estimate=`, so this PR documents the reviewed status instead of
  adding fake estimates.

Links:
- Tracking issue: #6597
- Logbook: `.agents/logbooks/6597-moe-mgpu.md`
- 2026-06-28 logbook archive:
  `.agents/logbooks/6597-moe-mgpu-20260628.md`
```

## Reproducibility Commands

Focused local checks:

```bash
uv run --package marin-levanter --group test pytest \
  lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py -q

uv run --package marin-levanter --group test pytest \
  lib/levanter/tests/grug/test_grugformer_moe.py -q \
  -k 'pallas_mgpu_rejects or ordered_implementation'

uv run --package marin-levanter --group test pytest \
  lib/levanter/tests/grug/test_grugformer_moe.py -q
```

H100 correctness refresh:

```bash
uv run --package marin-iris --extra controller iris \
  --cluster=cw-us-east-02a job run --no-wait \
  --cpu 16 --memory 128GB --disk 16GB \
  --gpu H100x8 --reserve H100x8 \
  --enable-extra-resources --extra gpu -- \
  uv run --package marin-levanter --group test pytest \
    lib/levanter/tests/grug/test_grugformer_moe.py -q -n 0 \
    -k 'moe_mlp_pallas_mgpu_matches_ragged_a2a_on_hopper_when_available or moe_mlp_pallas_mgpu_grad_matches_ragged_a2a_on_hopper_when_available'
```

H100 benchmark artifact smoke:

```bash
uv run --package marin-iris --extra controller iris \
  --cluster=cw-us-east-02a job run --no-wait \
  --cpu 16 --memory 128GB --disk 16GB \
  --gpu H100x8 --reserve H100x8 \
  --enable-extra-resources --extra gpu -- \
  uv run --package marin-levanter --group test python \
    lib/levanter/scripts/bench/bench_grug_moe_pallas_mgpu.py \
    --tokens-per-rank 8 \
    --hidden-dim 128 \
    --intermediate-dim 128 \
    --experts-per-rank 2 \
    --topk 2 \
    --ep-size 8 \
    --routing balanced \
    --warmup 1 \
    --steps 1 \
    --implementations none \
    --include-pallas-stages \
    --pallas-stages permute_metadata permute_values \
    --jsonl /tmp/moe_mgpu_default_capacity_smoke.jsonl \
    --git-sha <git-sha> \
    --fail-on-error
```

Target-shape public full-step performance refresh:

```bash
uv run --package marin-iris --extra controller iris \
  --cluster=cw-us-east-02a job run --no-wait \
  --cpu 16 --memory 128GB --disk 16GB \
  --gpu H100x8 --reserve H100x8 \
  --enable-extra-resources --extra gpu -- \
  uv run --package marin-levanter --group test python \
    lib/levanter/scripts/bench/bench_grug_moe_pallas_mgpu.py \
    --tokens-per-rank 32768 \
    --hidden-dim 2560 \
    --intermediate-dim 1280 \
    --experts-per-rank 32 \
    --topk 4 \
    --ep-size 8 \
    --capacity-factor 1.25 \
    --routing balanced \
    --warmup 1 \
    --steps 3 \
    --implementations pallas_mgpu \
    --pass-mode forward_backward \
    --candidate-timeout-seconds 900 \
    --jsonl /tmp/moe_mgpu_target_fwd_bwd.jsonl \
    --git-sha <git-sha> \
    --fail-on-error
```
