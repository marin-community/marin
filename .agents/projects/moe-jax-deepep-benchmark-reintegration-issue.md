## Description

This experiment continues the sealed JAX DeepEP root-cause thread in `#3677`.

The previous session established two important facts:

1. the earlier negative JAX result in `#3665` was not measuring the real winning DeepEP transport path, because it only replaced layout metadata production and then still used JAX `ragged_all_to_all`;
2. a pure-JAX DeepEP transport path now runs end to end on H100x8, and under corrected world-size-8 defaults it lands within about `1.10x` to `1.54x` of the matched Torch transport baseline on the same fixed-shape workload.

That means the main unresolved question is now narrower:

> if the working pure-JAX DeepEP transport path is reinserted into the original fixed-shape JAX benchmark, does the old negative benchmark story materially change?

This thread is specifically about:

1. localizing the remaining small JAX/Torch transport delta on one authoritative cell;
2. reinserting the working pure-JAX DeepEP transport path into the original fixed-shape JAX benchmark;
3. determining whether the earlier negative benchmark conclusion was mostly a layout-only artifact or whether a broader benchmark bottleneck still dominates.

## Hypothesis or Goal

- Hypothesis: most of the original negative JAX benchmark result came from testing the wrong transport path rather than from an intrinsic JAX-vs-Torch DeepEP deficit.
- Goal: produce a small, decisive benchmark-level control that shows whether the corrected pure-JAX transport materially changes the original fixed-shape JAX ranking.
- Goal: attribute the remaining JAX/Torch transport delta before paying for broad sweeps or deeper tuning.

### Links

* Prior fixed-shape GPU issue (`#3633`): https://github.com/marin-community/marin/issues/3633
* Prior torch-side DeepEP / Hybrid-EP issue (`#3641`): https://github.com/marin-community/marin/issues/3641
* Prior layout-only JAX issue (`#3665`): https://github.com/marin-community/marin/issues/3665
* Prior Megatron scaling issue (`#3666`): https://github.com/marin-community/marin/issues/3666
* Prior root-cause issue (`#3677`): https://github.com/marin-community/marin/issues/3677
* Research branch: https://github.com/marin-community/marin/tree/research/moe-jax-deepep-benchmark-reintegration
* Research logbook: https://github.com/marin-community/marin/tree/research/moe-jax-deepep-benchmark-reintegration/.agents/logbooks/moe-jax-deepep-benchmark-reintegration.md

## Results

Current state as of 2026-03-16:
- new thread created from sealed `#3677` commit `6baa08edbd8ae9a782d0070a3b7cf0e1f38ba005`
- working branch head: `994456a064e5f4e34f628f2b2d60195f93e4a244`
- experiment issue: https://github.com/marin-community/marin/issues/3711
- baseline from `#3677`:
  - same-shape pure-JAX DeepEP transport now sits within `1.10x` to `1.54x` of Torch transport on H100x8
  - the benchmark-level reintegration question is still unanswered
- one-cell transport attribution on the authoritative `random, topk=2` full-shape cell is complete:
  - JAX: `layout_s=0.000205`, `dispatch_combine_cached_s=0.000700`, `step_s=0.000725`, `45.17M tokens/s`
  - Torch: `layout_s=0.000025`, `dispatch_combine_cached_s=0.000450`, `dispatch_combine_full_s=0.000491`, `66.77M tokens/s`
  - this means layout is not the dominant remaining JAX/Torch delta on the authoritative cell
- benchmark reintegration work landed locally:
  - new `deepep_transport` kernel added to `bench_moe_hillclimb.py`
  - current benchmark-lane fixes already made:
    - launcher switched to the newer working DeepEP ref
    - launcher installs `nodejs` / `npm` so Iris can regenerate protobufs during `uv sync`
    - launcher now uses the stronger DeepEP Python-module load mode instead of the weaker extension-only load mode
- the stronger benchmark-lane load path reached the real benchmark body, but full `deepep_transport` still failed at runtime with:
  - `jax.errors.JaxRuntimeError: INTERNAL: [0] There was an error before calling cuModuleGetFunction (704): cudaErrorPeerAccessAlreadyEnabled : peer access is already enabled`
- narrowing controls already completed inside the same hillclimb harness:
  - `shared_expert_dim=0` still fails with the same error
  - `deepep_transport_identity` succeeds
  - `deepep_transport_assignments_identity` succeeds
  - a first trustworthy consumed-intermediate probe, `deepep_transport_first_ragged_dot_probe`, fails again with the same error
- staged follow-up controls now completed:
  - splitting transport and local compute into separate compiled stages did **not** fix the failure by itself
  - prewarming `current` in the same pod did **not** fix the staged DeepEP path
  - exact prewarming of the staged local-compute executable before any DeepEP dispatch **did** unblock the staged forward path
  - authoritative successful staged result on the fixed-shape single-cell case:
    - `RESULT kernel=deepep_transport_staged ep=8 pass=forward time_s=0.016974 tokens_per_s=1930450.75`
- monolithic reintegration is now runnable too:
  - reusing the same exact local-compute prewarm before timing the original monolithic `deepep_transport` path unblocked a new `deepep_transport_prewarmed` kernel on the fixed-shape single-cell case
  - authoritative successful monolithic-prewarmed result:
    - `RESULT kernel=deepep_transport_prewarmed ep=8 pass=forward time_s=0.017159 tokens_per_s=1909665.39`
- first larger token step is now measured at the user-requested `32768/device` scale (`tokens=262144` global):
  - `random, topk=2`:
    - `current`: `0.027038 s`, `9695506.06 tok/s`
    - `deepep_transport_prewarmed`: `0.139256 s`, `1882465.68 tok/s`
    - ratio: `current / deepep_transport_prewarmed = 5.15x`
  - `runs, topk=2`:
    - `current`: `0.026870 s`, `9755985.37 tok/s`
    - `deepep_transport_prewarmed`: `0.146935 s`, `1784078.77 tok/s`
    - ratio: `current / deepep_transport_prewarmed = 5.47x`
- higher scale / higher top-k follow-ups hit memory limits before showing any compensating gain:
  - `tokens=262144, topk=8` fails during exact prewarm of the dummy local-compute buffer with:
    - `jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 8589934592 bytes.`
  - `tokens=524288, topk=2` lets `current` finish at:
    - `RESULT kernel=current ep=8 pass=forward time_s=0.059733 tokens_per_s=8777239.74`
  - but `deepep_transport_prewarmed` then fails during exact prewarm with:
    - `jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 68736253952 bytes.`
- full-mesh layout-only sweeps are now complete at the requested token scales:
  - `tokens=262144`, `524288`, and `1048576`
  - `distribution in {random, runs}`
  - `topk in {2, 8}`
  - across all twelve cells, the JAX wrapper is materially overprovisioned in a stable way:
    - `topk=2`: recv capacity is about `4.23x` over the actual per-rank receive count, while local assignment rows are about `7.98x` over the actual per-rank local assignments
    - `topk=8`: recv capacity is about `1.50x` over the actual per-rank receive count, while local assignment rows are still about `7.98x` over the actual per-rank local assignments
  - representative `32768/device` random cells:
    - `topk=2`: wrapper `recv_capacity=262144`, actual `max_recv_tokens=61836`, wrapper `assignment_rows=524288`, actual `max_local_assignments=65715`
    - `topk=8`: wrapper `recv_capacity=262144`, actual `max_recv_tokens=175129`, wrapper `assignment_rows=2097152`, actual `max_local_assignments=262690`
  - current best factual narrowing:
  - raw DeepEP dispatch/combine transport works in the hillclimb process
  - local assignment pack/collapse also works
  - the original monolithic reintegration path also becomes runnable if the exact local-compute executable is prewarmed before timing
  - making the benchmark path runnable did **not** make it competitive with `current` on the first larger token point
  - the current exact-prewarm implementation is already memory-limited at the next requested token / top-k points
  - the large-shape bottleneck is now localized more sharply: the hillclimb path is compiling the local expert stage against roughly `8x` more assignment rows per rank than the layout ever produces, and this factor stayed stable across `32768/device`, `2x`, and `4x`
  - exact-cap reintegration is now partially landed and changed the first large-token result materially:
    - new `deepep_transport_capped_prewarmed` kernel computes full-mesh exact caps, rounds them to a small multiple, and compiles the local expert path against those smaller shapes
    - on the first authoritative larger-token cell (`tokens=262144`, `random`, `topk=2`, `EP=8`):
      - `current`: `0.027047 s`, `9.69M tok/s`
      - `deepep_transport_prewarmed`: `0.144642 s`, `1.81M tok/s`
      - `deepep_transport_capped_prewarmed`: `0.022019 s`, `11.91M tok/s`
    - exact caps used there:
      - `max_recv_tokens=61952`
      - `max_local_assignments=65920`
      - `recv_factor=4.23x`
      - `assign_factor=7.95x`
    - the uncapped path still drags a long teardown-failure tail after it returns, so the next scale runs should compare `current` directly against the capped path only
  - the capped improvement is now replicated on both `random` and `runs`, and at `2x` token scale:
    - `tokens=262144`, `topk=2`:
      - `random`: `current=0.027047 s` vs `capped=0.022019 s` (`1.23x` faster)
      - `runs`: `current=0.027035 s` vs `capped=0.022958 s` (`1.18x` faster)
    - `tokens=524288`, `topk=2`:
      - `random`: `current=0.055837 s` vs `capped=0.045087 s` (`1.24x` faster)
      - `runs`: `current=0.056047 s` vs `capped=0.045484 s` (`1.21x` faster)
    - exact caps at `tokens=524288` were stable across both distributions:
      - `max_recv_tokens=123904`
      - `max_local_assignments=131712`
      - `recv_factor=4.23x`
      - `assign_factor=7.96x`
  - the topk-2 token sweep is now complete through `4x` and stays positive everywhere:
    - `tokens=1048576`, `topk=2`:
      - `random`: `current=0.123285 s` vs `capped=0.087961 s` (`1.40x` faster)
      - `runs`: `current=0.114987 s` vs `capped=0.091437 s` (`1.26x` faster)
    - exact caps at `tokens=1048576`:
      - `random`: `max_recv_tokens=247424`, `max_local_assignments=262784`, `recv_factor=4.24x`, `assign_factor=7.98x`
      - `runs`: `max_recv_tokens=246912`, `max_local_assignments=262528`, `recv_factor=4.25x`, `assign_factor=7.99x`
  - current best factual narrowing:
    - exact-cap reintegration has already flipped the benchmark story at the first two requested token scales
    - the observed gain is no longer a one-off random control; it is stable across `random` and `runs`
    - the operational drag is still mostly teardown hygiene after result emission, not failure to reach a valid timing
    - with the `4x` results in hand, the topk-2 story is now materially settled:
      - exact-cap reintegration beats `current` at `32768/device`, `2x`, and `4x`
      - speedups range from `1.18x` to `1.40x`
      - the remaining highest-value open question is whether the same fix unlocks the previously blocked `topk=8` regime
  - the previously blocked `topk=8` regime is now unlocked at `32768/device`, and the capped path wins there too:
    - `tokens=262144`, `topk=8`:
      - `random`: `current=0.091821 s` vs `capped=0.076401 s` (`1.20x` faster)
      - `runs`: `current=0.095239 s` vs `capped=0.076829 s` (`1.24x` faster)
    - exact caps:
      - `random`: `max_recv_tokens=175360`, `max_local_assignments=262912`, `recv_factor=1.49x`, `assign_factor=7.98x`
      - `runs`: `max_recv_tokens=175360`, `max_local_assignments=263040`, `recv_factor=1.49x`, `assign_factor=7.97x`
    - current best factual narrowing:
    - the positive benchmark-level story is no longer limited to `topk=2`
    - the same exact-cap fix both:
      - removes the old `topk=8` prewarm OOM at `32768/device`
      - and yields a `1.20x` to `1.24x` speedup over `current`
    - this is consistent with the earlier layout-only sweeps:
      - even when receive-cap overprovision is only modest, the local-assignment row inflation remains near `8x` and is still costly
  - the `topk=8` win now also holds at `2x` token scale:
    - `tokens=524288`, `topk=8`:
      - `random`: `current=0.193091 s` vs `capped=0.156848 s` (`1.23x` faster)
      - `runs`: `current=0.191187 s` vs `capped=0.158486 s` (`1.21x` faster)
    - exact caps:
      - `random`: `max_recv_tokens=350208`, `max_local_assignments=524928`, `recv_factor=1.50x`, `assign_factor=7.99x`
      - `runs`: `max_recv_tokens=350336`, `max_local_assignments=524928`, `recv_factor=1.50x`, `assign_factor=7.99x`
  - current best factual narrowing:
    - exact-cap reintegration now has stable wins across:
      - `topk=2` at `32768/device`, `2x`, and `4x`
      - `topk=8` at `32768/device` and `2x`
    - the pattern remains internally consistent:
      - receive-cap inflation matters at `topk=2`
      - but the more universal culprit is the nearly `8x` local-assignment row inflation, which persists across token scales and both `topk=2` and `topk=8`
  - the exact-cap reintegration is now also positive on the original fixed-shape `#3633` forward quadrant at `tokens=32768`, `EP=8`:
    - `random, topk=2`:
      - `current=0.003876 s`
      - `deepep_transport_capped_prewarmed=0.003325 s`
      - `1.17x` faster
    - `random, topk=8`:
      - `current=0.011067 s`
      - `deepep_transport_capped_prewarmed=0.009854 s`
      - `1.12x` faster
    - `runs, topk=2`:
      - `current=0.003867 s`
      - `deepep_transport_capped_prewarmed=0.003785 s`
      - `1.02x` faster
    - `runs, topk=8`:
      - `current=0.011085 s`
      - `deepep_transport_capped_prewarmed=0.010062 s`
      - `1.10x` faster
    - exact caps on that small-shape quadrant stayed in the same pattern as the larger-token sweeps:
      - `topk=2`: `max_recv_tokens` around `7.8k–8.1k`, `max_local_assignments` around `8.3k–8.4k`, `recv_factor` about `4.1x–4.2x`, `assign_factor` about `7.8x`
      - `topk=8`: `max_recv_tokens=22144`, `max_local_assignments=33152`, `recv_factor=1.48x`, `assign_factor=7.91x`
  - current best factual narrowing:
  - the exact-cap reintegration no longer needs larger token counts to look positive
  - it now beats `current` even on the original fixed-shape forward-only quadrant
  - the forward-only caveat still matters because the full `forward_backward` JAX transport column is not filled yet
  - the still-missing major comparison is the matched full-layer Torch baseline on the same benchmark question
- the same-shape full-layer Torch baseline is now captured too via the Megatron Qwen harness:
  - cases:
    - `marin_3633_topk_2`
    - `marin_3633_topk_8`
  - dispatchers:
    - `alltoall`
    - `deepep`
  - authoritative results:
    - `topk=2`:
      - `alltoall`: `forward_ms=14.271280`, `backward_ms=19.888771`, total step `34.160051 ms`, about `0.96M tok/s`
      - `deepep`: `forward_ms=4.997616`, `backward_ms=6.588206`, total step `11.585822 ms`, about `2.83M tok/s`
      - DeepEP speedup over all-to-all: `2.95x`
    - `topk=8`:
      - `alltoall`: `forward_ms=12.211341`, `backward_ms=22.626626`, total step `34.837966 ms`, about `0.94M tok/s`
      - `deepep`: `forward_ms=5.017590`, `backward_ms=6.007670`, total step `11.025261 ms`, about `2.97M tok/s`
      - DeepEP speedup over all-to-all: `3.16x`
  - scope caveat:
    - these are same-shape, full-layer H100x8 rows
    - but they are not distribution-controlled in the same way as the JAX hillclimb harness, because the Megatron benchmark uses its own router behavior instead of the explicit `{random, runs}` input generator
- current best factual narrowing:
  - the original fixed-shape benchmark story has materially changed on both sides:
    - JAX forward-only exact-cap reintegration is now positive on all four original cells
    - Torch full-layer DeepEP is strongly positive on the same shape
  - the main unresolved hole is now very specific:
    - the JAX exact-cap transport kernels currently only support `--bench-pass=forward`
    - the full JAX `forward_backward` column is still not implemented for the DeepEP transport path
- the remaining JAX `forward_backward` hole is now narrowed to a concrete AD blocker:
  - I added a minimal backward probe around the exact-cap path:
    - compute exact caps
    - prewarm the exact local compute executable
    - run `jax.value_and_grad(...)` over `_forward_deepep_transport_capped(...)`
  - authoritative control:
    - `current`, `random`, `topk=2`, `EP=8`, `forward_backward`
    - `time_s=0.010978`
    - `tokens_per_s=2.98M`
  - authoritative exact-cap failure:
    - `DEEPEP_EXACT_CAPS max_recv_tokens=7808 max_local_assignments=8320 recv_factor=4.196721 assign_factor=7.876923`
    - then:
      - `ValueError: The FFI call to levanter_deepep_dispatch_intranode cannot be differentiated. You can use jax.custom_jvp or jax.custom_jvp to add support.`
  - traceback location:
    - `loss_fn(...)`
    - `_forward_deepep_transport_capped(...)`
    - `_moe_mlp_deepep_transport(...)`
    - `_moe_mlp_ep_deepep_transport_local(...)`
    - `deepep_dispatch_intranode(...)`
    - `jax.ffi.ffi_call(...)`
    - JAX raises from `ffi_call_jvp`
- current best factual narrowing:
  - the missing JAX `forward_backward` column is no longer a vague performance gap
  - it is specifically blocked by the lack of JAX AD support for the DeepEP dispatch custom call
  - the fixed-shape scientific picture is now:
    - JAX exact-cap DeepEP is positive in forward mode on all four original cells
    - Torch DeepEP is strongly positive on the same shape in full forward+backward mode
    - JAX full forward+backward cannot currently be measured because autodiff stops at `levanter_deepep_dispatch_intranode`
