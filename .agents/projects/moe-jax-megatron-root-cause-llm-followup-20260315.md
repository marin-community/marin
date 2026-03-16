# Follow-up Addendum for External LLMs: JAX DeepEP Root-Cause Thread (`#3677`)

This document is a follow-up to the earlier long brief at:

- `/Users/romain/marin-wt/moe-jax-megatron-root-cause/.agents/projects/moe-jax-megatron-root-cause-llm-brief.md`

It is meant to be appended into all five prior LLM sessions, even though those five answers diverged and were written at different points in the investigation.

## How To Read This Addendum

Please assume:

- you may have answered earlier when the project was still blocked on raw build / symbol-resolution problems,
- or when the project was still blocked on same-process host-only transport hangs,
- or when the pure-JAX path had not yet succeeded end to end.

This addendum is meant to resynchronize you on the current state.

Please treat this addendum as **newer than your earlier answer**. If any of your earlier assumptions conflict with this document, prefer this document.

## Requested Answer Style

Please answer in a **detailed, structured, evidence-based** way.

Please:

1. Separate `source-backed findings` from `your inference`.
2. Include links for any public references you use.
3. Give confidence levels where appropriate.
4. Prioritize the most promising next debugging steps.
5. Explicitly say which hypotheses are now outdated given the new evidence below.

## What This Thread Is About

The experiment issue is:

- `#3677`: <https://github.com/marin-community/marin/issues/3677>

The broad question is still:

> Why did Torch / Megatron DeepEP experiments show strong gains on H100x8, while the JAX / Marin path did not, and what are the concrete remaining blockers to a pure-JAX zero-Torch-in-the-step-path integration?

The sealed earlier issues that matter are:

- `#3633`: fixed-shape JAX/Marin GPU benchmark that originally motivated this line of work
- `#3641`: Torch-side DeepEP / Hybrid-EP experiments on H100x8
- `#3665`: JAX layout-only DeepEP FFI experiment
- `#3666`: Megatron-LM / Qwen-like scaling experiments on H100x8

## The Most Important Update Since The Earlier Brief

The pure-JAX path is no longer just a bring-up attempt.

It now **does run end to end on H100x8 with zero Torch in the step path**.

That means several earlier hypotheses are now outdated:

- It is no longer correct to say “the pure-JAX path still cannot run.”
- It is no longer correct to say “the problem is still raw kernel-symbol resolution.”
- It is no longer correct to say “the remaining blocker is purely that DeepEP cannot be called from JAX.”

The remaining problem is now narrower:

- the pure-JAX path works and scales materially as `num_sms` increases from `2` to `8`,
- but it still underperforms Torch significantly,
- and it currently crashes when pushed to the intended `20`-SM DeepEP setting under one tested configuration.

So the project has moved from:

- “Can pure JAX call DeepEP transport at all?”

to:

- “Why does the pure-JAX path still lag Torch materially?”
- “Why does it scale through `num_sms=8` but fail at `num_sms=20`?”
- “Is the remaining issue in the wrapper/runtime/config, or is it a deeper limitation?”

## Executive Summary of the Current State

### What is already established

1. The old negative JAX result in `#3665` was mostly a benchmark-path mismatch.

   That experiment only replaced DeepEP’s dispatch-layout logic, then still used JAX `ragged_all_to_all` for actual transport. It did **not** exercise the DeepEP transport kernels that win in Torch/Megatron.

2. A naive JAX-to-Torch bridge is not a viable solution.

   The bridge cost is much larger than the transport kernel itself.

3. A real pure-JAX DeepEP transport path now exists and runs on H100x8.

   It is implemented as a JAX custom-call / FFI path with no Torch involved in the actual timed step path.

4. The pure-JAX path is not just a toy smoke anymore.

   It has succeeded on:

   - tiny deterministic H100x8 transport,
   - medium random H100x8 transport,
   - the full sealed `#3633` shape on H100x8.

5. The remaining gap is now performance/scaling/debugging, not “can it run at all?”

### What is newly interesting

The pure-JAX path was initially timed under a deliberately conservative debug configuration:

- `dispatch_num_sms=2`
- `combine_num_sms=2`

That produced about:

- `6.64M tokens/s`

The matching Torch DeepEP microbench on the same sealed shape produced about:

- `64.89M tokens/s`

That originally looked like a roughly `9.8x` gap.

But once I increased the JAX transport path to:

- `num_sms=4`

the JAX result improved to about:

- `12.51M tokens/s`

and at:

- `num_sms=8`

the JAX result improved again to about:

- `21.95M tokens/s`

So a large fraction of the apparent “JAX gap” was actually “I was still running a reduced debug channel-count configuration.”

That changes the current question substantially.

The new frontier is:

- why does JAX improve strongly from `2 -> 4 -> 8`,
- but still remain about `3x` behind Torch at `8`,
- and crash at `20` under one tested configuration?

## The Current Numbers That Matter Most

All of these are on H100x8 on the sealed `#3633`-style transport workload:

- shape: `tokens=32768 hidden=2048 experts=128 topk=2 distribution=random`
- execution focus: transport dispatch/combine, not full training

### Torch / Megatron-style transport baseline

Matching Torch-side transport microbench:

- `layout_s=0.000039`
- `dispatch_combine_cached_s=0.000472`
- `dispatch_combine_full_s=0.000505`
- `tokens_per_s=64886314.95`

Short version:

- Torch DeepEP transport baseline is about `64.89M tokens/s`

### Pure-JAX transport results so far

#### Conservative debug config

- `dispatch_num_sms=2`
- `combine_num_sms=2`
- large debug caps in that successful run:
  - `dispatch_num_max_send_tokens=8192`
  - `dispatch_num_max_recv_tokens=8192`
  - `combine_num_max_send_tokens=8192`
  - `combine_num_max_recv_tokens=8192`

Result:

- `step_s=0.004933`
- `tokens_per_s=6642731.73`

Rerun after gating verbose host-stage logging:

- `step_s=0.004859`
- `tokens_per_s=6743422.86`

Conclusion:

- removing logging helped only slightly,
- logging was **not** the main cause of the large gap.

#### Higher channel-count / SM-count results

At `num_sms=4`:

- `step_s=0.002619`
- `tokens_per_s=12509612.48`

At `num_sms=8`:

- `step_s=0.001493`
- `tokens_per_s=21946751.71`

At `num_sms=20` under the tested large-cap config:

- run failed with `EXIT_CODE=1`
- surfaced `CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered`

### Current interpretation of those numbers

The transport path is clearly not “flat JAX overhead.”

It responds strongly to more transport parallelism:

- `2 -> 4 -> 8` improves materially

So the remaining question is much more specific:

> What is wrong about the pure-JAX path when it is pushed toward the real DeepEP H100 channel-count regime?

## A Very Important Caveat About The `num_sms=20` Failure

The `num_sms=20` failure above was **not** tested under the exact same token-cap defaults that Torch uses.

That matters.

The Torch DeepEP benchmark code uses these world-size `8` defaults:

From:

- `lib/levanter/scripts/bench/bench_deepep_dispatch.py`

Dispatch config summary for `world_size=8`:

- `num_sms=20`
- `num_max_send_tokens=6`
- `num_max_recv_tokens=256`

Combine config summary for `world_size=8`:

- `num_sms=20`
- `num_max_send_tokens=4`
- `num_max_recv_tokens=256`

By contrast, the failing JAX `num_sms=20` run used the large debug caps:

- `dispatch_num_max_send_tokens=8192`
- `dispatch_num_max_recv_tokens=8192`
- `combine_num_max_send_tokens=8192`
- `combine_num_max_recv_tokens=8192`

So the `num_sms=20` crash is real, but the exact tested configuration was **not yet apples-to-apples with Torch’s intended defaults**.

One of the next concrete experiments I was about to run was:

> Re-test `num_sms=20` on the JAX side using the actual DeepEP-style world-size `8` defaults (`20/6/256` for dispatch, `20/4/256` for combine), instead of the oversized debug caps.

Please take that into account when judging the current evidence.

## What I Think Is Already Ruled Out

The following explanations are now weak or outdated.

### 1. “The routing metadata is wrong”

Earlier deterministic Torch-vs-JAX payload probes matched on the semantically important fields before transport launch:

- local `x`
- local `topk_idx`
- local `topk_weights`
- layout outputs:
  - `num_tokens_per_rank`
  - `num_tokens_per_expert`
  - `is_token_in_rank`

That does not prove every low-level handle is correct, but it does weaken the old “payload is just wrong” story.

### 2. “The raw build / symbol-resolution blocker is still the main issue”

That used to be true.

It is not the main issue anymore.

The earlier raw build problem was real:

- `named symbol not found`
- broken kernel attribute lookup

But the stronger extension-style build/load path got past that.

The pure-JAX path now executes real DeepEP transport on H100x8.

### 3. “The whole issue is just XLA custom-call weirdness”

This is also too broad now.

Earlier in the thread, there were same-process host-only controls showing that some problems survived outside `jax.jit`.

Now, more importantly, the real pure-JAX transport path actually works on H100x8 under some configs.

So the remaining problem is not that custom calls are impossible in principle.

### 4. “JAX is just generically too slow”

This is too vague to be useful now.

The newer data shows:

- the JAX path gets much faster as transport SM/channel count increases,
- so a large fraction of the early gap was configuration,
- and the remaining gap is now specifically about scaling to the intended H100 transport regime.

## What I Think Is Still Live

These are the highest-value still-live hypothesis classes.

### Hypothesis A: there is still a channel-count or buffer-shape bug in the JAX wrapper

This is very plausible because:

1. a previous real wrapper bug already existed here,
2. that bug was specifically about `num_sms` vs `num_channels`,
3. the current failure frontier also appears exactly when transport parallelism increases.

The previous real bug was in:

- `lib/levanter/src/levanter/kernels/deepep/csrc/deepep_transport_ffi.cu`

Inside `DispatchOnCurrentDevice(...)`, the call to:

- `deep_ep::intranode::notify_dispatch(...)`

originally passed the wrong quantity:

- `runtime.dispatch_config.num_sms`

where DeepEP expected:

- `runtime.dispatch_num_channels()`

That bug had to be fixed to get the real JAX path working.

This makes me suspicious that there may still be another latent high-channel-count mismatch elsewhere.

### Hypothesis B: the current JAX handle/output shapes are fine at low channel count but wrong or insufficient at higher channel count

Relevant code:

- `lib/levanter/src/levanter/kernels/deepep/transport_ffi.py`
- function: `deepep_dispatch_intranode(...)`

That Python wrapper currently sets:

- `max_recv_tokens = x_bf16.shape[0] * num_ranks`
- `num_channels = dispatch_config.num_sms // 2`

and allocates outputs shaped roughly as:

- `recv_x`: `(max_recv_tokens, hidden)`
- `recv_topk_idx`: `(max_recv_tokens, topk)`
- `recv_topk_weights`: `(max_recv_tokens, topk)`
- `recv_src_idx`: `(max_recv_tokens,)`
- `rank_prefix_matrix`: `(num_ranks, num_ranks)`
- `channel_prefix_matrix`: `(num_ranks, num_channels)`
- `recv_channel_prefix_matrix`: `(num_ranks, num_channels)`
- `send_head`: `(tokens_per_rank, num_ranks)`
- `local_expert_counts`: `(local_experts,)`
- `num_recv_tokens`: `(1,)`

I do not currently know whether any of those output/handle assumptions are subtly wrong once `num_channels` becomes large.

### Hypothesis C: the current JAX runtime/init model is still overly eager or otherwise mismatched at higher transport parallelism

Relevant code:

- `lib/levanter/src/levanter/kernels/deepep/transport_ffi.py`
- function: `deepep_dispatch_intranode(...)`

That function still calls:

- `ensure_intranode_runtime(...)`

before it performs the actual `jax.ffi.ffi_call(...)`.

So runtime creation still begins from Python-side control flow rather than being entirely owned inside the custom call itself.

This may not be the primary problem anymore, but it is still a structural difference from the cleanest possible callback-owned runtime model.

### Hypothesis D: the crash at `num_sms=20` is partly configuration misuse rather than a fundamental JAX-vs-Torch mismatch

Again, this is because:

- Torch world-size `8` defaults are `20/6/256` for dispatch and `20/4/256` for combine,
- while the crashing JAX run used `8192/8192` caps carried over from earlier debug-friendly settings.

So one live possibility is:

- the large-cap JAX config is an unrealistic / pressure-inducing configuration that Torch itself does not use,
- and the correct apples-to-apples comparison at `20` SMs has simply not been run yet.

### Hypothesis E: there is a real remaining high-SM / high-channel-count bug in the wrapper even after configuration is corrected

This remains very plausible too.

The strongest evidence:

- JAX scales through `2`, `4`, and `8`,
- but not yet through `20`,
- and there was already at least one real channel-count-related wrapper bug earlier.

So even if the large-cap `20` run was not the cleanest apples-to-apples control, it is still a meaningful warning sign.

## Code Pointers

The external LLM will not have local grep access, so here is a map of the most relevant code.

### 1. JAX-side Python wrapper

Path:

- `lib/levanter/src/levanter/kernels/deepep/transport_ffi.py`

Relevant responsibilities:

- build/load the DeepEP transport extension,
- expose JAX FFI entry points,
- define default dispatch/combine configs,
- define output shapes for the dispatch and combine handles,
- call `ensure_intranode_runtime(...)`.

Important objects/functions:

- `IntranodeConfig`
- `_DEFAULT_DISPATCH_CONFIGS`
- `_DEFAULT_COMBINE_CONFIGS`
- `deepep_dispatch_intranode(...)`
- `deepep_combine_intranode(...)`

Important current defaults:

- for world size `8`, the JAX wrapper’s default config also says `num_sms=20`

Important suspicious areas:

- eager `ensure_intranode_runtime(...)`
- output tensor shapes for the dispatch handle
- `max_recv_tokens = x.shape[0] * num_ranks`
- `num_channels = num_sms // 2`

### 2. JAX-side CUDA/C++ transport wrapper

Path:

- `lib/levanter/src/levanter/kernels/deepep/csrc/deepep_transport_ffi.cu`

Relevant responsibilities:

- same-process runtime manager,
- peer-access/UVA runtime setup,
- dispatch/combine FFI entry points,
- host-only controls and launch-debug helpers.

Important objects/functions:

- `DeviceRuntime`
- `RuntimeManager`
- `DispatchOnCurrentDevice(...)`
- `WaitForRecvCounts(...)`
- `DispatchIntranode(...)`
- `CombineIntranode(...)`

Important already-fixed bug:

- `notify_dispatch(...)` now correctly receives `runtime.dispatch_num_channels()`
- it previously received `runtime.dispatch_config.num_sms`

Important suspicious areas:

- anything that depends on:
  - `dispatch_num_channels()`
  - `combine_num_channels()`
  - `num_max_send_tokens`
  - `num_max_recv_tokens`
- receive-count waiting / synchronization discipline
- interaction between runtime buffers and higher channel counts

### 3. JAX benchmark harness

Path:

- `lib/levanter/scripts/bench/bench_deepep_dispatch_jax.py`

Important functions:

- `_transport_local(...)`
- `_transport_step(...)`
- `_dispatch_config_from_args(...)`
- `_combine_config_from_args(...)`

Important facts:

- the real working JAX benchmark path is using `shard_map`, not `pmap`
- the CLI already supports overriding:
  - `dispatch_num_sms`
  - `dispatch_num_max_send_tokens`
  - `dispatch_num_max_recv_tokens`
  - `combine_num_sms`
  - `combine_num_max_send_tokens`
  - `combine_num_max_recv_tokens`

### 4. Torch-side transport benchmark

Path:

- `lib/levanter/scripts/bench/bench_deepep_dispatch.py`

Important functions:

- `_dispatch_config_summary(world_size)`
- `_combine_config_summary(world_size)`

Important world-size `8` defaults:

- dispatch: `(num_sms=20, num_max_send_tokens=6, num_max_recv_tokens=256)`
- combine: `(num_sms=20, num_max_send_tokens=4, num_max_recv_tokens=256)`

This file is the best local reference for what the Torch baseline actually considers “normal.”

### 5. CoreWeave launcher for the JAX transport experiment

Path:

- `.agents/scripts/deepep_jax_transport_krt.py`

Important facts:

- this is how I launch the pure-JAX transport benchmarks on H100x8
- it accepts the dispatch/combine override flags listed above
- it can run the real benchmark path, probe-only mode, and some earlier host/debug modes

## Timeline Of The Most Important Changes Since The Earlier Brief

### Phase 1: prove that pure JAX can work at all

This succeeded.

The pure-JAX path now runs end to end on H100x8 with zero Torch in the actual step path.

Examples already observed:

- tiny deterministic case:
  - `CHECK x_max_abs=0.000000e+00 topk_max_abs=0.000000e+00`
  - `RESULT step_s=0.000480 tokens_per_s=266519.71`
- medium random case:
  - `RESULT step_s=0.000552 tokens_per_s=1855976.92`
  - `EXIT_CODE=0`
- full sealed shape:
  - `RESULT step_s=0.005023 tokens_per_s=6523372.00`
  - `EXIT_CODE=0`

### Phase 2: understand the seeming “hang”

The earlier apparent hang turned out to be misleading.

The pure-JAX transport step was often succeeding, then the process emitted a flood of late teardown errors such as:

- `DeepEP timeout check failed: rank = ...`
- `CUDA_ERROR_LAUNCH_FAILED: unspecified launch failure`
- XLA/JAX stream/event/module-unload shutdown errors

But the process still exited `0` on successful benchmark runs.

So the earlier “still hangs” description is outdated for the working configs.

### Phase 3: compare real pure-JAX transport against Torch

This exposed a large gap at the initial debug setting:

- JAX `num_sms=2`: about `6.64M tok/s`
- Torch: about `64.89M tok/s`

### Phase 4: realize the JAX comparison was still using a reduced debug transport parallelism

This was the big new insight.

At higher `num_sms`:

- JAX `num_sms=4`: about `12.51M tok/s`
- JAX `num_sms=8`: about `21.95M tok/s`

So the current problem is not well described as “JAX is 10x slower.”

It is better described as:

- JAX was initially run in a low-parallelism debug mode,
- increasing transport parallelism helps a lot,
- but there is still a large gap and a new crash frontier at the intended higher-SM regime.

## What I Most Want Advice On Now

### Question 1

Given the new evidence, what do you think is the most likely explanation for:

- strong improvement from `num_sms=2 -> 4 -> 8`,
- but failure at `num_sms=20` under the tested large-cap config?

Please be specific.

Do you think this now looks more like:

- a configuration mismatch,
- a channel-count handle-shape bug,
- a runtime buffer sizing bug,
- a deeper same-process runtime assumption,
- or something else?

### Question 2

How much weight should I put on the fact that the `num_sms=20` crash was tested with oversized debug caps rather than the actual Torch world-size `8` defaults?

Concretely:

- Should the very next experiment be a clean apples-to-apples `num_sms=20` rerun using `20/6/256` dispatch and `20/4/256` combine?
- Or do you think the existing evidence is already strong enough that I should inspect code first before spending more GPU time?

### Question 3

Given the current code structure, where would you look first for a hidden high-channel-count bug?

Please rank the most suspicious areas among:

- JAX-side output/handle shapes in `transport_ffi.py`
- `DispatchOnCurrentDevice(...)` / `DispatchIntranode(...)` in `deepep_transport_ffi.cu`
- same-process runtime manager state
- send/recv token-cap semantics
- teardown/lifecycle interactions

### Question 4

Do you think the remaining ~`3x` gap at `num_sms=8` is more likely due to:

- still-too-low transport parallelism relative to Torch’s intended `20`,
- JAX/XLA overhead around the custom call,
- wrapper/runtime inefficiency,
- or some remaining apples-to-oranges mismatch in what the JAX vs Torch benchmarks are timing?

### Question 5

What exact minimal experiment matrix would you run next?

For example, do you recommend something like:

1. `num_sms=20` with real Torch defaults
2. `num_sms=10`, `12`, `16`, `20` with matched caps
3. a smaller-shape sweep first
4. code inspection before more runs

Please prioritize.

## Commands That Represent The Current Frontier

### Working JAX run on the sealed shape

```bash
uv run .agents/scripts/deepep_jax_transport_krt.py \
  --config lib/iris/examples/coreweave-moe-jax-3677.yaml \
  --worktree /Users/romain/marin-wt/moe-jax-megatron-root-cause \
  --build-with-torch-extension \
  --load-as-python-module \
  --tokens 32768 \
  --hidden 2048 \
  --experts 128 \
  --topk-list 2 \
  --distributions random \
  --dispatch-num-sms 2 \
  --dispatch-num-max-send-tokens 8192 \
  --dispatch-num-max-recv-tokens 8192 \
  --combine-num-max-send-tokens 8192 \
  --combine-num-max-recv-tokens 8192 \
  --warmup 1 \
  --iters 3
```

Representative result family:

- `num_sms=2`: `6.64M tok/s`
- `num_sms=4`: `12.51M tok/s`
- `num_sms=8`: `21.95M tok/s`

### Matching Torch transport benchmark

```bash
uv run .agents/scripts/deepep_dispatch_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3677.yaml \
  --worktree /Users/romain/marin-wt/moe-jax-megatron-root-cause \
  --tokens 32768 \
  --hidden 2048 \
  --experts 128 \
  --topk-list 2 \
  --distributions random \
  --input-sources torch \
  --warmup 1 \
  --iters 3
```

Representative result:

- about `64.89M tok/s`

## Bottom Line

The most important correction to the earlier discussion is this:

- the pure-JAX DeepEP transport path is now real and working,
- the remaining problem is no longer basic bring-up,
- and a large fraction of the early performance gap was because I was still running a reduced debug transport configuration.

The current unresolved problem is much narrower and more actionable:

> Why does the JAX transport path scale nicely through `num_sms=8`, but still lag Torch materially and fail at `num_sms=20` under the tested config?

Please advise on the most likely root causes and the best next debugging sequence.
