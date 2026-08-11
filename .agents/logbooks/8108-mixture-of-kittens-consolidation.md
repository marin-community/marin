---
issue: https://github.com/marin-community/marin/issues/8108
title: Supported MoK-like Grug backend consolidation
---

# #8108 Mixture-of-Kittens consolidation logbook

This append-only logbook continues the experiment record in
`.agents/logbooks/8108-mixture-of-kittens.md` at commit `0f7069e55c`. The
original logbook remains the canonical record for the kernel-development and
four-GB200 training checkpoints; this continuation covers supported-package
and normal-Grug integration work.

### 2026-08-10 - Started supported `mok_like` consolidation

- Hypothesis: The proven #8108 native forward/backward can become a normal Grug
  backend without changing its numerical kernel, provided the integration owns
  both the routed and shared expert leaves at the block boundary.
- Baseline: Current main at `a5f0269edc`; #8108 substrate at `f0a8dcb3f` plus
  its final correctness fixes through `0f7069e55c`; upstream-MoK adapter API
  reference at `6dca6bd553`.
- Changes: Extracted the final CUDA adapter, deterministic padded schedule,
  fused custom VJP, saved forward context, FP32 macrobatch routed-weight
  accumulation, and production-width router allocation into
  `levanter.kernels.mixture_of_kittens`. Added explicit source/build config,
  read-only preflight status, and a caller-owned runtime handle. Added a
  separate block-level `mok_like` selector to the normal
  `experiments/grug/moe_hero_ep` model while retaining the routed-only
  `moe_implementation` as its reference backend.
- API contract: Inputs and canonical gradients are `x`, combine weights,
  routed gate/up/down, and one shared gate/up/down; selected expert indices
  have no cotangent. Packing and transposition remain inside the kernel API.
  The name `mok_like` identifies Marin's PyTorch-free reimplementation; `mok`
  remains reserved for the optional adapter to the upstream package.
- Local result: The schedule, validation, availability, runtime lifecycle, and
  normal-model contract suite passes 24 focused tests. Imports and compilation
  succeed without loading CUDA, invoking NVCC, or initializing workspaces.
- Caveat: The native workspace is fixed to one local-token shape, so the
  supported training entry point rejects changing batch schedules before CUDA
  initialization.
- Next action: Add the four-GB200 numerical and training launch gates against
  the supported package, then reproduce the all-gradient, multi-update, and
  25-step checkpoints without retry.

### 2026-08-10 - Supported package passed the four-GB200 correctness matrix

- Commands: Three isolated four-GB200 Iris gates through the public
  `levanter.kernels.mixture_of_kittens` API: balanced routing with named-context
  rematerialization; zero-token and 3:1-skewed routes; and hidden width 512 with
  production expert width 3,072.
- Result: Jobs
  `/dlwh/mok-like-supported-correctness-20260810-1110`,
  `/dlwh/mok-like-supported-routes-20260810-1118`, and
  `/dlwh/mok-like-supported-width3072-20260810-1123` succeeded without retry.
  Forward output, input gradient, combine/router gradient, routed gate/up/down,
  and shared gate/up/down matched ordinary Grug EP at the #8108 BF16 limits.
  The zero-token experts returned exactly zero routed weight gradients. The
  production-width gate retained the 3:1 route construction used by #8108 and
  had zero mismatches in every gradient group.
- Saved-context result: The rematerialized gate recorded four native forward
  and four native backward calls across four GPUs. The enclosing checkpoint did
  not replay forward communication.
- Interpretation: The supported API preserves the proven native numerical path,
  including the production-width router-partial allocation and FP32
  macrobatch routed-weight reduction, across balanced, empty, skewed, and
  multi-macrobatch schedules.
- Next action: Run the full-shape normal-Grug one-update, two-update, and
  25-step gates through the block-level `mok_like` selector.

### 2026-08-10 - Normal Grug update gates passed after runtime-tree fix

- Diagnostic: The first full-model attempt
  `/dlwh/mok-like-supported-update-coord-20260810-1127` reached optimizer
  update and failed because the runtime handle had been added to the model's
  static Equinox metadata after Muon initialized its parameter-shaped state.
  The parameter and optimizer pytrees therefore had different metadata. This
  was an integration lifecycle bug, not a native numerical failure.
- Fix: Initialize the explicit native runtime before model-state construction
  and bind the same handle before optimizer initialization. This keeps the
  canonical parameter leaves unchanged while giving parameters, gradients,
  EMA parameters, and parameter-shaped optimizer state one stable pytree
  definition. Added a focused regression that maps the bound parameter tree
  against Optax trace state.
- One-update result: Child
  `/dlwh/mok-like-supported-update-coord-20260810-1135/grug-train-mok-like-supported-update-20260810-1135`
  succeeded on four GB200s with no failure, retry, or preemption. The normal
  48-layer Grug model completed native forward/backward and one optimizer
  update with finite loss 11.8.
- Two-update result: Child
  `/dlwh/mok-like-supported-two-update-coord-20260810-1139/grug-train-mok-like-supported-two-update-20260810-1139`
  succeeded with no failure, retry, or preemption. Both consecutive optimizer
  updates completed with finite loss (11.8 at the final reported step).
- Local result: The focused kernel and normal-model suite passes 32 tests;
  changed-file lint/format checks pass; package-scoped Pyrefly reports zero
  errors.
- Next action: Complete the no-retry 25-step acceptance gate and record its
  terminal loss and profile result.

### 2026-08-10 - Supported normal-Grug 25-step gate passed

- Command: The normal `experiments/grug/moe_hero_ep` launcher ran the
  production 48-layer, hidden-width 6,144, expert-width 3,072 configuration
  with eight routed experts, top-4 routing, one shared expert, global batch 64,
  and the block-level `mok_like` selector on one four-GB200 Iris task. Iris and
  the child were configured with zero retries.
- Result: Child
  `/dlwh/mok-like-supported-25step-coord-20260810-1146/grug-train-mok-like-supported-25step-20260810-1146`
  succeeded on attempt zero with zero failures and zero preemptions. All 25
  optimizer updates completed; loss stayed finite and fell from 11.8 to 6.50.
  The final route-overflow count was zero.
- Throughput: The final step took 13.51 seconds at 22.69% MFU and 19,397
  tokens/s; W&B reported 21.32% mean MFU, 22.50% p50 MFU, and 22.70% p90 MFU.
  Performance was not tuned during this consolidation.
- Profile: Steps 5 through 10 uploaded successfully to
  `s3://marin-us-east-02a/tmp/ttl=30d/xprof/mok-like-supported-25step-20260810-1146/plugins/profile/steps-5-to-10`.
  The browser view is
  `https://iris.oa.dev/proxy/xprof/open?uri=s3%3A%2F%2Fmarin-us-east-02a%2Ftmp%2Fttl%3D30d%2Fxprof%2Fmok-like-supported-25step-20260810-1146`.
- W&B: `https://wandb.ai/marin-community/marin_moe/runs/mok-like-supported-25step-20260810-1146`.
- Interpretation: The supported backend reproduces #8108's finite training
  trajectory through the normal Grug model, while retaining the explicit name
  and claim: this is Marin's PyTorch-free reimplementation (`mok_like`), not
  the optional direct adapter to upstream MoK (`mok`).

### 2026-08-10 - Completion audit tightened the supported contract

- Audit fixes: Made the pinned-host limit and device-memory fraction explicit
  launch fields; extended runtime ownership across model initialization,
  checkpoint restore, evaluator construction, training, and setup failures;
  added CUDA 13/NVVM and clean-revision preflight checks; and made native builds
  locked and atomic. The normal model now fuses the first shared expert and
  evaluates any remaining shared experts through the ordinary Grug path.
- Correctness result: Four-GB200 job
  `/dlwh/mok-like-supported-current-correctness-20260810-1227` passed at expert
  width 3,072 with eight real macrobuffers and offloaded saved context. Forward,
  input, combine-weight, routed gate/up/down, and shared gate/up/down gradients
  passed the existing #8108 elementwise BF16 checks. The end-to-end router
  matrix gradient had 0.791% relative L2 error, below the existing 1% BF16
  relative tolerance. Native counters recorded four forward and four backward
  calls across four GPUs, proving no rematerialized forward replay in the
  offload policy.
- Scope finding: The native API accepts one shared gate/up/down triple. The
  production two-shared model remains semantically supported by evaluating the
  second expert through ordinary Grug, but that full shape does not fit current
  JAX lowering on four GB200s. The default, 0.85, and 0.99 allocator-fraction
  screens all failed before step one; the 0.99 screen exhausted a 182.46 GiB
  XLA pool. The acceptance launcher therefore keeps the supported singular
  fused-shared contract instead of implying the two-shared full shape passed.
- Corrections to earlier entries: The two-update gate's final progress loss was
  about 10.8, not 11.8. The final local selection below supersedes the earlier
  intermediate focused-test counts.

### 2026-08-10 - Current-tree acceptance gate passed

- Command: The normal `experiments/grug/moe_hero_ep` launcher selected
  `mok_like` with one fused shared expert, E8 top-4 routing, hidden width 6,144,
  expert width 3,072, 48 layers, global batch 64, explicit 0.85 device-memory
  fraction, explicit 192 GiB pinned-host limit, and zero retries.
- Result: Child
  `/dlwh/mok-like-supported-final-25step-coord-20260810-1305/grug-train-mok-like-supported-final-25step-20260810-1305`
  completed all 25 optimizer updates on attempt zero. Iris reported exit zero,
  zero failures, and zero preemptions. W&B recorded finite loss from
  11.8054094315 to 6.5060281754 and a zero final drop fraction.
- Throughput: The final step took 13.5156 seconds at 22.6841% MFU and 19,395.65
  tokens/s. W&B reported 21.7254% mean MFU, 22.7123% p50 MFU, and 23.0126% p90
  MFU. Performance tuning remained out of scope.
- Artifacts: W&B is
  `https://wandb.ai/marin-community/marin_moe/runs/mok-like-supported-final-25step-20260810-1305`.
  The successful steps 5-10 XProf capture is
  `https://iris.oa.dev/proxy/xprof/open?uri=s3%3A%2F%2Fmarin-us-east-02a%2Ftmp%2Fttl%3D30d%2Fxprof%2Fmok-like-supported-final-25step-20260810-1305`.
- Local result: The kernel, normal-model, and Grug variant-contract selection
  passes 49 tests with one expected skip. The full repository pre-commit suite,
  package-scoped Pyrefly, wheel build, and wheel-content inspection pass; the
  wheel contains the native CUDA adapter source.
- Identity: This result is the Marin-native, PyTorch-free `mok_like` backend.
  It is not the optional direct adapter to upstream MoK, whose reserved name is
  `mok`.
