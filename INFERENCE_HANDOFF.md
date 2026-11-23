# Native learner and inference integration

November 22, 2026, 17:09 PST snapshot. Branch:
[`codex/dlwh-levanter-rl`](https://github.com/marin-community/marin/tree/codex/dlwh-levanter-rl).
This branch contains the unified policy learner, pipeline backend, native
Snowball learner, inference integration, and the validation/diagnostic work from
this effort. The implementation inventory below is against frozen main
`fda7484bbfe957c89d08b7d58dbaac3396d2f4bc` and includes the cache-ownership repair
described below. Executed GPU evidence covers its explicitly pinned source.

Inference still runs in SkyRL's vLLM engines. The new work lets the JAX learner
publish its compute-policy weights into those engines and verifies which policy
produced returned tokens. This work has not added a native Snowball serving path
or a public binary inference API.

Levanter already has a paged inference engine and OpenAI server. The trainable
Snowball model currently does not implement the `initial_cache` interface used
by that engine, so native Snowball paged decode still needs implementation and
qualification. The active quick loop uses vLLM throughout. See the existing
[engine](lib/levanter/src/levanter/inference/engine.py),
[OpenAI server](lib/levanter/src/levanter/inference/openai.py), and
[Snowball model](lib/levanter/src/levanter/models/snowball.py).
The existing Levanter server exposes chat/completions and text completions;
its completion input schema accepts text rather than raw token IDs. Its SSE
adapter emits completed generations rather than incremental decoding events.
Those surfaces need separate compatibility and performance qualification.

## What is here

| Area | Added in this branch | Existing infrastructure retained | Remaining work |
| --- | --- | --- | --- |
| Weight transfer | A Snowball JAX publication plan and Torch parent sender map sharded learner tensors into the existing expert-block inventory. Dense transfers are sliced and individual expert matrices are bounded; no HF checkpoint is written for each update. | SkyRL's expert-block installation, NCCL transport, receiver placement and publication fencing. | Qualify updated full-model weights after restart, measure transfer latency and staging memory, then reduce host staging/serialization where measurements justify it. |
| Binary protocol | Length-prefixed pickle-protocol-5 frames carry NumPy arrays between the trusted local Torch parent and isolated JAX child. A dedicated response pipe keeps logs off the protocol. | Ray calls and the existing engine interfaces. | A binary/tensor inference request-response API, sharing semantics with the OpenAI adapter. The new learner IPC is a separate interface. |
| OpenAI protocol | Generation receipts on the existing `generate` and chat paths record returned-token policy stamps, including retry handling and bounded accounting. | SkyRL's OpenAI HTTP bridge, vLLM engines, HTTP routing and pause/resume machinery. | One backend contract exercised through both binary and OpenAI adapters, with exact token IDs, logprob semantics, policy identity and optional routing metadata preserved. |
| Router replay | No Levanter router-decision replay was added. Native configuration and batch validation reject it. | SkyRL/Megatron's existing replay facilities. | Carry captured MoE expert choices through rollout responses, batching, packing and the native model, and consume them in scoring and backward recomputation. |
| KL disabled | The native quick-loop bridge requires KL loss and reward KL disabled, uses no reference actor/scores, and supplies zero KL weights. | The generic policy objective can also consume fixed-reference scores when configured separately. | Training-versus-serving distribution parity is a separate acceptance gate; disabling the KL penalty does not establish it. |

The transfer path is JAX compute view → bounded host payload → Torch sender →
existing expert-block/NCCL receivers → vLLM installation. It does not yet provide
a direct JAX-to-vLLM GPU-buffer path. Receiver byte replay verifies installed
weights; it does not replay the MoE router's expert choices.

Publication stage-transform caches are now owned by each publisher, with a
64-entry limit per publisher. Previously, a module-global cache retained
shardings and a JAX backend client after the publisher was released. Fresh-process
CPU tests reproduce that retention and verify release with the owned cache,
including abort and multiple-publisher cases. Actual CPU publication checks still
match the expected weight bytes and reuse cached transforms across versions.
This repairs a demonstrated ownership gap. Its effect on full-model GPU process
exit remains unqualified; the failed GPU run below predates this repair.

Implementation:
[publication plan](lib/marin/src/marin/rl/levanter_publication.py),
[sender](vendor/skyrl/skyrl-train/skyrl_train/weight_sync/expert_block/levanter_sender.py),
[local learner protocol](lib/marin/src/marin/rl/levanter_worker_protocol.py),
[inference client and receipts](vendor/skyrl/skyrl-train/skyrl_train/inference_engines/inference_engine_client.py),
[native batch contract](vendor/skyrl/skyrl-train/skyrl_train/workers/levanter_batch.py),
[native configuration guards](vendor/skyrl/skyrl-train/skyrl_train/utils/utils.py).

## What we want from inference

For native Levanter serving, implement and qualify Snowball's cache, prefill and
decode interface, then its versioned weight-installation boundary. Reuse the
learner's model, precision and routing contracts where possible.

Use one inference backend contract with a binary/tensor adapter for the training
loop and an OpenAI-compatible adapter for agent and tool clients. Both should
preserve the same prompt/completion token identity, positions, sampling settings,
stop/truncation behavior, policy version and logprob definitions. Raw policy
scores and sampling-distribution behavior scores need distinct fields. An
adapter must expose unsupported capabilities explicitly.

Weight publication needs a complete, versioned transaction: freeze the learner
snapshot, transfer the declared tensors, verify all receivers, install the
version, then admit generation. Measure publication time and memory independently
of rollout time. A successful transfer, a resumed engine and a response from the
new policy are separate observations.

For MoE router replay, preserve token/layer/top-k alignment and padding through
both protocols and all batching transformations. Replay must work through the
native forward and recomputation paths before it can be an inference-parity
claim. The current Levanter bridge rejects `routed_experts` and
`moe_router_replay`; accepting those fields without consuming them would be
incorrect.

“Zero KL” has two meanings. Zero reference-KL penalty is already the native
quick-loop configuration: `use_kl_loss` and `use_kl_in_reward` are false. A zero coefficient alone
does not remove the existing reference-model requirement when a KL flag remains
enabled. Zero training/serving KL means that scoring and
inference implement the same policy distribution. That requires a shared
contract for tokens, routing, masks, positions, compute precision and sampling
transforms. Test identical weights and fixed token sequences through training
scoring, inference prefill and cached decode; report logprob differences and a
stated KL measurement. Exact weight bytes alone do not prove equal
distributions. Router replay can remove routing differences; other numerical
differences still need measurement.

Existing SkyRL importance-ratio diagnostics can compare selected-token learner
and rollout logprobs at zero staleness, including absolute log-ratio mean, maximum
and p99. These are sampled-token measurements, not full-vocabulary KL. The
`policy/rollout_train_prob_diff_mean` metric averages `exp(log_ratio)`, so its
perfect-match value is 1. See the existing
[forward metrics](vendor/skyrl/skyrl-train/skyrl_train/batch_metrics.py) and
[importance-ratio diagnostics](vendor/skyrl/skyrl-train/skyrl_train/utils/importance_ratio_diagnostics.py).
No training/serving parity result is claimed here.

## Evidence and open gates

The controlled follow-up `2026.11.22.4` is running from
`768ea250307c746f95464a81ab5944338c8ddc5c`, including the owned-cache repair.
Its coordinator is
`/dlwh/users-dlwh-checkpoints-snowball-inventory-2026.11.22.4-coordinator-4367d39b`.
At 17:09 PST all five GPU tasks were running with zero failures or preemptions.
Both accepted bundles match all 4,590 frozen entries and 49 required runtime
files. This is a two-update, 40-H100 BATCH diagnostic with the original numerical
settings and five-second exit requirement; startup does not qualify the repair.

The previous full-model diagnostic, `2026.11.22.3`, executed from
`baa10702aec4cfd62e0aebb2cef8c336974172ae`. Its accepted coordinator and GPU
bundles match all 4,588 frozen source entries. All 670 materialized configuration
fields match the frozen recipe after the two generated attempt identifiers and
source-derived staging paths are accounted for.

All four native learner processes initialized the
67,078,882,816-parameter model on 32 H100s with data4/expert8. Initial-state
and step-one finiteness candidates and joint passing verdicts were present for
every rank. Model and Adam fingerprints changed after the update, and
step-1/version-1 checkpoint metadata was committed.
All eight vLLM receivers reported zero byte mismatches for initial policy version
0, comparing 19,687,358,464 bytes each. Returned generation carried stamp 1;
the convention is weight version `v` → returned-token stamp `v + 1`.

The run failed at 16:30 PST during the checkpoint validator's five-second natural
process-exit check. Cleanup phases, including public backend release and
distributed shutdown, returned; the learners wrote shutdown acknowledgements and
entered Python `atexit`. This did not establish successful process exit or fresh
restoration. All five GPU pods from that failed run are gone.

This establishes a finite first full-model update, initial publication and
returned generation. Updated-policy publication, full-model restart/continuation, and the matched 25-update
Megatron/SkyRL-versus-native-Levanter comparison remain open. Tiny GPU checkpoints
restore exactly, but a live update and a fresh-process update from the same saved
state fail strict post-update state/score equality. Identical calls to one pure
compiled gradient program can also differ. The short-sequence MHA fixture with
64 values per attention head (D64) has a reproduced history-dependent NaN
failure; the full model uses a separate D128/GQA4 path. These results do not establish training/serving
zero KL, and no numerical tolerance has been relaxed.

The next gates are full-model finite continuation plus updated publication and
generation; a common two-protocol inference contract with router replay;
fixed-input training/serving numerical qualification; and measured quality and
performance for Megatron/SkyRL and native Levanter on the same quick-loop recipe.
Detailed execution history and current run receipts
are summarized in the [quick-loop status](experiments/post_training/inventory/levanter-quick-loop-status.md).
