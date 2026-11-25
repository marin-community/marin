# Native Snowball inference and learner handoff

The inference stack on `codex/dlwh-levanter-rl` uses Snowball's paged prefill
and cached decode through a shared runtime. The Marin backend owns checkpoint
and mesh setup. The OpenAI and tensor APIs share request admission, stopping,
sampling, and policy installation. See [native inference](lib/levanter/docs/design/inference.md).
At runtime source `634a4baa9c`, the production observer-OFF tiny Snowball
workload passed two finite updates, exact checkpoint restoration and the original
five-second native-exit contract on one H100. The latest full-model attempt,
`2026.11.25.8`, completed one finite, changed update on all four native learner
processes, then failed the unchanged five-second post-ACK exit wait before
restore. A separately read 321-byte checkpoint marker records step 1 and policy
version 1; checkpoint arrays were not decoded. The shape-cache ownership fix did
not resolve full-model shutdown. Both Iris jobs failed, all five GPU Pods are
absent and all coordinator containers are terminal; no hardware remains active.
At later source `d4dcbb8cac`, an isolated eight-H100 diagnostic completed
device-to-host transfer of 201.7 GB across 4,411 independent buffers and exited
naturally. Shutdown request through process reap took 2.903 seconds under the
original five-second contract. This diagnostic did not reproduce the full
learner's failure; it omitted compiled training, checkpoint save and restore,
multihost coordination, and policy publication and generation.
Current-source full-model restore, update two, updated-policy publication and
generation, and the matched 25-update comparison remain open. The integrated
inference endpoints and benchmarks below have their own source and workload
boundaries. The experimental shutdown observer is disabled by default.

Checkpoint restoration exposed a separate ownership leak: the cached replica
reducer retained its mesh and backend client after cleanup. It now registers
with JAX's backend-cache cleanup. A fresh-process CPU regression checks an exact
replicated checkpoint round trip, including signed zero, a NaN payload and a
subnormal, then verifies that cleanup releases the mesh and client. The 82
checkpoint and TensorStore tests pass. This fix has no GPU checkpoint-restore qualification and
does not explain R8's first shutdown failure, which occurred before restoration.

Checkpoint-validation runs now write bounded shutdown-phase records to
`native-finalization/<pid>/shutdown-phases.jsonl` under the existing debug
artifact directory, falling back to the Iris output directory when no debug
directory is configured. Each record is closed before ACK; ACK, atexit and
watchdog events retain their existing stderr behavior and add no file writes.
The artifact is limited to 64 records and 32 KiB per native process. Missing
or incomplete phase evidence cannot qualify a diagnostic. This adds evidence
collection; the full-model shutdown failure remains unresolved.

| Gate | Integrated path |
| --- | --- |
| Paged inference | Snowball initializes KV pages and runs prefill and decode with GPU and TPU implementations. Tests cover mixed requests, compact short-window KV, and reuse of retained prompt pages. |
| Versioned installation | The runtime stages and checks candidate arrays before swapping model and version. Pause and version fences invalidate earlier admission and cache state. Both endpoints return the emitting policy identity and reject stale expectations. |
| OpenAI and binary serving | Both endpoints accept exact prompt IDs and expose generated IDs, finish reasons, policy version, and separate policy and behavior scores. Each can return the Snowball route payload. |
| Router replay | Capture uses the optimized forward and keeps route rows alongside retained prefixes. The learner can validate the emitted positions, align routes against padded rows, move them during packing, and consume them in differentiable policy scoring. |
| Fixed-input qualification | The benchmark compares learner scoring, prefill and cached decode with fixed weights and input tokens. Report numerical KL separately from the reference-KL penalty used in the RL objective. |

Policy scores are selected-token log probabilities from the model's raw logits.
Behavior scores, also called sampled logprobs here, use the temperature and
top-p transformed distribution. The public serving runtime returns zero
behavior log probability for greedy sampling and retains the raw policy score
separately.

The binary serving endpoint is `/v1/levanter/generate`: a versioned frame
containing JSON metadata and safetensors tensors. It is separate from the
learner's local pickle-protocol-5 IPC and the protobuf worker-pipe proofs.

Penalty-free training is supported by disabling both `use_kl_loss` and
`use_kl_in_reward`. The quick-loop bridge requires both off, supplies no
reference scores, and uses zero KL weights. A zero coefficient alone does
not remove the reference-actor requirement when a KL flag remains enabled.
Numerical zero KL between training and serving is a separate qualification;
the exact result below applies to the stated TPUv4 fixture.

## Benchmarks

The pinned AgentX workload uses the 67B-A2B v0.1 revision, 64K context, 128
offered clients, and a 180-second measurement. The published inference source
at `15bc5e751c` measured 575.55 versus 549.46 output tokens/s on v5p-8,
583.52 versus 554.00 on v6e-8, and 621.18 versus 590.24 on 8× H100. These
are the pre-integration native and compiled vLLM baselines. The integrated
replay-on/off overhead and vLLM gap have their own measurements.
[AgentX results and launch settings](docs/reports/snowball-inference-status-2026-11-20.md)

The TPUv4 results below qualify diagnostic fixtures separately from the 67B
AgentX workload. The [current fixture source](lib/levanter/benchmarks/snowball_zero_kl.py)
uses hidden width 128 and a 256-token context; the historical job's exact
configuration was not re-audited for this handoff. Full-67B training/serving
equality remains unqualified.

The four-chip TPUv4 arm with shared `fixed_schedule` attention matched all
evaluated FP32 logits between the learner, prefill, and cached decode. Max
logprob error and full-vocabulary forward KL were both zero. Within the native
Splash arm, the learner, prefill and cached-decode comparisons had maximum
absolute logit error 0.0004404 and maximum mean forward KL `7.55e-9` nats.

The production-engine replay A/B used the same requests and seeds at 32
decode rounds. Generated tokens, sampled logprobs, and finish reasons
matched bit-for-bit. Timing includes the serialized route payloads.

| TPUv4 fixture | Replay off (output tok/s) | Replay on (output tok/s) | Throughput change |
| --- | ---: | ---: | ---: |
| Homogeneous burst | 5,701 | 5,027 | −11.8% |
| Mixed requests with slot reuse | 948 | 739 | −22.1% |

[TPUv4 qualification](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-rl-v4-replay-qualification-20261122-v3)

After packing route IDs and thresholds into contiguous device buffers,
the repeat on the same TPUv4 worker reached 5,134
versus 5,709 output tok/s for the homogeneous burst, a 10.1% slowdown.
The mixed batch reached 754 versus 963 output tok/s, a 21.6% slowdown.
All compared outputs still matched bit-for-bit.
[Compact route buffer repeat](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-rl-v4-replay-compact-20261122)

Terminal capture records the final token's routes before releasing a finished
request's KV pages. A separate terminal executable added a device dispatch
for short scanned requests. It left throughput 11.4% lower for the homogeneous
burst and 27.7% lower for mixed requests with capture enabled. Capturing terminal
rows inline on scanned models reached 5,189 versus 5,739 output tokens/s
(−9.6%) on the homogeneous burst and 761 versus 964 (−21.1%) on mixed
requests. Tokens, scores, and finish reasons still matched exactly.
Unrolled models share one terminal executable across prefill and decode
buckets. With the shared executable, the reported first cold v5p-8 prefill
and extraction took 61 s, compared with 133 s in the earlier trial.
[Separate terminal capture](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-rl-v4-replay-terminal-20261122)
· [Inline terminal capture](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-rl-v4-replay-scan-inline-20261122)

The initial full-model replay arm on v5p-8 generated 400.97 output tokens/s.
That snapshot buffered text until completion for requests with raw policy
scores, so its first-token latency includes generation time.
[Paired v5p-8 AgentX runs](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-rl-agentx-v5p8-c128-replay-pair-20261122)

The follow-up runs use the shared terminal kernel and incremental chat
streaming. They retain the same context, offered concurrency, checkpoint,
and frozen prompts. Both native arms request raw policy scores. The
compiled vLLM result is the previous same-platform AgentX baseline, using
its standard response fields.

| v5p-8 run | Output tok/s | P90 first token (s) |
| --- | ---: | ---: |
| Compiled vLLM baseline | 549.46 | 1.339 |
| Native, replay off | 570.53 | 1.340 |
| Native replay, original archive and framing | 442.62 | 2.153 |
| Native replay, original archive and byte-oriented framing | 463.50 | 1.577 |
| Native compact replay, two prefill slots | 522.65 | 1.782 |
| Native compact replay, four prefill slots | 518.90 | 1.619 |
| Native compact replay, eight prefill slots | 522.62 | 1.687 |

Replay with the original archive and framing lowered throughput by 22.4%
against the native control and 19.4% against compiled vLLM. Both jobs reported
zero profiling errors.
The replay arm cancelled one outstanding request at the drain deadline.
Its median HTTP response was 33.4 MiB, compared with 15.3 KiB for the control.
[Optimized replay](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-rl-agentx-v5p8-c128-optimized-replay-20261122)
· [Replay-off control](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-rl-agentx-v5p8-c128-optimized-control-20261122)

The streaming adapter now writes a route event as JSON bytes and sends the
SSE framing separately. A synthetic 46 MiB route event took 56 ms to format,
compared with 118 ms for the old formatter, and produced identical bytes.
The normal OpenAI stream and tensor endpoint still agree on route identity
and policy scores.
[Byte-oriented replay run](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-rl-agentx-v5p8-c128-byte-stream-replay-20261122)

Route archive version 2 uses one byte per expert ID when all IDs in the
layer fit. The decoder widens IDs before validating the unchanged evidence
digest and still reads version 1 archives. A synthetic 64K capture with
the 67B model's route dimensions shrank from 34.63 MiB to 15.13 MiB before
base64 encoding. The fresh-process score and gradient tests and the OpenAI
stream and binary endpoint checks passed.

The four-slot compact AgentX run's median HTTP response dropped to 14.5 MiB. The
median interval from final text to HTTP completion fell from 420 to 179 ms.
All 366 measured requests completed with zero errors. Compact replay was
5.6% behind compiled vLLM on output throughput and 9.0% behind native with
replay off. Its p90 first token was 21.0% above compiled vLLM, 13 ms above
the 20% limit. Two and eight prefill slots each added fewer than four
output tokens/s. Their p90 first-token latencies increased by 162 ms and
68 ms, respectively. All compact runs completed their measured requests
with zero errors. Use four prefill slots for this workload. It had the
lowest measured p90 among the compact runs while keeping output throughput
within 10% of the native replay-off run.
The four-slot recommendation is the best measured configuration; it still
misses the p90 target of at most 20% above compiled vLLM.
[Four prefill slots](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-rl-agentx-v5p8-c128-compact-wire-replay-20261122)
· [Eight prefill slots](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-rl-agentx-v5p8-c128-eight-prefill-replay-20261122)
· [Two prefill slots](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-rl-agentx-v5p8-c128-two-prefill-replay-20261122)

On CoreWeave, a four-GB200 smoke exercised native generation,
including scores and terminal routes. Both full AgentX runs
used the existing regional 67B checkpoint mirror at 64K and concurrency 128,
with route capture off. Full-run GPU counts remain unverified here.
Native reported 571.43 output tokens/s versus 654.80 for compiled vLLM,
a 12.7% throughput gap. Native p90 first-token
latency was 0.665 s versus 1.047 s for compiled vLLM.
[GB200 smoke](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-native-cw-gb200-smoke-20261122-v3)
· [Native AgentX](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-native-cw-gb200-c128-agentx-20261122)
· [Compiled vLLM AgentX](https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-vllm-cw-gb200-c128-agentx-20261122)

## Distributed rollout and publication

Current distributed publication copies JAX shards to host bytes, stages them
as Torch GPU tensors, and uses the existing SkyRL expert-block/NCCL receivers.
Native `reload` stages and atomically installs weights within one serving
runtime. Distributed learner-to-native transport and coordinated replica
installation/admission still need wiring and qualification, including transfer
latency and staging-memory measurements.

The SkyRL quick-loop recipe uses its vLLM engine group and expert-block
NCCL publication. Run `2026.11.22.3` at `baa10702ae` completed one finite
native learner update on 32 H100s. The initial publication verified
19,687,358,464 bytes per receiver on eight inference H100s and returned
generation stamp 1 for weight version 0. The run committed step-1/version-1
checkpoint metadata and reached Python shutdown. It failed the validator's
five-second natural exit deadline.

Commit `768ea25030` bounds stage-transform caches to 64 entries per publisher
and releases them with their owner. Fresh-process CPU tests verified release
after normal and aborted publication, and kept exact publication bytes.
Run `2026.11.22.4` tested the repair from `768ea25030` on 40 H100s. After
one coordinator preemption, retried child `2026.11.22.4-a7ebaaff0409` passed
initial and step-one finite-state checks on all four learners. Model and Adam
fingerprints changed, and step-1/version-1 checkpoint metadata was committed.
The learners completed all seven cleanup phases, acknowledged shutdown and
entered Python `atexit`, but again exceeded the five-second natural-exit limit.
The driver failed at 18:04 PST; the cache fix did not resolve GPU exit.
All five current GPU pods and the five interrupted-attempt GPU pods are gone.
The accepted bundles match all 4,590 frozen source entries and 49 required
runtime files; all 670 materialized configuration fields match the frozen recipe.
Progress from the interrupted attempt is kept separate. Fresh-process restore,
step two, updated-policy publication and generation remain unverified.
This run covers source `768ea25030`; it does not qualify the newer native
inference integration now on this branch.

A separate two-layer MoE diagnostic at `8502f3c7f9` completed on one host
with eight H100s. Checkpoint state was restored bit-for-bit. The uninterrupted
learner and the learner resumed in a fresh process both exited naturally.
After the same frozen next batch, their model/Adam state and policy scores
differed, while loss and reported scalar metrics matched. Maximum model and
Adam differences were `3.51e-6` and `4.58e-5`. Two calls to one compiled gradient
object used identical state, batch, PRNG key and projection plan; two of 46
gradient leaves differed, and all were finite. This fixture does not qualify
full-model restart.
[Current-source repeatability](https://iris.oa.dev/#/job/%2Fdlwh%2Fnative-lifecycle-repeatability-2026-11-23-1)

A follow-up at `e7ffcd3ac1` compared two calls to one compiled gradient
object in each of two fresh processes on eight H100s. Both processes used
the original saved state, batch and PRNG key. The baseline differed in 134
embedding elements and one value-projection element, with maximum absolute
differences of `0.001953125` and `5.96e-8`. With
`--xla_gpu_deterministic_ops=true`, all 46 gradient leaves matched byte-for-byte
between the two calls. All four saved gradient sets were finite and independently
decoded. The deterministic compiler profile changed scatter lowering; both
executables retained the same opaque CuTe call configuration. This bounded
result does not establish determinism inside the native attention kernel or
qualify optimizer updates and resumed continuation. The earlier strict paired
failure remains unresolved.
[Compiler-profile diagnostic](https://iris.oa.dev/#/job/%2Fdlwh%2Fnative-raw-repeat-determinism-2026-11-23-1)

The next eight-H100 run at `94f1d6b171` used the deterministic compiler profile
with the unchanged warm-process versus fresh-process checkpoint oracle.
It passed all 18 strict checks. An independent audit decoded all 411 checkpoint
arrays and found them finite. After the same frozen second batch, the two
branches matched byte-for-byte in model, Adam, RNG, step and all 32 policy
scores. Model averaging was disabled. Both native processes exited naturally
with code zero under the unchanged five-second post-ACK deadline. The complete
archive and both task and uploader exits were verified. This qualifies the
instrumented two-layer MoE fixture under that explicit compiler profile;
the baseline-profile failures and full-model continuation remain unqualified.
The genuine Ray lifecycle results below have their own compiler-profile and
fixture boundaries.
[Strict paired checkpoint qualification](https://iris.oa.dev/#/job/%2Fdlwh%2Fnative-deterministic-paired-qualification-2026-11-23-1)

A single-H100 test of the FA4/CuTe attention backward fallback, with head
dimension 64 and multi-head attention, reproduced NaN query gradients on finite
inputs after a deliberately NaN output-gradient call. Clearing K padding alone failed. A broader patch loads
Q, K, V and output gradients through zero-initialized registers before writing
shared tiles. In a same-GPU comparison, the original kernel reproduced the
failure; all four finite-input calls with the patch returned finite gradients
that matched the original healthy call bit-for-bit. This branch includes the
four-load patch and a saved-input regression. At `5ae2bf475f`, that regression
and nine attention shape cases passed on one H100. The baseline again returned
NaN query gradients for finite inputs. These checks do not establish the cause
of the learner update differences or the full-model exit failure.
[Four-load comparison](https://iris.oa.dev/#/job/%2Fdlwh%2Fnative-fallback-masked-load-ab-2026-11-23-1)
· [Permanent GPU regression](https://iris.oa.dev/#/job/%2Fdlwh%2Fnative-fa4-permanent-regression-ab-2026-11-23-2)

Linux CPU controls captured stacks in a deliberately delayed C destructor,
using the unchanged learner protocol and five-second post-ACK exit limit.
One-H100 controls then executed a tiny CUDA kernel before shutdown and captured
C-destructor, CUDA-driver and XLA CUDA-plugin frames. Healthy shutdown and
hung-debugger cleanup controls passed. These are isolated process fixtures;
at that point, no actual GPU learner shutdown stack had been captured.
[CPU shutdown observer](https://iris.oa.dev/#/job/%2Fdlwh%2Fpolicy-native-stack-protocol-controls-20261123-2)
· [GPU-loaded observer controls](https://iris.oa.dev/#/job/%2Fdlwh%2Fpolicy-native-stack-protocol-gpu-controls-20261123-1)

The opt-in production shutdown observer remains experimental.
An actual worker filesystem probe found that inherited group permissions made
a fresh private directory mode `02700`. Setting only that new directory to
`0700` allowed all strict directory checks to pass. The follow-up CPU suite at
`f5e53f48b8` passed its observer-disabled, healthy and delayed-C-destructor
cases, then exceeded the unchanged five-second exit limit in a Python finalizer
case. A separate observer-disabled run also exceeded that limit. The observer
is therefore unnecessary for the newly measured timeout; these runs do not
establish the cause of the earlier GPU learner hang. The debugger also exceeded
its 128 KiB retained-output cap. This branch raises that cap to 1 MiB. The
CPU-only follow-up hit the guardian deadline before shutdown acknowledgment
and retained no stack, so actual retention beyond the old limit remains
unqualified. The original five-second exit check also failed. Outputs were
backed up before cancellation, and the pod was removed. Neither deadline was
relaxed. That run retained no actual GPU learner shutdown stack.
[Trace-cap diagnostic](https://iris.oa.dev/#/job/%2Fdlwh%2Fproduction-shutdown-cap-python35-239f1d-20261123-1)

The guardian now waits separately for debugger readiness and shutdown
acknowledgment. A late readiness marker can no longer revive an expired phase.
The 15-second outer debugger cap, ACK-relative sampling and cleanup deadlines,
and native five-second exit requirement are unchanged. Sixteen portable tests
pass, including arrivals and reads crossing a deadline. Two Linux controls at
`114e278e74` exercised the repaired path. The Python-finalizer control retained
all 226,380 bytes of a stack and module listing, with safe cleanup and all nine
phase checks passing. It had no qualifying C-destructor/XLA frame. The separate
C-destructor control captured `late_cleanup` and passed its expected
five-second timeout/exit-9 checks, but its trace was only 34,820 bytes. Both
strict expanded-retention verdicts remain false; these complementary results
do not qualify a real GPU learner shutdown stack.
[Python-finalizer control](https://iris.oa.dev/#/job/%2Fdlwh%2Fproduction-shutdown-ready-python35-114e278e-20261123-1)
· [C-destructor control](https://iris.oa.dev/#/job/%2Fdlwh%2Fproduction-shutdown-native-c6-114e278e-20261123-1)

After observer retirement, the worker now attempts a bounded upload of its
selected diagnostic files before propagating a shutdown failure. The route
uses the physical task and attempt identity; a commit receipt records uploaded
file hashes separately from observation success. Upload errors preserve the
original learner exception. A one-H100 run at `f846f6eae4` now qualifies this
integration on the genuine Ray `HostWorker` path. Its actual Linux controls
verified process-group cleanup for hung and exited uploader leaders with forked
descendants. The learner completed two finite updates with changed model and
Adam state. An independent audit decoded all 137 step-one checkpoint arrays
and verified exact restoration; all four restored FP32 scores matched with zero
absolute error and zero ULP difference. Both native processes exited naturally,
and final public close took 2.369 s under the unchanged post-ACK exit limit.

The complete archive, all 4,642 source entries and 13 delivered assets, task and
uploader exit zero, and Iris `SUCCEEDED` state were verified. A separate fresh
object-store read matched the commit and all eight selected diagnostic files
against the archive by bytes and hashes; object revisions stayed stable during
the read. The original
learner exited before sampling, so its empty stack is **not** a native C++ stack
capture. This pass qualifies healthy single-host lifecycle and diagnostic
delivery; full-model shutdown, multihost continuation, updated publication and
generation, and the 25-update comparison remain open. Step-two full state and
raw final scores were not serialized; its finite/change checks use producer
fingerprints and native child receipts. Model averaging was absent.
[Observer and durable delivery](https://iris.oa.dev/#/job/%2Fdlwh%2Ftiny-real-ray-observer-handoff-20261124-1)

A one-H100 smoke at source `f5e53f48b8` exercised the genuine Ray
`HostWorker` and public checkpoint roundtrip with the observer disabled. Its
receipts record one finite update, a natural exit from the original native
process, a new native process, and identical saved/restored model, Adam, RNG,
step and model-averaging fingerprints. It then failed the strict restored-score
check before update two. The initial driver did not persist the restored scores.
A follow-up with saved operands confirmed that all four finite FP32 scores
differed despite identical dtype, shape, recorded state and forward inputs.
The maximum absolute difference was `0.0001192093`, or 500 FP32 ULPs.
Independent decoding confirmed the differences and the numerical-failure
classification. Bounded forensic backups retained the score operands and
receipts but omitted three large model/checkpoint files. Both failed jobs were
cancelled and their pods removed. Neither run reached update two or final close;
the baseline-profile Ray lifecycle remains unqualified.
[Ray checkpoint smoke](https://iris.oa.dev/#/job/%2Fdlwh%2Ftiny-real-ray-hostworker-default-off-20261123-1)
· [Ray score operands](https://iris.oa.dev/#/job/%2Fdlwh%2Ftiny-real-ray-hostworker-default-off-20261123-2)

A follow-up on one H100 at the same source, `f5e53f48b8`, passed
the bounded genuine Ray `HostWorker` lifecycle with the observer disabled and
`--xla_gpu_deterministic_ops=true`. Both updates were finite and changed the
model and Adam fingerprints. An independent audit decoded all 137 arrays in
the saved step-one checkpoint, found them finite, and verified matching
saved/restored state fingerprints. All four restored FP32 scores matched
byte-for-byte: maximum absolute error and ULP difference were both zero.
The original and restored native processes exited naturally with
code zero. Public final close took 3.187 s under the unchanged five-second
post-ACK exit deadline. The complete archive, task and uploader exits, and
Iris `SUCCEEDED` state were independently verified.

Step-two full state and raw final scores were not serialized; its finite
state and changed model/Adam claims rely on producer fingerprints, native
child receipts and the pinned control flow. This one-host lifecycle pass is
separate from the eight-H100 paired optimizer comparison above. It does not
explain the baseline-profile failures or qualify full-67B or multihost
continuation, updated-policy publication, generation, observer-enabled
integration, or 25-update acceptance.
[Deterministic Ray lifecycle](https://iris.oa.dev/#/job/%2Fdlwh%2Ftiny-real-ray-hostworker-deterministic-20261123-2)

The quick-loop CLI now exposes `--deterministic-gpu-ops`. Each native worker
configures that compiler profile directly, including replacement processes;
runtime identities record actual raw and normalized flags and selected
platforms. Conflicting inherited compiler settings fail before process startup.
The default retains the existing environment behavior. The separate
`--max-retries-preemption 0` option bounds the coordinator, GPU gang and nested
materialized jobs; leaving it unset preserves their existing retry defaults.
Local tests cover these paths and exact native checkpoint continuation.

Full-model run `2026.11.23.1` tested source `ab3e704c55` on 40 H100s: 32 for
the native learner and eight for the existing vLLM rollout group. It used the
deterministic compiler profile with the observer disabled. Actual accepted
requests were BATCH with zero failure and preemption retries. The coordinator
and GPU bundles matched all 4,639 frozen source entries. The resolved recipe
had no unexpected differences across all 693 leaves of the resolved launch configuration; selection,
initializer, RLOO-N, graders, rollout engines and evaluation were preserved.

Durable initial and step-one proofs cover every native rank. Model and Adam
fingerprints changed, all state components were finite, and their schemas and
layouts were retained. The checkpoint writer reported 749.66 GiB; the fetched
321-byte metadata object records temporary step 1 / policy version 1. Its array
payloads were not downloaded or independently decoded. The original exception
shows shutdown acknowledgment followed by the unchanged five-second natural
exit timeout. The four original worker logs record all seven cleanup phases
returning, acknowledgment and entry into Python finalization. Restore never
started. The deterministic profile therefore did
not fix the full-model shutdown failure.

The job is terminal with one failure and no preemptions. All five GPU pods were
verified absent; the failed coordinator CPU pod was retained at that check. Canonical
output archives were unavailable: root and inference-head attempts advertised
`EMPTY`, and the four coscheduling-failed native hosts advertised no archive.
The evidence above comes from bounded, revision-checked durable debug objects
and the original exceptions. No real learner C++ shutdown stack was captured.
This result motivated the subsequent observer diagnostics. The later
observer-ON tiny run failed observer retirement; production runs keep it
disabled until that failure is resolved. Successful instrumentation alone
would not establish a shutdown fix.
[Full-model deterministic run](https://iris.oa.dev/#/job/%2Fdlwh%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.23.1-coordinator-e253e06c)

Observed run `2026.11.24.1` at `082025c4d8` scheduled the same 40-H100
allocation and reached Ray startup. The training entry point then stopped in
configuration validation because `WANDB_API_KEY` was missing. The isolated
launch wrapper had omitted that credential, which the unchanged `wandb` logger
requires. No learner update, checkpoint continuation or native shutdown sample
was recorded. Both jobs failed with zero retries and preemptions, and all five
GPU pods were verified absent. The next launcher must deliver the existing
logging key explicitly and check required credentials before compute. The
one-H100 lifecycle and delivery qualification above remains bound to its own
source and fixture.
[Observed full-model startup](https://iris.oa.dev/#/job/%2Fdlwh%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.24.1-coordinator-227e2e97)

Corrected diagnostic `2026.11.24.2`, executing source `40723ba125`, completed
one finite update on all four native learner ranks, each using eight GPUs.
Model and Adam fingerprints changed,
with schemas and layouts preserved. The 40-H100 run used 32 learner GPUs and
eight for the existing vLLM group, BATCH priority and zero automatic retries.
Explicit delivery of the W&B key restored normal logging. The coordinator and
GPU source bundles matched all 4,640 frozen files. Compared with the original
`2026.11.23.1` run, all 693 payload leaves matched after applying the 17 reviewed
changes to attempt paths, diagnostic observation and the two-hour job execution cap.

The checkpoint writer completed at 05:27 UTC on November 24. The fetched
321-byte metadata records step 1, policy version 1 and the same Snowball SFT
initializer. The array payload, reported as 749.66 GiB, was not independently
decoded. The run then failed
at the unchanged five-second natural process-exit limit, before restore.
The reported failure was native rank 2 on physical task 4. The observer watched
native rank 0 on physical task 3 and hit its own deadline after the native
shutdown acknowledgment. Its helpers
were safely retired, but it retained no qualifying C++ stack. The automatic
attach display's single `ioctl` frame does not identify the failed rank's exit
cause. The underlying exit cause remains unknown. The coordinator and GPU jobs
each failed once, with no preemptions; all five GPU pods were verified absent.
Iris task-output archives were unavailable. Independently fetched per-rank
state proofs and error tracebacks support these results. This failure motivated
the streamed, all-rank observer tested below. Restore, update
two, updated-policy publication and generation, and the matched 25-update
Megatron-versus-Levanter comparison remain unverified.
[Corrected observed diagnostic](https://iris.oa.dev/#/job/%2Fdlwh%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.24.2-coordinator-8a56842f)
· [W&B run](https://wandb.ai/marin-community/marin-snowball-curriculum/runs/02o09b9k)

The observer at `1524e77170` now streams numeric PCs before symbol lookup,
keeps partial traces and phase receipts, and observes every opted-in native
rank. Its zero-GPU Linux control captured the real main-thread `late_cleanup`
frame, but failed complete coverage: 33,373 bytes, 439 complete frames and 55
completed thread stacks out of 386 observed threads. Capture stopped near ACK + 2.75 seconds, reserving 0.20 seconds for detach
before the guardian deadline at ACK + 2.95 seconds. The guardian
safely retired the debugger, and the strict control verdict remains failed.
The failed control deliberately held its allocation for output collection.
The bounded outputs were backed up before cancellation; Iris reports
`KILLED` and the exact pod was verified absent. This is progress
in retaining diagnostics, not qualification of capture capacity or learner exit.
The 439 frames use only 17 distinct PCs. Symbol memoization and fewer
per-thread journal writes were the next optimization candidates; the following
CPU control tested both together without separating their costs. No capture deadline or coverage gate is relaxed.
[Streamed native capture control](https://iris.oa.dev/#/job/%2Fdlwh%2Fpolicy-streamed-native-c6-20261124-1)

The `python_atexit_begin` marker proves entry into Python's ordinary atexit
callbacks. It runs before previously registered callbacks, so it does not
prove that every callback finished. The full-model exit cause still requires
a discriminating native stack; loaded libraries and an idle attach frame do
not identify the owner of the wait.

Control `policy-streamed-native-c6-20261124-2` at `8f520676ef` passed the strict
Linux capture gate. Its real frame-only trace contains 235,588 bytes, 3,087
frames and all 386 observed threads, including all 384 fixture threads and the
main thread's `late_cleanup` frame. Stop/backtrace/detach took 0.548 seconds;
all nine phase checks and the original cleanup and identity guards passed.
The injected six-second C finalizer still produced the expected five-second
protocol timeout and native exit -9. The outer control and original suite
exited zero, every tracked owned process exited, and all four Pod containers
terminated zero with no restarts. Iris succeeded and its complete output
archive was independently inspected. The Succeeded Pod was retained. This
qualifies CPU capture instrumentation; full-model native exit remains unproven.
Symbol caching and journal throttling were tested together, with no separate
causal comparison of their costs.
[Qualified streamed native capture](https://iris.oa.dev/#/job/%2Fdlwh%2Fpolicy-streamed-native-c6-20261124-2)

Full diagnostic `2026.11.24.3` at `8f520676ef` failed during checkpoint
roundtrip validation. The native learner acknowledged shutdown, then exceeded
the unchanged five-second natural process-exit limit before restore. All four
initial and step-one state proofs report finite model and optimizer values,
changed model and optimizer hashes, and preserved schemas and layouts.
The independently fetched 321-byte checkpoint metadata records step 1, policy
version 1 and the preserved initializer; array payloads were not decoded.
The run used 32 learner and eight existing vLLM H100s at BATCH priority,
with zero automatic retries. All 286 numeric configuration values match
`2026.11.24.2` exactly in type and value. The accepted bundles match all 4,640
frozen source files. The GPU bundle has two extra generated launcher entries:
`full_observed_coordinator.py` and its timestamped alias, with identical bytes.
The startup Pod profiles matched the requested resources and isolation.

The observer retained real post-acknowledgment stacks from all four native
ranks, sampled 2.001–2.004 seconds after acknowledgment. Every main-thread
trace contains `ioctl` followed by seven CUDA-driver frames. The eight-frame
limit ends inside the driver, so the caller responsible for the wait remains
unknown. Captures completed 47–67 thread stacks out of 430–448 observed
threads before the detach reserve; every capture remains incomplete and
unqualified. Complete CPU capture did not establish GPU coverage; the reason
for the coverage difference has not been isolated. Debuggers detached and
guardian helper cleanup passed. This result motivated deeper main-thread
capture before choosing a lifecycle repair.

The terminal failure was first observed at 09:43 UTC after a missed heartbeat;
durable debug objects were written around 08:42–08:43 UTC. Continuous monitoring
and exact failure time are not established. Iris output archives were
unavailable; 78 indexed debug files totaling 390,937 bytes were fetched and
checked against their receipt hashes. R3 has no active jobs. Restore,
update two, updated-policy publication and generation, and the matched
25-update comparison remain unverified.
[Full diagnostic R3](https://iris.oa.dev/#/job/%2Fdlwh%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.24.3-coordinator-9efc0292)

The follow-up observer implementation permits 64 main-thread frames and keeps
eight frames per supplementary thread. It records the main stack's end or
truncation explicitly and rejects incomplete capture. All original deadlines,
the one-MiB output limit, process identity and cleanup checks remain in force.
Local guardian and artifact-handoff tests passed 65 cases, with three Linux-only
skips. The Linux CPU control at `b2aa2d9ab8` passed with a 236,638-byte real
frame-only trace: 3,102 frames across all 386 observed threads. The complete
23-frame main stack retained the C destructor caller at frame 15, beyond the
old eight-frame limit. Stop/backtrace/detach took 0.343 seconds. All nine phase
checks passed, the deliberately delayed destructor produced the expected
five-second protocol timeout and native exit -9, and the outer control and
original suite exited zero. Every tracked owned process exited; all four Pod
containers terminated zero with no restarts. The Succeeded Pod was retained.
The canonical archive and independent terminal audit qualify CPU capture
instrumentation; full-model native exit remains unproven.
[Deep main-stack CPU control](https://iris.oa.dev/#/job/%2Fdlwh%2Fpolicy-native-deep-main-20261124-1)

Full-model diagnostic `2026.11.24.4` executed the frozen `b2aa2d9ab8` runtime;
the later `842978442a` report commit changed documentation only. All four
initial and step-one state proofs are finite, with changed model and Adam
hashes and preserved schemas and layouts. The fetched 321-byte checkpoint
metadata records step 1, policy version 1 and the original SFT initializer;
array payloads were not decoded. The trainer traceback identifies native rank 1
acknowledging shutdown, then exceeding the unchanged five-second natural-exit
limit before restore. Both Iris jobs failed once with no retries or preemptions.
All five GPU Pods were verified absent. The retained coordinator Pod is terminal
Failed: its task exited 1 and its three helper containers exited zero.

The four main stacks are complete, with 13, 44, 10 and 41 frames for ranks
0–3. Rank 1 was sampled in `cuMemUnmap`, and rank 3 in `cuMemRelease`, through
XLA VMM allocator and PJRT client destruction under `Py_FinalizeEx`. Ranks 0
and 2 were sampled in Python finalization. Background captures remain incomplete
and unqualified; debugger detach and helper cleanup passed. These samples
identify teardown paths, without proving a driver defect or a repair.
[Full diagnostic R4](https://iris.oa.dev/#/job/%2Fdlwh%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.24.4-coordinator-b9bab541)

The cleanup candidate at [72cfb8e4de](https://github.com/marin-community/marin/commit/72cfb8e4decc4e5c096cf175637f27bc2d5ffeb7)
calls Haliax's existing cache-clear API before clearing
JAX caches and releasing backend clients. Fresh-process CPU reproductions using
`hax.named_jit` and tiny Snowball initialization showed that the previous
cleanup retained the mesh and Python JAX client; clearing Haliax's caches
released both. The affected worker suite passed 61 cases, with three skipped
and two deselected; the ownership regression also passed with four CPU devices.
On one H100, `tiny-real-ray-haliax-cleanup-rno-20261124-3` succeeded with
real Ray workers, two finite updates, exact restoration of 137 model/Adam/RNG/
step arrays, identical frozen-score bytes, and natural native-process exits
within the original five-second limit. All Pod containers exited zero. Model
averaging was absent. This qualifies the tiny workload only.
[One-H100 cleanup qualification](https://iris.oa.dev/#/job/%2Fdlwh%2Ftiny-real-ray-haliax-cleanup-rno-20261124-3)

Full run `2026.11.24.5` used the same published source `21409c1ef4`, 32 learner
and eight vLLM H100s at BATCH priority, with the native shutdown-stack observer
disabled and no retries. Each native learner process owned eight GPUs; a fifth
GPU Pod hosted vLLM. All four processes emitted finite live-state proofs at
initialization and after the first update; model, Adam, RNG and step
fingerprints changed with preserved schemas and layouts. The fetched 321-byte
checkpoint metadata records step 1, version 1 and the source URI and identity of
the pinned Datakit-SFT initializer; arrays were not decoded. The trainer
traceback reports that actor index 3's native learner acknowledged shutdown,
then exceeded the original five-second exit wait before restore. Four Python
finalization traces contain no Python frames and do not
identify a native caller. Both Iris jobs failed without preemption. All five GPU
Pods were verified absent; the coordinator's task exited 1 and its three helpers
exited zero. The tiny pass therefore leaves full-model cleanup unqualified.
[Full run R5](https://iris.oa.dev/#/job/%2Fdlwh%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.24.5-coordinator-ad7e87d0)

A later CPU probe found that Haliax's separate shape-evaluation cache retained
the Trainer after cleanup. Registering that cache with the existing clear API
releases the Trainer. Saved-state fingerprints and frozen input bytes matched
exactly before and after the change in the four-device Snowball control. The
eight-CPU Haliax suite passed 427 tests, with 37 skipped. The previous cleanup
already released the CPU client and mesh, so this result does not identify the
GPU shutdown cause.

At runtime source `634a4baa9c`, the production observer-OFF one-H100 tiny
Snowball run passed all 24 independent terminal checks with the deterministic
compiler profile. Its genuine Ray `HostWorker` completed two finite updates;
model, Adam, RNG and step fingerprints changed with preserved schemas and
layouts. Independent decoding covered all 137 arrays in the saved step-one
checkpoint, totaling 333,140,240 bytes, and matched the restored state exactly.
All four frozen FP32 policy scores matched byte-for-byte, with zero maximum
absolute error and zero ULP difference. Both native processes exited naturally
with code zero under the unchanged five-second post-ACK limit. Final public
close took 2.931 seconds, the Ray owner retired, and all four Pod containers
exited zero without restarts. Model averaging was absent. Step-two state was
verified through runtime fingerprints, finite receipts and the frozen control
flow; a separate full step-two checkpoint was not serialized or decoded. This
qualifies the tiny lifecycle only, with the observer disabled.
[Current-source production-OFF qualification](https://iris.oa.dev/#/job/%2Fdlwh%2Ftiny-real-ray-shape-cache-off-20261124-1)

The observer-ON tiny run at the same source failed observer retirement at
`stop_requested` after its original native learner had already exited naturally
with code zero. It did not fail the learner's five-second native-exit wait.
The failed diagnostic and its durable forensic handoff remain separate from the
production-OFF pass; observer qualification was not waived.
[Current-source observer-ON failure](https://iris.oa.dev/#/job/%2Fdlwh%2Ftiny-real-ray-shape-cache-20261124-1)

Full-model attempt `2026.11.24.6` used `634a4baa9c` with the observer disabled,
requesting 32 learner and eight vLLM H100s at BATCH priority, a two-hour execution
cap and zero automatic retries. All five GPU Pods remained Pending with no node
assignment behind Kueue admission and topology gates. A fresh check at
23:52:23 UTC confirmed that none had started; the coordinator was cancelled at
the reviewed 900-second queue guard. Both Iris jobs became `KILLED`, and the
coordinator and five GPU Pod names were verified absent at 23:55 UTC. Pod absence
does not establish exit code zero. No full-model learner update, restore or
shutdown result was produced. Full-model restore, update two, updated-policy
publication and generation at current source, and the matched 25-update
comparison remain unqualified. R6 produced no scientific full-model result;
the subsequent R8 result is recorded below.
[Full attempt R6](https://iris.oa.dev/#/job/%2Fdlwh%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.24.6-coordinator-7ee9eb5d)

Full-model attempt `2026.11.25.7` at the same runtime source failed in the
CPU coordinator before any GPU child was created. The regional adapter replaced
`build_config` after `expected_fingerprint` had been pinned. The production
`ArtifactStep.lower()` guard raised `FingerprintMismatchError` before step
execution; configuration rendering and SDK/Pod checks had not exercised this
hard-pin check. The independent audit reproduced the mismatch through the actual
production lowering API and distinguished it from a learner or GPU shutdown
failure. Iris reported one failure and no preemptions. All four coordinator
containers terminated: task exit 1, three helper exits zero. R7 has no active
job or GPU allocation. The subsequent R8 launcher repair and terminal
full-model result are recorded below; R7 provides no
learner or GPU shutdown verdict. Full-model continuation and the matched
25-update comparison remain open.
[Full attempt R7 CPU bootstrap failure](https://iris.oa.dev/#/job/%2Fdlwh%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.25.7-coordinator-3e5503f2)

Full-model attempt `2026.11.25.8` was submitted on November 25 at 01:16 UTC
with the same `634a4baa9c` runtime source. The launcher now pins the completed
regional configuration before calling production lowering. The actual lowering
check passes with that non-null pin and rejects a later learning-rate change;
the failed R7 assembly remains rejected. The coordinator created an actual GPU
child, discovered through its exact parent relationship. Both accepted bundles
match all 4,640 frozen source files; the GPU bundle adds only the reviewed flat
coordinator and its identical AtomicWriter alias. The coordinator and five GPU
Pod profiles match the pinned task image, resources, privileged containers,
private PID namespaces and explicit `02a` artifact prefix.

All five GPU Pods were running by 02:13 UTC on November 25. Both Iris jobs
were `FAILED` by 02:19 UTC, without preemption. Independent audit of the durable
initial and step-one receipts confirms finite state on all four native learner
processes. Model, Adam, RNG and step fingerprints changed, with schemas and
layouts preserved; model averaging was absent. Actual resolved configuration
retained all 286 scientific numeric types and values. The separately read
321-byte checkpoint commit marker records step 1, policy version 1 and the
original Datakit-SFT source identity. This marker proves neither checkpoint
array contents nor restoration; no checkpoint arrays were decoded.

During checkpoint validation, a native learner acknowledged shutdown and then
exceeded the unchanged five-second `subprocess.wait` deadline before restore.
The source shape-cache ownership fix therefore did not resolve full-model
shutdown. The production observer was disabled, and no current native C++ stack
identifies an owner or establishes the cause. All five exact GPU Pods were
verified absent; all four coordinator containers are terminal, with task exit 1
and three helper exits zero. No job or hardware remains active. Canonical
archive advertisements were empty or absent; no archive URI was guessed.

R8 is now the latest full-model scientific result. Full-model exact restore,
update two, updated-policy publication and subsequent generation, and the
matched 25-update comparison remain unqualified. The one-H100 production-OFF
pass retains its narrower scope. No native-exit deadline or numerical tolerance
has been relaxed.

The subsequent eight-H100 synthetic VMM teardown diagnostic, with canonical
pageable device-to-host fingerprinting, was submitted as
`native-large-vmm-exit-rno-20261125-1` and then cancelled without execution.
The original attempt never had an observed GPU node assignment. Iris is
`KILLED`, the exact Pod is absent and no hardware remains active; Pod absence
does not prove process exit zero. No probe output or allocator-teardown result
was produced, so R8 remains the latest full-model scientific result.

The reviewed manual 900-second unstarted-job guard was missed: cancellation was
requested 1,969.5 seconds after its deadline. The SDK's configured 900-second
scheduling timeout did not stop the Kueue-gated `BUILDING` task. These are
operational guard failures, not learner or allocator measurements. A later
independent deadline guard cancelled a second unstarted attempt within 2.743
seconds of its deadline. Subsequent CW02 attempts used admission checks covering
both Kueue reservations and actual Pod resource requests.
[Unexecuted VMM diagnostic](https://iris.oa.dev/#/job/%2Fdlwh%2Fnative-large-vmm-exit-rno-20261125-1)

The latest isolated diagnostic, `native-large-vmm-exit-cw02-20261125-4`, ran
source `d4dcbb8cac` on one host with eight H100s. Iris succeeded, the downloaded
output archive passed independent inspection, and all four Pod containers
terminated with exit code zero. Both large cases materialized 201,720,475,152
bytes in 4,411 independent buffers and completed one canonical pageable
device-to-host transfer. Both used the observer-OFF cleanup path. The diagnostic
case allowed a 60-second post-ACK wait; the production-protocol case retained
the original five-second wait. These names distinguish exit contracts, not
full-model acceptance. The diagnostic case transferred in 195.872 seconds and
exited naturally 1.165 seconds after ACK. The production case transferred in
195.302 seconds and passed the unchanged five-second exit contract; shutdown
request through process reap took 2.903 seconds. Its setup allowance was 360
seconds after the preceding attempt's 180-second setup cap expired during
transfer. The total harness and job caps remained 900 and 1,200 seconds.
Allocator statistics were unavailable. These measurements establish neither
physical memory residency nor fragmentation, and do not identify R8's cause.
The diagnostic exercised neither real model training nor TensorStore checkpoint
save or restore, multihost coordination, or Torch/NCCL publication and generation.
R8 remains the latest full-model result, and no hardware job is active.
[Completed isolated teardown diagnostic](https://iris.oa.dev/#/job/%2Fdlwh%2Fnative-large-vmm-exit-cw02-20261125-4)

[R8 coordinator](https://iris.oa.dev/#/job/%2Fdlwh%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.25.8-coordinator-8a5af5c9)
· [Actual R8 GPU child](https://iris.oa.dev/#/job/%2Fdlwh%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.25.8-coordinator-8a5af5c9%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.25.8-aa7e6e4857ca)

The native server stages and installs weights inside one runtime. The Levanter
learner bridge accepts a version-bound native route payload per trajectory and
preserves that binding through batching and policy scoring. The distributed
driver still needs to launch native serving replicas and attach their payloads
during trajectory assembly. Updated-policy publication, full-model restart,
and a matched 25-update run on the maintained branch remain open. The
quick-loop design records the separate protobuf worker-pipe proofs and the
planned H100 comparisons. The protobuf worker-pipe proof responses carry the
same replay payload used by the native HTTP endpoints.
[Quick-loop migration design](https://github.com/marin-community/marin/pull/11934)
· [Quick-loop run status](experiments/post_training/inventory/levanter-quick-loop-status.md)

