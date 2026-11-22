# Levanter quick-loop status

November 21, 2026 (PST).

No completed full Snowball optimizer update is established. Updated-weight
publication, subsequent generation, exact native checkpoint continuation, and
the matched 25-update comparison remain unverified.

BATCH smoke `2026.11.21.13` is running from
`cb3fd4e7d767e52f0409ecb0b0fda5b98902e957`, configured to disable NumPy's
hugepage advisory in native children. At 23:17 PST all five GPU tasks were
running; both accepted bundles match all 4,583 frozen entries, including 43
required runtime files, and the executed configuration matches the preflight.
All five physical hosts changed, so improved loading would not isolate the
policy's effect from placement. Effective Linux policy and full-model
initialization, publication, updates, and resume remain pending.

Recovery BATCH smoke `2026.11.21.12` ran from
`a5534aa7aa72be3b1512bc3ad8283a3e8ee1d82b`. Its 43 required runtime files
are identical to `e9fa30e8a2230a4015d8488280d2b375b81ea07d`; only this
writeup changed. The scheduler reused all five hosts, and the same `g83c00a`
host stalled again. At 22:45 PST its container recorded 68.42%
memory-pressure PSI avg60, without memory-limit/OOM, swap, or CPU-throttling
events. Three of 12 bounded kernel samples caught its NumPy tensor-conversion
thread in `isolate_freepages_block → compaction_alloc`. This establishes
compaction involvement; the sample count is not a fraction of elapsed time.
The run was intentionally stopped at 22:49 PST rather than waiting for the
same initialization deadline. All five GPU pods and the coordinator pod are
gone. It did not qualify initialization or training.

A process-local NumPy hugepage-advisory opt-out is under validation. The native
environment pins NumPy 2.3.5, whose
[documented import-time flag](https://numpy.org/doc/2.3/reference/global_state.html#madvise-hugepage-on-linux)
`NUMPY_MADVISE_HUGEPAGE=0` suppresses NumPy's own hugepage hints. No node policy
is changed, and this does not disable all process hugepages or guarantee that
loading, publication, or training will succeed. Both accepted `.12` bundles
and its executed configuration matched the preflight.

The optional native policy defaults to `None`; the quick-loop candidate selects
`False` before its child interpreter starts. Exact NumPy 2.3.5 / JAX 0.11.0 CPU
tests passed fresh-process policy validation and exact continuation of model,
Adam, RNG, step, and averaging state under both policies. Required
checks and independent review passed. The macOS NumPy getter always returns
false, so effective Linux policy and improved loading remain runtime gates.

BATCH smoke `2026.11.21.11`, from
`e9fa30e8a2230a4015d8488280d2b375b81ea07d`, failed at the existing
3,600-second model-initialization deadline. Three learner hosts finished
reading the weights; one remained in HF import. A nonblocking Python stack
sample caught its fsspec event-loop thread converting tensor dtypes. The slow
container recorded a 66.66% one-minute average for memory-pressure stalls,
while using 29.30 GiB of its 1,800-GiB limit, with no limit/OOM events, swapping,
or CPU throttling. The underlying stall cause remains unproven.

This run did not qualify dynamic VMM, publication, updates, or resume. Both
accepted bundles match all 4,583 frozen entries, including 43 required runtime
files, and the executed configuration matches the preflight. The `.12` run
preserves the runtime source, recipe, allocation, and timeout. VMM retains the
same 0.90 budget with preallocation disabled. A fresh allocation alone does not
establish that the stall is resolved.

BATCH smoke `2026.11.21.10`, from
`42a248d50564f44d6010a35591d49d2f65c0ea38`, initialized all four full-model
learners with BFC preallocation and the same observed 0.90 limit on all 32
H100s. Initial publication then failed: a new JAX NCCL communicator could not
allocate 2 MiB of CUDA memory on any learner host. Receiver timeouts and actor
termination followed. No completed publication payload or train step is
established for this source. The subsequent `.11` smoke attempted dynamic VMM.

Diagnostic BATCH smoke `2026.11.21.9`, from
`a7c751443ffbc55ca5e8b3603d0e41c09af17d3d`, initialized all four full-model
H100 learners, byte-verified initial publication, and saved GPU compiler and
allocator evidence before its first update. That update failed requesting a
41.05-GiB temporary arena per device. Each allocator had enough aggregate headroom but
could provide no free block larger than 32 GiB. The evidence supports
fragmentation of the growing BFC pool as the immediate blocker. The subsequent
smokes test allocator remedies; no completed optimizer update is established.

BATCH smoke `2026.11.21.8`, from
`233030789c6afd6aa61d315a001ce65ac7436007`, initialized the full 67B Snowball
model on all four native learner hosts and byte-verified initial publication
to all eight vLLM receivers. The fixed XLA coordination checks completed on
every host. The first compiled update then failed at 18:04:31 PST, requesting
41.05 GiB in `jit__train_step`. The exact buffer cause remains unknown. Both
jobs are failed and all five GPU pods are gone.

The quick loop now reaches generation and the compiled learner update with
the original recipe, but no completed optimizer update is established.
Publication and generation after an update, exact checkpoint continuation,
and the matched 25-update comparison remain unverified. No comparison arm
has been submitted; full smoke acceptance is still required.

## Scope and fixed recipe

[The quick-loop CLI](train_quick_loop.py) retains SkyRL orchestration, its
RolloutEngine/vLLM generation, task graders, evaluation, and coordinator-computed
RLOO-Neutral (RLOO-N) advantages. `--learner levanter` replaces the learner with
one isolated JAX process per learner host. It consumes finalized SkyRL batches,
including failure masks and fractional loss weights. Full native RolloutEngine
orchestration migration remains outside this integration.

The instrumented Megatron control and native Levanter candidate preserve these
inputs and settings:

- Selection: `s3://marin-us-east-02a/marin/rl-data/task-selection/snowball-100k/2026.11.04`
  ([Filer](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Frl-data%2Ftask-selection%2Fsnowball-100k%2F2026.11.04)).
  Read-only preflight verified its published manifest, 100,000 training rows,
  and fixed 100-task evaluation panel: 20 each from calendar, competitive coding,
  reasoning gym, STEM, and structural tasks.
- Initializer: `open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21`, revision
  `fe666478454e79fe6027a0bab0388d9abd68f51a`, from
  `s3://marin-us-east-02a/models/open-athena--Grug-67B-A2B-Datakit-SFT-262K-2026.09.21`.
  Preflight matched all 39 weight-file LFS hashes and sizes to that Hub revision
  and verified tokenizer/config metadata; it did not download all weights locally.
- Synchronous RLOO-N: 128 prompt groups × four answers per update, zero staleness,
  infrastructure failures excluded, agent failures retained at zero reward,
  zero-reward groups filtered, and no batch advantage normalization.
- Base optimizer: AdamW, learning rate `1e-6`, betas `(0.9, 0.999)`, weight decay
  `0`, gradient clip `0.5`, constant schedule with zero warmup. One optimizer
  pass per batch, token-mean clipped PPO with symmetric `0.2` clipping; KL and
  entropy terms disabled.
- Allocation: four eight-H100 learner hosts and one eight-H100 inference host in
  `cw-us-east-02a`; the native mesh is data4/expert8. Automatic failure retries are zero.
- Rollouts: the same task/grader definitions and backends, temperature `1.0`,
  top-p `1.0`, 24,576-token request window, and 8,192-token response limit.
  Evaluation uses the fixed panel before and after training, seed `20261020`,
  temperature `0.6`, and top-p `0.95`. Reasoning-marker penalty remains `0.1`;
  overlong penalty remains zero.

Rendered recipe comparison against main
`fda7484bbfe957c89d08b7d58dbaac3396d2f4bc` found only `trainer.strategy`,
`trainer.batch_builder`, and `trainer.policy.levanter_config` differ between the
instrumented control and native candidate. Both are configured with publication
verification enabled. The native
profile keeps FA4 attention, ring-gather/combine MoE, FP32 routing and selective
attention operations, and BF16 projection logits with FP32 softmax. The platform
and storage fixes change no kernels, objective math, precision settings, or
numerical tolerances. See [policy learning](../../../docs/references/policy-learning.md#skyrl-quick-loop).

## Evidence and acceptance

| Gate | Observed result | Remaining proof |
| --- | --- | --- |
| Source and CPU | JAX/JAXLIB `0.11.0` data4/EP1 proof: exact state continuation, 78 publication payloads / 232,384 bytes; four initial and four restarted JAX children exited zero. Four-device Snowball regression: 35 passed, two skipped. | CPU checks do not qualify CUDA, H100, EP8, or the full online loop. |
| Actual H100 | Smoke `2026.11.21.10` reports four native process identities, H100 80GB devices, 67,078,882,816 installed parameters, data4/expert8, and BFC preallocation at the expected per-device limit. It fails on NCCL allocation during initial publication. The earlier `.9` source reached its first train step and failed a contiguous arena allocation. | Qualify memory management through publication and two real parameter updates with the pinned CUDA kernels. |
| Publication and resume | Earlier smoke `2026.11.21.9` passed initial model-version-zero byte comparisons on all eight receivers, with generation stamped 1. Smoke `.10` records no completed initial payload. CPU Gloo replay and four fresh-process native checkpoint transactions passed exact model, Adam, RNG, step, and model-averaging fingerprints. | Publication of actual updated weights with zero byte mismatches, nonempty generation stamped 2 after model version 1, terminal checkpoint, fresh learner processes restoring exact state, and a further update. |
| Matched 25-update comparison | Both arms are configured; neither comparison has run. | Same inputs, optimizer, rollout/grading and evaluation exposure; measure training reward and fixed-panel evaluation separately, plus groups/sequences/tokens, wall time, and environment/tool failures, with terminal receipts and archives. |

The earlier November 4 single-host, four-H100 toy-pipeline acceptance does not
qualify this current 32-GPU full-MoE learner source.

The first failed smoke was version `2026.11.20.1`, source
`646ccb48641442cf3d9b3223f06c1bca0c5a2e0a`, coordinator
`/dlwh/users-dlwh-checkpoints-snowball-inventory-2026.11.20.1-coordinator-96655281`,
with GPU child suffix `users-dlwh-checkpoints-snowball-inventory-2026.11.20.1-a02f6689f4d1`.
[W&B run](https://wandb.ai/marin-community/marin-snowball-curriculum/runs/uzjtcvcc).
It failed on November 20 at 12:28:12 PST; the actual child configuration confirmed
inherited `JAX_PLATFORMS=cpu`. No successful model update, publication, generation, or checkpoint
continuation receipt resulted. Previous GPU pods were confirmed gone.

The fixes in `fc9a60939b` explicitly select `JAX_PLATFORMS=cuda,cpu`, preserve Ray's
`CUDA_VISIBLE_DEVICES`, and require the actual GPU backend before loading
weights. Bootstrap checks CUDA in the isolated learner environment before Ray
starts. Both Megatron and Levanter now place recovery checkpoints, Ray logs,
and rendezvous state in lifecycle-managed 02a scratch storage beside compute,
overriding shared 08a or ambient temporary-storage defaults. See
[RL storage and retention](../../../docs/references/rl-launching.md#artifact-and-runtime-boundaries).

Relevant source pins:
[checkpoint validation, `646ccb4864`](https://github.com/marin-community/marin/commit/646ccb48641442cf3d9b3223f06c1bca0c5a2e0a),
[regional coordinator route, `32123524a1`](https://github.com/marin-community/marin/commit/32123524a13df6f03385f58114466c55672929b0),
and [GPU selection / 02a storage, `fc9a60939b`](https://github.com/marin-community/marin/commit/fc9a60939b4bc40b34b4dce7ffdf3e3d33f47d70).
Focused native, bootstrap, SkyRL, required lint/type checks, strict docs, and
independent source review passed for these fixes.

## Failed smokes and reproduction

Smoke `2026.11.20.5` used coordinator
`/dlwh/users-dlwh-checkpoints-snowball-inventory-2026.11.20.5-coordinator-2ea612c2`,
with final GPU child suffix `users-dlwh-checkpoints-snowball-inventory-2026.11.20.5-a8ae164919e7`.
Both jobs ran at BATCH priority from frozen source `6936528c55`. After six CPU
coordinator preemptions, the GPU gang ran for 6m49.7s and failed on November 21
at `2026-11-21T12:50:15.820Z` (04:50:15.820 PST):
`ValueError: DCN product 1 must equal num_slices 4.` The preserved traceback
locates the failure in native trainer mesh setup, before weights loaded.
No update, publication, generation, or checkpoint-continuation acceptance passed.
The [preserved exception record](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Ftmp%2Fttl%3D14d%2Fskyrl%2Fmarin%2Fusers%2Fdlwh%2Fcheckpoints%2Fsnowball-inventory%2F2026.11.20.5%2Fattempts%2Frendezvous%2Fdebug_artifacts%2Frank0-g73a62a%2Foutcomes%2Fskyrl-entrypoint-driver.g73a62a.12655.1795265415820256921.exception.json)
is in `s3://marin-us-east-02a/tmp/ttl=14d/skyrl/marin/users/dlwh/checkpoints/snowball-inventory/2026.11.20.5/attempts/rendezvous/debug_artifacts/rank0-g73a62a/outcomes/skyrl-entrypoint-driver.g73a62a.12655.1795265415820256921.exception.json`.
Failure retries were zero; preemption retry budgets were 100 for the coordinator
and 1,000 for the child. These are separate retry budgets.

The mesh repair uses TPU slice counts only for TPU devices. GPU PJRT's host
`slice_index` metadata no longer splits the configured global GPU mesh.
The data4/expert8 geometry is unchanged. A 32-device topology regression checks
exact batch and expert shard ownership for GPU descriptors, and checks that
TPU expert groups stay within each slice. These are local topology checks;
the repaired source still requires accelerator acceptance.

Smoke `2026.11.21.1` used coordinator
`/dlwh/users-dlwh-checkpoints-snowball-inventory-2026.11.21.1-coordinator-e6c36b61`.
Its child suffix was `users-dlwh-checkpoints-snowball-inventory-2026.11.21.1-a38272c61e63`.
Both jobs had actual BATCH priority; their accepted bundles matched all 4,582
entries of the frozen `1b6f4a9ca2` archive, including 41 required runtime files.
It failed at `2026-11-21T20:06:58.770Z` during initialization, with `MemoryError`
in the parent process's `stream.read(size)` for a binary response frame.
[W&B run](https://wandb.ai/marin-community/marin-snowball-curriculum/runs/o9tedbzi).
The other JAX processes subsequently aborted during an initialization broadcast.
No GPU or host out-of-memory verdict was recorded. The five GPU pods are gone;
the coordinator pod remains in `Failed` state. Task output archives were empty
or unavailable; the original exception and actual resolved config were collected.

The existing transport puts binary responses on stdout and redirects Python
prints to stderr. Native writes to file descriptor 1 can still corrupt a frame,
including during imports. That vulnerability is reproducible locally; the
failed run's original frame-header bytes were not captured, so the exact cause
of its `MemoryError` remains an inference. The repair gives responses a dedicated
pipe and leaves stdout available for logs.

Smoke `2026.11.21.4` used coordinator
`/dlwh/users-dlwh-checkpoints-snowball-inventory-2026.11.21.4-coordinator-2c045a7f`.
Its accepted source bundle matches all 4,582 entries of frozen `c44f448e65`,
including 41 required runtime files. The final child suffix was
`users-dlwh-checkpoints-snowball-inventory-2026.11.21.4-afb536b55704`; both jobs
had verified BATCH priority. It failed at `2026-11-21T21:15:15.594Z`:
`Levanter learner initialize exceeded 1800 seconds`.

Three native hosts reported reads through shard 39 of 39 by 20:48–20:53 UTC.
The fourth still had unfinished earlier shards: a 3.36-GB chunk's progress span
took 27m46s, and shard 29 began at 21:13:13 UTC. Loader progress includes device
placement, so those timings do not isolate storage latency from transfer time.
They do not establish completed model initialization or training. The exact blocking
frame at the deadline was not captured. All five GPU pods are gone; the original
exception and resolved configuration were collected, with no uploaded task
archives to fetch.

Smoke `2026.11.21.5` used coordinator
`/dlwh/users-dlwh-checkpoints-snowball-inventory-2026.11.21.5-coordinator-4be6473c`.
Its actual coordinator priority is BATCH, and its accepted source archive matches
all 4,582 frozen entries, including 41 required runtime files. Its child suffix
was `users-dlwh-checkpoints-snowball-inventory-2026.11.21.5-ae02721e5e1a`; it
ran at BATCH priority after one automatic gang preemption. It failed at
`2026-11-21T21:48:02.250Z` with `RESOURCE_EXHAUSTED`, requesting 45.28 GiB in
`jit_load_from_state_dict`. The error surfaced while synchronizing a router-bias
fingerprint transfer after asynchronous import; that does not establish that
the small fingerprint itself caused the large allocation. This is a concrete
GPU memory failure, separate from the earlier parent IPC `MemoryError`.
All five GPU pods are gone, and the coordinator is failed. No uploaded task
archives were available; the original exception and resolved config were collected.

The targeted fix donates the importer's owned HF state-dict buffers and waits
for the imported model before allocating Adam state. Previously the donation
flag covered only an abstract template. A tiny CPU compiler probe showed zero
aliased bytes before that correction and 68,816 afterward, while retaining
exact FP32 values and final shardings. This is local evidence of buffer reuse;
the full-model import still failed its H100 fit check.

The public import regression passes for FP32 and BF16 serialized weights with
exact promoted-FP32 values, unchanged final shardings, and consumption of owned
inputs. The old donation setting fails that ownership check. Three related HF
converter checks and four real CPU update/checkpoint/restart cases pass, as does
a shared-storage Torch checkpoint probe. A shape-only census gives 9.30 GiB of
FP32 model parameters per device on data4/expert8; it excludes optimizer state,
import scratch, activations, and allocator overhead, and does not prove H100 fit.

Smoke `2026.11.21.6` used coordinator
`/dlwh/users-dlwh-checkpoints-snowball-inventory-2026.11.21.6-coordinator-6eb0f104`.
Its child suffix was `users-dlwh-checkpoints-snowball-inventory-2026.11.21.6-a450bbb12bfe`.
Both jobs had verified BATCH priority. Their accepted archives match all 4,582
frozen entries, including 42 required runtime files. The executed input identities,
native settings, allocation, and Ray configuration match the source freeze.
The child failed at `2026-11-21T22:25:55.714Z`, with `RESOURCE_EXHAUSTED`
requesting 56.79 GiB in `jit_load_from_state_dict`. This surfaced at
`jax.block_until_ready(model)`, before optimizer state allocation. The source
fix therefore separates import from Adam, but the converter itself still exceeds
available GPU memory. No native initialization or update acceptance passed.
Both jobs are failed, with zero preemptions and zero automatic failure retries.
All five GPU pods are gone; no task output archives were uploaded. The preserved
original exception and actual resolved configuration were collected.

A tiny 32-CPU reproduction on data4/expert8 found 62 all-gathers and eight
all-to-alls in the monolithic import. The follow-up derives each HF tensor's
layout from the abstract imported model and its normal export, then reshards
one owned tensor at a time, waiting for completion. The final conversion is
unchanged. In this reproduction its compiler temporary memory falls from
2,225,668 to 492,544 bytes, and the final conversion has no collectives. The
largest individual reshard uses 540,676 bytes of compiler temporary memory.
Exact FP32 values and every final leaf's sharding are preserved. Three public
import/compiler cases, three generic HF converter cases, and four real CPU
update/checkpoint/restart cases pass on pinned JAX/JAXLIB `0.11.0`. These tiny
compiler measurements establish a bounded-memory mechanism; they do not predict
the full H100 peak or prove that the model fits.

Fresh smoke `2026.11.21.7` uses coordinator
`/dlwh/users-dlwh-checkpoints-snowball-inventory-2026.11.21.7-coordinator-e1479295`.
Both coordinator and child have actual BATCH priority, and their accepted
source archives match all 4,582 frozen entries, including 42 required runtime
files. The child suffix is `users-dlwh-checkpoints-snowball-inventory-2026.11.21.7-aaaf10b4951f`.
After one coordinator preemption and a wait for gang admission, all five GPU
tasks ran. Four native initialization receipts cover process IDs 0–3,
32 global/eight local devices, data4/expert8, the expected model geometry and
immutable checkpoint identity, step zero, and 16 microbatches. Import completed
at `2026-11-21T23:43:23Z`. This establishes initialization; update, publication,
generation, checkpoint continuation, and successful terminal receipts remain
required. Installed parameter counts and detailed runtime identity are emitted
by the later checkpoint acceptance receipts.

Initial publication completed at `2026-11-21T23:46:23.646Z`, with model version
zero and receiver participants 32–39 each comparing 19,687,358,464 bytes with
zero mismatches. Returned generation receipts contain real tokens stamped 1.
The fixed 100-task evaluation completed before training. These establish the
initial integration path, not publication of updated weights.

The run failed at `2026-11-21T23:55:55.101Z` during its first `ppo_train` call.
The preserved exception reaches `fused_cross_entropy_loss/api.py`'s selection-plan
exchange, then `multihost_allgather_sync4::0`: three of four tasks arrived,
with task 3 missing. This exchange also runs for the recipe's fixed XLA path
with BF16 projection logits; it is not evidence that a tuning sweep or optimized
GPU CE kernel ran. The reason task 3 did not reach the barrier is unresolved.
Both jobs are failed, all five GPU pods are gone, and no uploaded task archives
were available. No successful optimizer update or checkpoint continuation is
established.

A separate four-process CPU test reproduces a structural tracing hazard with
the actual CE API. All ranks agree on fixed XLA/BF16 settings and warm the same
JIT. Clearing only ranks 0–2's trace caches makes those ranks enter the Python
selection barrier again, while cached rank 3 returns the same loss without
entering it; the other three time out. This proves the cache-dependent barrier
is unsafe, but does not establish why rank 3 missed the hardware run's barrier.
A follow-up keeps the existing timeouts and kernels and adds opt-in phase,
coordination, and main-thread stack receipts for checkpoint-validation smokes.
The repair checks cross-rank agreement at every deterministic host command
boundary, including cache hits, and passes an immutable projection plan through
the traced loss. That path checks exact geometry, dtypes, and options without a
Python collective. A four-process test now completes the same cache-divergent
case, with bit-exact FP32/BF16 XLA losses and gradients (maximum deviation zero).
Plan, profile, and mode mismatches fail on all hosts before tracing. Production
data4/expert8 geometry with a 512-row batch and 32-row execution microbatches also
passes locally, as do native checkpoint/restart and default FP32 compatibility
checks. Generic unprepared callers retain their prior behavior; this repair
does not qualify their multihost tracing or optimized GPU kernels. The repaired
native path completed these host checks in the subsequent H100 smoke; its
training-memory failure still prevents update acceptance.

A supplemental affected-suite run on JAX/JAXLIB `0.11.1` recorded 4,124 passes,
468 skips, and 29 failures with a 600-second session limit. All 13 launcher
cases pass when inherited UV environment overrides are removed. Clean-main
controls reproduce the Linux-only sandbox error seen in six cases on macOS.
Three router representatives reproduce both startup/cleanup failure types with
clean-main router code under the same dependency context; the deeper router
package-build stall is unresolved. This
broader run is not green. The focused fixed-plan and native checks use the
runtime's pinned JAX/JAXLIB `0.11.0` and pass independently.

Smoke `2026.11.21.8` uses coordinator
`/dlwh/users-dlwh-checkpoints-snowball-inventory-2026.11.21.8-coordinator-31727573`.
Both coordinator and child ran at BATCH priority from frozen source
`233030789c6afd6aa61d315a001ce65ac7436007`. After one coordinator preemption,
the recovered child suffix was
`users-dlwh-checkpoints-snowball-inventory-2026.11.21.8-adaad932e340`.
Both accepted source archives match all 4,582 frozen entries, including 42
required runtime files; the actual model, inputs, native configuration,
allocation, and Ray settings match the freeze.

All four native processes reported GPU execution on H100 80GB devices,
67,078,882,816 installed parameters, 32 global/eight local devices, step zero,
and policy version zero. All eight initial receivers compared 19,687,358,464
bytes each with zero mismatches. Returned generation carried stamp 1. The
fixed CE host checks completed for both forward and train on every host.

The first compiled update failed at `2026-11-22T02:04:31.543Z` with
`RESOURCE_EXHAUSTED`, requesting 41.05 GiB in `jit__train_step`. The error
surfaced at `Trainer.train_step`'s loss synchronization; the allocation size
does not identify the offending HLO buffer or measure total peak memory.
The [original trainer exception](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Ftmp%2Fttl%3D14d%2Fskyrl%2Fmarin%2Fusers%2Fdlwh%2Fcheckpoints%2Fsnowball-inventory%2F2026.11.21.8%2Fattempts%2Frendezvous%2Fdebug_artifacts%2Frank0-g73e352%2Foutcomes%2Fskyrl-trainer.g73e352.17071.1795313071543634745.exception.json)
is preserved at
`s3://marin-us-east-02a/tmp/ttl=14d/skyrl/marin/users/dlwh/checkpoints/snowball-inventory/2026.11.21.8/attempts/rendezvous/debug_artifacts/rank0-g73e352/outcomes/skyrl-trainer.g73e352.17071.1795313071543634745.exception.json`.
Both jobs are failed; all five GPU pods are absent. The child head's output
archive is empty and its siblings report none. No completed optimizer update,
updated-policy publication, or native checkpoint continuation is established.

Small CPU compiler probes show trainer-state donation and existing layer
checkpointing, while a publication lifetime probe releases the BF16 snapshot
after the session. Those checks do not explain the GPU allocation: CPU and
GPU MoE kernels differ, and GPU allocator reservations were not measured.
The memory diagnosis therefore remains open, with recipe, numerical
tolerances, kernels, and memory fraction unchanged.

The follow-up adds compiler evidence to the existing checkpoint-validation
option: memory analysis, scheduled optimized HLO, buffer assignment, and
available allocator counters are saved before the first update. Bounded
uploads use a CPU-only process and explicit 02a lifecycle storage. Real CPU
tests preserve exact diagnostics-on/off model, Adam, RNG, step, and averaging
state, and retain the original training exception when upload fails.
Two-process checkpoint continuation and independent source review pass.
Diagnostic smoke `2026.11.21.9` uses coordinator
`/dlwh/users-dlwh-checkpoints-snowball-inventory-2026.11.21.9-coordinator-9066931e`,
submitted at `2026-11-22T02:53:39.792Z`. It has verified BATCH priority and zero
automatic failure retries. Both jobs are failed, with zero preemptions. All
five GPU pods are absent; the CPU coordinator pod remains terminal `Failed`.
Head and coordinator task archives are empty, with sibling archives unreported.
Original trainer and driver exceptions were fetched separately.

Its child suffix was `users-dlwh-checkpoints-snowball-inventory-2026.11.21.9-af1cfc374bb8`.
Both accepted bundles match all 4,583 frozen entries, including 43 required
runtime files. The executed inputs, native configuration, allocation, and Ray
settings match the freeze. All four native H100 processes initialized the full
model and all eight initial receivers compared 19,687,358,464 bytes each with
zero mismatches. No completed update or checkpoint continuation followed.

The child failed in its first compiled update. All four hosts uploaded their
pre-update compiler and allocator receipts. The scheduled optimized HLO,
buffer assignment, and memory report were fetched for rank zero, with numeric
receipts fetched for every rank. The
[rank-zero receipt](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Ftmp%2Fttl%3D14d%2Fxla-dumps%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.21.9-af1cfc374bb8%2Fprocess-0%2Fpid-16431%2Fbefore-update.json)
is at
`s3://marin-us-east-02a/tmp/ttl=14d/xla-dumps/users-dlwh-checkpoints-snowball-inventory-2026.11.21.9-af1cfc374bb8/process-0/pid-16431/before-update.json`.

All values below are per device. Compiled memory accounting is identical across
all four ranks.

| Quantity | Observed GPU value |
| --- | --- |
| Compiled temporary arena, each device | 44,076,557,280 bytes (41.04949 GiB) |
| Rounded failed allocation | 44,076,557,312 bytes |
| Largest free block, every device | 32 GiB |
| Aggregate allocator headroom | 43.072–43.170 GiB |
| Existing pool size | 64.00195 GiB |
| Allocator limit | 71.26367 GiB |
| Remaining pool growth within the limit | 7.26172 GiB |
| Compiled allocation accounting, with state aliases | 68.95833 GiB |

The failed request matches buffer-assignment allocation 3482, the compiled
temporary arena, rather than one tensor of that size. Failure snapshots retain
the same pool and largest free block. Even combining the largest free block
with all remaining growth would provide only 39.26172 GiB. Independent review
therefore supports a contiguity/region-growth failure. JAX documents the
fragmentation risk of [disabling preallocation](https://docs.jax.dev/en/latest/gpu_memory_allocation.html).
This is not proof that every later training operation fits: measured live
memory plus the arena leaves only 2.023–2.121 GiB of optimistic headroom.

The `.10` remedy kept BFC and enabled preallocation at the same 0.90
memory limit. This deliberately changed the earlier disabled-preallocation
setting to address the measured pool fragmentation. It changed no training
math, kernels, precision, geometry, or numerical tolerance. The run still
needed to prove that the resulting pool could serve publication and updates;
initialization alone did not qualify the full loop.
The fresh acceptance attempt is recorded below.

The candidate passes 13 focused real-process CPU cases, including rejection of
the pinned allocator overrides before backend creation and one/two-process
update/checkpoint continuation. Three schema round-trip checks, required lint
and types, strict docs, and independent source review pass. The rendered
Megatron control is unchanged; the native runtime configuration differs from
smoke `.9` only by `preallocate: true`. These checks qualify the startup
boundary and preserve local state semantics; they do not establish H100 fit.

Smoke `2026.11.21.10` uses coordinator
`/dlwh/users-dlwh-checkpoints-snowball-inventory-2026.11.21.10-coordinator-ddb628ca`,
submitted at `2026-11-22T03:55:21.099Z`. Its actual priority is BATCH, with zero
automatic failure retries. Root source, configured priority, retry budgets,
and all frozen bundle contents were verified. The source pin is
`42a248d50564f44d6010a35591d49d2f65c0ea38`; later status-only documentation
commits are not its executed source.

Its child suffix was `users-dlwh-checkpoints-snowball-inventory-2026.11.21.10-a4e4cc5c9f7f`.
All four learners initialized the full model with resolved BFC/preallocate=true,
fraction 0.90, unified memory disabled, and observed per-device limits of
76,518,785,024 bytes. The first proven failure is at
`2026-11-22T04:20:15–16Z`: NCCL fails a 2,097,152-byte CUDA allocation while
creating a new 32-device JAX communicator during initial publication staging.
External communicator/context memory pressure is supported; exact per-library
physical usage was not recorded. Training memory fit remains unverified.

Torch receiver groups had completed their warmup. Their first real payload
then timed out; parent actors waited for the native publication response and
were terminated by the Torch watchdog. The original trainer exception is
`ActorDiedError` during `_sync_policy_for_rollouts`. Ray's generic list of
possible causes does not establish SIGSEGV; the full stderr records an explicit
watchdog abort. Both jobs failed with zero preemptions, all five GPU pods are
absent, and the CPU coordinator pod remains terminal `Failed`. Task archives
are empty or unreported; original exceptions and bounded full native stderr
were fetched separately.

A dynamic VMM allocator candidate is under review at the same 0.90 budget
for physical allocations, with preallocation disabled. Pinned XLA source enforces
that budget on padded physical allocations and maps P2P-capable peers.
Managed collective colors share the budget; external NCCL, Torch, and context
allocations do not. If CUDA recommends 2-MiB allocation granularity, rounding
adds about 1.61 GiB for this recorded train program; the driver's actual
granularity has not been measured. Fit remains uncertain. Standard
JAX allocator counters do not support this pinned VMM path; the candidate
adds explicitly labeled parent Torch and global driver-memory receipts.
VMM hardware acceptance is pending in the fresh attempt below.

The VMM candidate passes three schema checks and a 22-case native CPU run.
After isolating observer errors, three normal host continuations and the real
blocked-output/NaN-error case pass on the final production source. Required
lint/types, strict docs, and independent review pass. The rendered control is
unchanged; native configuration differs from `.10` only by `allocator: vmm`
and `preallocate: false`. These checks do not qualify GPU allocation or fit.

Smoke `2026.11.21.11` uses coordinator
`/dlwh/users-dlwh-checkpoints-snowball-inventory-2026.11.21.11-coordinator-b6cb00ac`,
submitted at `2026-11-22T05:14:39.249Z`, with child suffix
`users-dlwh-checkpoints-snowball-inventory-2026.11.21.11-ac3a25737ef9`.
Both jobs have verified BATCH priority and zero automatic failure retries.
All five child tasks are running with zero failures/preemptions in the startup
snapshot. Source pin `e9fa30e8a2230a4015d8488280d2b375b81ea07d` is frozen;
later status-only documentation commits are not its executed source.

Native initialization
budget is 3,600 seconds, matching SkyRL's existing outer model-setup budget;
the outer deadline can expire slightly earlier because it includes dispatch
and finalization. Ordinary RPCs retain their 1,800-second limit. This changes
no model, optimizer, objective, kernel, precision, or numerical tolerance.
It does not guarantee that a slow cold load will finish in time.
Comparison versions `2026.11.21.2` and `2026.11.21.3` remain unsubmitted.
The transport repair passes five pinned JAX/JAXLIB `0.11.0` cases: native
import-time stdout isolation and four real CPU update/checkpoint/restart cases.
Required lint/type checks, strict docs, and independent transport review pass.
No successful accelerator update is established by this record.
Run from the repository with the required storage, evaluation, and judge
credentials loaded. The smoke command records the submitted attempt; use a
fresh unused version for another attempt.
Each uses 40 H100s at BATCH priority with zero automatic failure retries. The regional route
submits directly to the configured CoreWeave context. The smoke restarts JAX
learners within the same allocation to validate native checkpoint continuation.
It retains the parent actors and rollout engines.

```bash
SELECTION_URI=s3://marin-us-east-02a/marin/rl-data/task-selection/snowball-100k/2026.11.04
uv run python -m experiments.post_training.inventory.train_quick_loop \
  --selection-uri "$SELECTION_URI" --version 2026.11.21.8 \
  --learner levanter --steps 2 --verify-publication \
  --validate-native-checkpoint --coordinator-route regional --run
```

After the smoke passes initialization/update, publication/generation, and
checkpoint continuation gates in the table, submit the matched arms:

```bash
uv run python -m experiments.post_training.inventory.train_quick_loop \
  --selection-uri "$SELECTION_URI" --version 2026.11.21.2 \
  --learner megatron --steps 25 --verify-publication --coordinator-route regional --run
uv run python -m experiments.post_training.inventory.train_quick_loop \
  --selection-uri "$SELECTION_URI" --version 2026.11.21.3 \
  --learner levanter --steps 25 --verify-publication --coordinator-route regional --run
```

Checkpoint restart validation is disabled for both comparison arms. Record the
executed source/bundle pins and run links after submission, then inspect the
acceptance receipts and output archives. Native terminal HF export remains
unimplemented. No throughput or reward comparison is available yet.
