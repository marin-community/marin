# Levanter quick-loop status

November 21, 2026 (PST).

Fresh BATCH smoke `2026.11.21.8` is running from
`233030789c6afd6aa61d315a001ce65ac7436007`, with the fixed XLA coordination
repair and unchanged recipe. At 17:23 PST its CPU coordinator was running,
with no GPU child yet discovered. Current-source accelerator acceptance is
pending.

The preceding BATCH smoke
`2026.11.21.7`, from `8f19c196456e2f600c0e4edb21f4339c0a8d5129`, has
initialized all four native Snowball learner hosts on the 32-H100 data4/expert8
mesh, byte-verified initial publication to all eight vLLM receivers, and completed
the fixed 100-task baseline evaluation. It then failed at 15:55:55 PST in the
first `ppo_train` call: three hosts reached a cross-entropy selection barrier,
but rank 3 did not arrive before its deadline. Why that rank did not arrive is
under investigation. All five GPU pods are gone. Real updates, publication and
generation after an update, and checkpoint continuation remain unverified.
The matched comparison is gated on full smoke acceptance.

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
| Actual H100 | Earlier smokes exposed device selection, mesh validation, response framing, timeout, and import-memory failures. Smoke `2026.11.21.7` initialized all four native hosts on the expected 32-device mesh, from the frozen checkpoint and model config. | Full runtime identity/installed parameter-count receipts, scoring, and two real parameter updates with the pinned CUDA kernels. |
| Publication and resume | Initial H100 publication at model version zero passed all eight receiver byte comparisons, followed by generation stamped 1 and the 100-task baseline evaluation. CPU Gloo replay and four fresh-process native checkpoint transactions passed exact model, Adam, RNG, step, and model-averaging fingerprints. | Publication of actual updated weights with zero byte mismatches, nonempty generation stamped 2 after model version 1, terminal checkpoint, fresh learner processes restoring exact state, and a further update. |
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
native path still requires a new H100 acceptance run.

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
Its actual priority is BATCH, and its accepted source archive matches all 4,582
frozen entries, including 42 required runtime files. No accelerator gate for
this new source has passed yet.

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
No successful accelerator rerun is established by this record.
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
