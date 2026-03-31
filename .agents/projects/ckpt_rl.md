# RL Checkpointing Analysis

Date: 2026-03-29

## Sources

- [ckpt_monitor_log.md](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/scratch/ckpt_monitor_log.md)
- [iris-rl-claude.md](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/.agents/logbooks/iris-rl-claude.md)
  Relevant sections:
  - continuous monitoring and RSS trend: around lines 1393-1479
  - memory composition analysis: around lines 1479-1568
  - `r2` timing slowdown + composition snapshots: around lines 1588-1738
- [iris_rl_e4ms2_500_train_filtered.log](/tmp/iris_rl_e4ms2_500_train_filtered.log)
  Exact trainer checkpoint chronology for the older pre-debug `/ahmed/iris-rl-e4ms2-500-0327-v2` run.
- [iris_rl_e4ms2_500_old_checkpoint_table.tsv](/tmp/iris_rl_e4ms2_500_old_checkpoint_table.tsv)
  Parsed machine-readable table derived from the old trainer Iris log.

## Short Synopsis

As of 2026-03-30, the evidence now tells a more unified story than this note originally did.

There are really three different checkpoint regimes in this investigation:

| Regime | Evidence | Checkpoint behavior | Outcome |
|---|---|---|---|
| old pre-debug production-like run | exact Iris trainer logs from `/ahmed/iris-rl-e4ms2-500-0327-v2` | `77` successful saves, mean **202.2s**, median **199s**, range **147-280s** | baseline checkpoint cost was already about **200s**, but the run suffered retries and `5` interrupted/incomplete checkpoint attempts |
| heavily instrumented debug `r2` run | Claude's timing and memory instrumentation | checkpoint totals stretched **164s -> 397s -> 554s -> 703s -> 907s -> 968s** with growth concentrated in `filesystem_ready`, `tensorstore_serialize`, and `metadata_write` | this was a pathological degraded mode, not a normal steady-state checkpoint profile |
| current clean 500-step control run | clean direct RL run with debug instrumentation removed and previous-temp deletion callback disabled | through Claude checkpoint `75`, totals remain in the same broad **~170-250s** band, with no monotonic drift and `0` checkpoint failures | operationally stable so far; trainer has survived `2` preemptions and resumed cleanly from checkpoints |

The unified read is:

1. Checkpointing in this RL setup was never cheap.
   - Both the old pre-debug run and the current clean run center on roughly **200 seconds per save**.
   - That means the main baseline problem is cost, not a newly introduced slowdown.

2. The checkpoint path is intrinsically memory-heavy.
   - The best evidence still points to:
     - a large trainer/runtime/native floor of about **90-130 GiB**
     - a stable Arrow Flight weight snapshot of about **15.3 GiB**
     - a temporary checkpoint materialization working set of about **89.75 GiB**
   - That naturally produces **200-300+ GiB** RSS windows during save without requiring a simple monotonic leak.

3. The dramatic late-run slowdown was specific to the heavy debug path, not the baseline production path.
   - The `r2` run clearly discovered a real progressive slowdown mode.
   - But the old pre-debug Iris logs and the current clean run both fail to show that same monotonic progression.
   - The strongest present explanation is therefore: the heavy checkpoint debugger materially perturbed the save path and created or amplified the pathological slowdown.

4. Disabling callback-based previous-temp deletion may still help, but it no longer looks like the primary story.
   - It remains worth isolating with a small clean A/B later.
   - But it does not explain why the debug run's `serialize` phase itself blew up while the old and clean runs stayed near the same baseline.

5. The operationally important question has narrowed.
   - It is no longer "is Iris RL checkpointing fundamentally collapsing over time?"
   - It is now:
     - can the clean run finish all `500` steps with the same bounded checkpoint behavior?
     - how much additional robustness or speed comes specifically from disabling previous-temp deletion?
     - can the stable but still-expensive **~200s** checkpoint baseline be reduced without destabilizing training?

In one sentence:

The checkpoint path is still expensive and memory-heavy, but the best current evidence says the catastrophic slowdown was largely a debug-induced artifact layered on top of a baseline save cost that was already about **200s** and is now behaving stably again in the clean control run.

## Concrete Evidence

### 1. Checkpoint memory is intrinsically large

From Claude's memory-composition analysis:

- checkpoint payload is about `89.75 GiB`
- this is dominated by model + optimizer state
- the weight-transfer store holds a separate persistent snapshot of about `15.3 GiB`

This is not speculative; it comes directly from the logged checkpoint size and the Arrow Flight debug snapshot.

### 2. The Arrow Flight store is not growing over time

The `r2` composition snapshots repeatedly show:

- `stored_arrow_bytes_mib ≈ 15316.5`
- `stored_param_count = 291`
- one current `weight_id`

That argues strongly against "the weight store is leaking every step." It looks stable at one snapshot's worth of data.

### 3. Baseline RSS is not a clean monotonic leak

The earlier run's serialize-start baselines oscillate:

- 69 GiB at step 256
- 132 GiB at step 260
- 182 GiB at step 264
- 150 GiB at step 268
- 188 GiB at step 272
- 90 GiB at step 276
- 156 GiB at step 280
- 76 GiB at step 284
- 150 GiB at step 288

That is not the shape of a simple ever-growing leak.

### 4. Peak memory is still very high

Observed peaks in the monitor log:

- 300 GiB at step 272
- 318 GiB at step 300
- 251 GiB at step 328
- around 200 GiB on several later `r2` checkpoints

So even without a monotonic leak, the checkpoint path is repeatedly entering a very high-RSS regime.

### 5. The `r2` run discovered a strong timing trend

The `r2` per-phase timing table is the strongest new signal:

| Step | filesystem_ready | serialize | async_commit | metadata_write | Total |
|---|---:|---:|---:|---:|---:|
| 322 | ~0s | ~90s | ~60s | ~14s | 164s |
| 325 | 61s | 190s | 73s | 74s | ~397s |
| 328 | 100s | 255s | 86s | 113s | ~554s |
| 331 | 177s | 324s | 78s | 211s | 703s |
| 334 | 248s | 401s | 89s | 313s | 907s |
| 338 | 326s | 397s | 87s | 293s | 968s |

This is not random noise. Three phases get steadily slower:

- `filesystem_ready`
- `tensorstore_serialize`
- `metadata_write`

Meanwhile:

- `async_commit` stays relatively flat at about `73-89s`

That asymmetry matters a lot.

### 6. The first `r2` memory composition snapshots show two different kinds of memory

Step 322 initially made it look like Python allocations tracked RSS closely enough that the mystery was mostly Python-side arrays.

But step 325 is more informative:

- at `+135s` in serialize:
  - RSS ≈ `136 GiB`
  - tracemalloc ≈ `20 GiB`
  - gap ≈ `116 GiB`
- later, tracemalloc jumps sharply as Python arrays materialize

This suggests two waves:

1. Native / TensorStore / JAX staging grows first
2. Python-visible `numpy` arrays from `device_get` show up later

So the memory story is not "all Python" and not "all native." It is a mixed pipeline.

### 7. Previous async-commit overlap has not actually been proven in the observed `r2` checkpoints

Claude explicitly notes that on steps 322 and 325:

- `previous_async_commit_alive = false` at serialize start

So the "two full checkpoints overlap" hypothesis is plausible, but not yet directly supported by the observed `r2` evidence.

That matters because it means we should not treat overlap as a solved explanation.

## What Seems To Be Happening

### Why so much memory is being used

The best evidence-backed breakdown is:

1. There is a substantial fixed trainer/native floor.
   - Depending on how you count it, this looks like roughly `90-130 GiB`.
   - This is not necessarily "leaked" memory; it likely includes runtime, model state, native buffers, and caches.

2. The weight-transfer system persistently holds about `15.3 GiB`.
   - This is stable, not growing.

3. Each checkpoint then adds a very large temporary working set.
   - Roughly `89.75 GiB` of checkpoint payload has to be staged/materialized.
   - Some of that is native/C++ staging.
   - Some later becomes Python-visible `numpy` arrays.

So a plausible steady-state checkpoint footprint is:

- native/runtime floor: `~90-130 GiB`
- Arrow Flight store: `~15 GiB`
- checkpoint staging/materialization: `~90+ GiB`
- plus transient handoff/copy effects

That naturally lands in the `200-300+ GiB` range without requiring any leak at all.

### Why checkpoint saves seem to get progressively slower

The strongest pattern is that the slowdown is not concentrated in one local compute phase.

The phases that get slower are exactly the ones that look storage-facing or storage-coupled:

- `filesystem_ready`
- `tensorstore_serialize`
- `metadata_write`

`async_commit` is comparatively stable.

That makes me think the dominant degradation is more likely to be one of:

1. GCS / TensorStore backpressure
2. old-checkpoint deletion in the hot path becoming increasingly expensive
3. object-store / metadata operations under the same prefix getting throttled or slowed
4. slower draining of TensorStore writes, which in turn slows serialize because staging cannot clear quickly

What I do **not** think the timing trend looks like:

- not a pure compute slowdown
- not a replay-buffer issue
- not an Arrow Flight store accumulation problem

The fact that `filesystem_ready` goes from `0s` to `326s` is especially damning. That phase is not about model math. Something external to the trainer step compute is degrading.

Important correction from code inspection:

- in the current debug build, `filesystem_ready` is not just storage preparation
- it also includes the forced `gc.collect()` and `tracemalloc` baseline-reset work that happens before `tensorstore_serialize`

So the `filesystem_ready` slowdown is real elapsed time, but it is not safe to attribute all of it to GCS or checkpoint cleanup. Some of that growth is likely self-inflicted debug overhead.

### A likely combined mechanism

The most coherent combined story is:

1. Checkpoint-related storage operations get slower over time.
2. Because they get slower, checkpoint staging buffers remain live longer.
3. Because those buffers remain live longer, the trainer spends more wall-clock time sitting inside native checkpoint code.
4. During that longer window, total RSS stays elevated for longer.
5. On a bad checkpoint, one of two things happens:
   - transient host memory pressure spikes high enough to kill or wedge the process
   - or the native serializer stalls long enough that the worker stops heartbeating and JAX coordination aborts it

This combined explanation fits both families of symptoms:

- high memory usage
- long `tensorstore_serialize` stalls

## Hypotheses

## Hypothesis 1: The checkpoint path is memory-heavy by construction, not because of a simple leak

Evidence:

- stable `15.3 GiB` Arrow Flight store
- non-monotonic baseline RSS
- repeated large checkpoint spikes consistent with checkpoint size
- mixed native + Python staging seen in `r2`

Conjecture:

- the fixed native floor is mostly runtime/caches/staging rather than leaked objects

Confidence: high

## Hypothesis 2: The progressive slowdown is primarily storage-path degradation, not trainer compute degradation

Evidence:

- `filesystem_ready`, `serialize`, and `metadata_write` all get slower
- `async_commit` stays comparatively stable
- first post-resume checkpoint is fast, later ones are much slower

Conjecture:

- the exact culprit is probably GCS / TensorStore / old-checkpoint deletion behavior, but we do not yet have per-operation timing inside those subcalls
- part of the measured `filesystem_ready` degradation is likely due to forced GC + tracemalloc overhead in the current debug build

Confidence: medium-high

## Hypothesis 3: Slower checkpoints increase both the duration and the danger of the memory spike

Evidence:

- later checkpoints spend much longer inside `filesystem_ready` and `tensorstore_serialize`
- memory composition shows large staging buffers during serialize
- earlier crashes always happened inside `tensorstore_serialize`, before commit finished

Conjecture:

- slower storage does not just hurt throughput; it likely increases the time window in which high-memory staging is live, raising the chance of a fatal unlucky checkpoint

Confidence: medium

## Hypothesis 4: The earlier crash was likely caused by high memory pressure plus a long native serialize stall, not by a monotonic leak

Evidence:

- crash boundary was repeatedly `tensorstore_serialize`
- no pre-emption
- no rollout-first failure
- current data does not show a monotonic runaway baseline
- current data does show very large transient memory and very long serialize stalls

Conjecture:

- the exact terminal trigger might have been:
  - cgroup OOM,
  - native TensorStore/JAX stall causing heartbeats to stop,
  - or some combination where memory pressure makes the native path unstable

We still do not have a direct smoking gun between those.

Confidence: medium

## Hypothesis 5: Previous async-commit overlap may contribute, but it is not yet demonstrated

Evidence:

- conceptually, overlap would be dangerous
- it would explain some very high peaks

Counter-evidence:

- on the observed `r2` composition snapshots, `previous_async_commit_alive` was false at serialize start

Conjecture:

- overlap may happen on later or failed checkpoints, but we have not yet caught it in the act

Confidence: low-medium

## What I Think Is Concrete vs Conjecture

### Concrete

- checkpoint payload is about `89.75 GiB`
- Arrow Flight store is about `15.3 GiB` and stable
- baseline RSS oscillates rather than monotonically climbs
- checkpoint phase times in `r2` are getting progressively worse
- `filesystem_ready` and `metadata_write` degrade dramatically
- `async_commit` is relatively stable
- `r2` shows both native and Python-visible memory components
- observed crashes happen in `tensorstore_serialize`
- the current `filesystem_ready` timings are contaminated by debug GC / tracemalloc work

### Conjecture

- the exact native component is TensorStore/GCS staging rather than some other JAX runtime buffer
- GCS rate limiting / storage throttling is the main root cause of the slowdown
- the earlier crash was specifically OOM rather than heartbeat-loss-from-stall
- previous async-commit overlap was present on the crashing checkpoints

## My Current Take

If I had to compress this into one sentence:

The checkpoint path has a large but apparently bounded baseline cost and memory footprint, and the strongest current evidence is that the catastrophic slowdown came from the heavy debug path rather than from the normal production checkpoint flow.

The old pre-debug run and the current clean run now anchor the baseline: both sit almost exactly at a **~200s** successful-save median/mean, even though the old run had retry-related checkpoint interruptions and the clean run has so far remained checkpoint-healthy. That changes the interpretation of the `r2` debug run. I still believe the `r2` slowdown was real, but I no longer think it should be read as "the production path inevitably degrades this way." The better read is "the production path is expensive but bounded, while the fully instrumented debug path created a separate pathological mode."

The remaining practical concern is not whether checkpointing is fundamentally broken, but whether we can both:

1. finish the clean `500`-step run without a checkpoint failure
2. reduce the stable but still high baseline checkpoint cost
3. isolate whether previous-temp deletion contributes a meaningful second-order effect

## Immediate Follow-up Questions

1. Does the clean control run finish all `500` steps with the same bounded checkpoint timing and no checkpoint failure?
2. In a small clean A/B, how much does `delete_previous_temporary_checkpoint_after_save=True` change checkpoint wall time or robustness?
3. Why do occasional **240-280s** checkpoint spikes still appear in both the old pre-debug run and the current clean run even when there is no monotonic degradation?
4. Can the stable baseline checkpoint cost of about **200s** be reduced without changing the direct RL topology?
5. Are trainer preemptions purely platform noise, or do they correlate with checkpoint windows strongly enough to justify extra mitigation?

## Historical Handoff Plan (pre-clean-run)

This section captures the plan written before the clean 500-step control run was launched.
It is useful as provenance, but it is no longer the top-line recommendation. Use the synopsis above
and the execution update below as the current state.

### Primary Goal

Optimize for:

1. direct RL speed
2. direct RL robustness

Do **not** optimize for maximum checkpoint observability in the main 500-step run.

### Current Run Policy

For the currently running heavily instrumented debug run:

1. Do **not** use it as the throughput or production-readiness signal.
2. If the user wants compute back, kill it.
3. If the user wants one final debug datapoint, allow only the current active checkpoint to resolve, then kill it.
4. Do **not** keep it running for another full long debugging session.

### Next Long Run: Required Settings

The next 500-step direct RL run should be launched with all of the following:

1. `debug_checkpointer=False`
2. `delete_previous_temporary_checkpoint_after_save=False`
3. same stable `run_name` resume scheme as current direct RL runs
4. same topology unless the user explicitly asks to change topology
5. explicit host-memory settings
6. for the current control run: keep `train_ram=400g` and `inference_ram=400g`

### Next Long Run: Forbidden Settings

The next 500-step direct RL run should **not** use:

1. `debug_checkpointer=True`
2. forced `gc.collect()` before every checkpoint
3. `tracemalloc`
4. automatic thread dumps
5. top-allocation diff logging
6. any new custom background thread that touches JAX arrays or checkpoint serialization

### Why

These are the reasons the next agent should not deviate:

1. The fully instrumented `debug_checkpointer` is materially perturbing checkpoint timing.
2. The current debug run is useful for forensics, but it is not a clean measurement of production direct RL.
3. `delete_previous_temporary_checkpoint_after_save=False` is needed to separate save performance from callback-based previous-temp cleanup.
4. JAX's built-in `GlobalAsyncCheckpointManager` is already the supported async mechanism; do not invent a new checkpoint thread model.

### What Observability Is Still Acceptable In Long Runs

If the next agent wants lightweight observability in the long run, keep only:

1. checkpoint phase markers
2. periodic RSS
3. Arrow Flight weight-store bytes
4. previous async commit alive / futures-done state

If these lightweight signals are not already available without `debug_checkpointer`, the next agent should prefer:

1. adding a separate lightweight checkpoint-monitor mode
2. not reusing the full forensic mode

### Separate Profiling Run Policy

If the next agent still needs checkpoint forensics, that work must happen in a **separate short run**, not the main 500-step run.

That separate profiling run should:

1. run only long enough to capture 2-4 checkpoints
2. keep the heavy checkpoint instrumentation on
3. split timing into these exact subphases:
   - `fs.makedirs`
   - forced `gc.collect()`
   - `tracemalloc.take_snapshot()`
   - `manager.serialize(...)`
   - metadata write
4. capture native memory more explicitly if possible:
   - cgroup memory
   - or `/proc/self/smaps_rollup`

### First Experiment The Next Agent Should Run

The next agent's first experiment should be:

1. a new 500-step direct RL run
2. with `debug_checkpointer=False`
3. with `delete_previous_temporary_checkpoint_after_save=False`
4. with the same stable run-name resume logic
5. with explicit `train_ram` / `inference_ram`
6. for the current control run: keep both at `400g`

The next agent should then measure:

1. actual checkpoint wall time
2. trainer failures / retries
3. end-to-end step wall time
4. whether checkpoint slowdown still happens without the heavy debug path

### Decision Tree After That Run

After the next long run:

1. If checkpoint time is now acceptable and the run is robust:
   - treat direct RL as much closer to ready
   - move on to executor-path parity

2. If checkpoint time is still too slow but the run is robust:
   - increase `save_interval`
   - and/or design separate checkpoint classes:
     - frequent light/model checkpoints
     - infrequent full optimizer checkpoints

3. If checkpoint failures still happen even without the heavy debug path:
   - run a short dedicated profiling job with the forensic instrumentation
   - do **not** immediately re-enable the heavy debugger on the main long run

### Explicit Non-Goals For The Next Agent

The next agent should **not** spend time on these before running the next clean 500-step direct RL experiment:

1. more full-forensic instrumentation in the long run
2. new custom async checkpoint thread designs
3. v6e work
4. packed rollout work
5. executor work

Those all come after we re-establish a clean speed/robustness signal for direct RL.

### My practical recommendation

If optimizing for speed + robustness right now, I would launch the next long RL run with:

- `debug_checkpointer=False`
- `delete_previous_temporary_checkpoint_after_save=False`
- same stable `run_name`
- same topology
- explicit RAM settings, with the current control run staying at `400g` / `400g`

Then I would keep a completely separate short profiling run for checkpoint diagnosis only.

That gets us:

- a real measurement of production-speed RL
- a clean separation between checkpoint-debug overhead and the actual checkpoint cost

## 2026-03-29 execution update

Important correction:

- `delete_old_temp_checkpoints=False` was not the right knob for the main in-run cleanup path
- that existing flag only controls whether a temporary checkpoint discovered at startup from a prior attempt is carried forward for later deletion
- the actual hot-path cleanup after each successful new save was the callback that deletes the previously saved temporary checkpoint
- the repository now has a separate explicit knob for that behavior:
  `delete_previous_temporary_checkpoint_after_save`

Platform constraint:

- v5 TPU host memory effectively tops out around `440 GiB`
- to keep this experiment single-variable, the current clean control run intentionally keeps both trainer and rollout at `400g`
- that means this run isolates:
  - heavy debug instrumentation removed
  - callback-based previous-temp deletion removed
- and does **not** simultaneously test a higher host-memory budget

Launched clean control run:

- stable run name: `iris-rl-e4ms2-500-clean-nodelprevtmp`
- root job: `/ahmed/iris-rl-e4ms2-500-0329-clean-nodelprevtmp-r1`
- script: [exp_iris_rl_regression_direct_gcs_prod.py](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/experiments/exp_iris_rl_regression_direct_gcs_prod.py)

Exact launch command:

```bash
uv run iris --config=lib/iris/examples/marin.yaml job run --no-wait \
  --user ahmed \
  --job-name iris-rl-e4ms2-500-0329-clean-nodelprevtmp-r1 \
  --region us-central1 \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e HF_TOKEN "$HF_TOKEN" \
  -- uv run python experiments/exp_iris_rl_regression_direct_gcs_prod.py \
    --run-name iris-rl-e4ms2-500-clean-nodelprevtmp \
    --num-train-steps 500 \
    --n-prompts 64 \
    --eval-frequency 1 \
    --num-rollout-workers 2 \
    --region us-central1 \
    --inflight-weight-updates \
    --train-ram 400g \
    --inference-ram 400g \
    --no-delete-previous-temporary-checkpoint-after-save
```

Initial controller state right after launch:

- root state: `JOB_STATE_RUNNING`
- `failure_count=0`
- `preemption_count=0`

Background monitor:

- track: `iris`
- mode: monitor-only
- PTY session id: `18111`
- state file:
  [20260330-0117_iris-rl-e4ms2-500-0329-clean-nodelprevtmp-r1_monitoring_state.json](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/scratch/20260330-0117_iris-rl-e4ms2-500-0329-clean-nodelprevtmp-r1_monitoring_state.json)
- event stream:
  [20260330-0117_iris-rl-e4ms2-500-0329-clean-nodelprevtmp-r1_events.jsonl](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/scratch/20260330-0117_iris-rl-e4ms2-500-0329-clean-nodelprevtmp-r1_events.jsonl)
- checkpoint sidecar:
  [20260330-0117_iris-rl-e4ms2-500-0329-clean-nodelprevtmp-r1_checkpoints.jsonl](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/scratch/20260330-0117_iris-rl-e4ms2-500-0329-clean-nodelprevtmp-r1_checkpoints.jsonl)
- text log:
  [20260330-0117_iris-rl-e4ms2-500-0329-clean-nodelprevtmp-r1_babysit.log](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/scratch/20260330-0117_iris-rl-e4ms2-500-0329-clean-nodelprevtmp-r1_babysit.log)

Monitor cadence:

- startup sleep: `60s`
- normal sleep: `180s`
- max monitor duration: `28800s` (`8h`)

What the monitor records:

- root + child job states
- failure / preemption counters
- filtered interesting log lines from root and children
- W&B URLs when present in logs
- structured JSONL events plus a plain-text tail log
- derived per-checkpoint records with `step`, `checkpoint_started_at`, `checkpoint_saved_at`,
  `train_step_completed_at`, `save_seconds`, and `start_to_step_complete_seconds`

Consolidated checkpoint coverage:

- Claude's logbook now contains a contiguous checkpoint table through **checkpoint 73 / step 459**.
- Claude's latest status block marks the run **healthy at step 459/500**, with about **41 steps** and
  roughly **3 more checkpoints** left.
- The consolidated table below is therefore complete through the latest completed checkpoint currently
  written into Claude's logbook.
- Rows `1-12` have detailed phase timing (`commit_start`, `serialize`, `commit`).
- Rows `13-73` only expose `save start`, `saved`, and total wall time.
- There is one trainer preemption between checkpoints `40` and `41`; post-resume rows remain in the
  same wall-time band rather than showing a new upward drift.

Checkpoint table analysis:

- All checkpoints `1-73`: mean **201.9s**, median **198s**, range **131-278s**
- Warm checkpoints `5-73`: mean **202.5s**, median **198s**, range **168-278s**
- Pre-preemption checkpoints `1-40`: mean **196.4s**, median **193s**, range **131-274s**
- Post-preemption checkpoints `41-73`: mean **208.5s**, median **210s**, range **168-278s**
- Step-gap distribution between checkpoints: mean **6.32** steps, with counts `{6: 45, 7: 26, 3: 1}`
- The lone 3-step gap is the checkpoint-resume boundary `254 -> 257` after the trainer preemption.
- Slowest warm checkpoints so far: `#43/step-270=278s`, `#33/step-212=274s`, `#34/step-218=257s`,
  `#65/step-409=254s`, `#70/step-441=254s`
- Fastest warm checkpoints so far: `#44/step-276=168s`, `#7/step-42=170s`, `#17/step-109=171s`,
  `#32/step-206=171s`, `#67/step-421=179s`
- Detailed phase timings are only available for checkpoints `5-12`, but even there serialize is
  extremely flat: mean **115.2s**, range **114-117s**. Commit is noisier but bounded: mean **68.2s**,
  range **55-81s**.
- The important qualitative read is unchanged: the clean run shows **oscillating latency spikes**,
  not the **monotonic phase-by-phase degradation** seen in the debug `r2` run.

Full clean-run checkpoint table through checkpoint 73:

| # | Step | Save Start | Saved | Commit Start | Serialize | Commit | **Total** |
|---|---|---|---|---|---|---|---|
| 1 | 4 | 04:12:28 | 04:14:39 | 04:13:38 | 70s | 61s | **131s** |
| 2 | 10 | 04:23:49 | 04:26:57 | 04:25:42 | 113s | 75s | **188s** |
| 3 | 16 | 04:35:44 | 04:39:33 | 04:38:22 | 158s | 71s | **229s** |
| 4 | 23 | 04:48:55 | 04:52:29 | 04:51:15 | 140s | 74s | **214s** |
| 5 | 30 | 05:02:33 | 05:05:36 | 05:04:30 | 117s | 66s | **183s** |
| 6 | 36 | 05:15:07 | 05:18:23 | 05:17:02 | 115s | 81s | **196s** |
| 7 | 42 | 05:27:48 | 05:30:38 | 05:29:42 | 114s | 56s | **170s** |
| 8 | 49 | 05:40:48 | 05:44:02 | 05:42:42 | 114s | 80s | **194s** |
| 9 | 56 | 05:54:07 | 05:57:05 | 05:56:01 | 114s | 64s | **178s** |
| 10 | 63 | 06:07:07 | 06:10:16 | 06:09:03 | 116s | 73s | **189s** |
| 11 | 70 | 06:19:46 | 06:22:52 | 06:21:41 | 115s | 71s | **186s** |
| 12 | 77 | 06:33:39 | 06:36:31 | 06:35:36 | 117s | 55s | **172s** |
| 13 | 84 | 06:47:58 | 06:51:05 | - | - | - | **187s** |
| 14 | 90 | 07:00:55 | 07:03:56 | - | - | - | **181s** |
| 15 | 96 | 07:14:12 | 07:17:24 | - | - | - | **192s** |
| 16 | 102 | 07:26:22 | 07:29:17 | - | - | - | **175s** |
| 17 | 109 | 07:39:00 | 07:41:51 | - | - | - | **171s** |
| 18 | 116 | 07:52:42 | 07:55:48 | - | - | - | **186s** |
| 19 | 123 | 08:05:50 | 08:08:54 | - | - | - | **184s** |
| 20 | 129 | 08:18:28 | 08:21:56 | - | - | - | **208s** |
| 21 | 135 | 08:30:56 | 08:34:07 | - | - | - | **191s** |
| 22 | 142 | 08:43:57 | 08:46:58 | - | - | - | **181s** |
| 23 | 149 | 08:58:05 | 09:01:23 | - | - | - | **198s** |
| 24 | 155 | 09:10:58 | 09:14:16 | - | - | - | **198s** |
| 25 | 161 | 09:23:38 | 09:26:52 | - | - | - | **194s** |
| 26 | 167 | 09:35:51 | 09:39:12 | - | - | - | **201s** |
| 27 | 174 | 09:49:14 | 09:52:31 | - | - | - | **197s** |
| 28 | 180 | 10:01:13 | 10:04:07 | - | - | - | **174s** |
| 29 | 186 | 10:13:13 | 10:16:31 | - | - | - | **198s** |
| 30 | 193 | 10:25:50 | 10:29:08 | - | - | - | **198s** |
| 31 | 200 | 10:40:10 | 10:44:09 | - | - | - | **239s** |
| 32 | 206 | 10:53:18 | 10:56:09 | - | - | - | **171s** |
| 33 | 212 | 11:06:28 | 11:11:02 | - | - | - | **274s** |
| 34 | 218 | 11:20:44 | 11:25:01 | - | - | - | **257s** |
| 35 | 224 | 11:34:24 | 11:37:50 | - | - | - | **206s** |
| 36 | 230 | 11:47:40 | 11:50:38 | - | - | - | **178s** |
| 37 | 236 | 12:00:46 | 12:04:52 | - | - | - | **246s** |
| 38 | 242 | 12:13:39 | 12:17:00 | - | - | - | **201s** |
| 39 | 248 | 12:27:32 | 12:30:59 | - | - | - | **207s** |
| 40 | 254 | 12:41:03 | 12:44:56 | - | - | - | **233s** |
| 41 | 257 | 13:03:56 | 13:07:21 | - | - | - | **205s** |
| 42 | 264 | 13:16:36 | 13:19:43 | - | - | - | **187s** |
| 43 | 270 | 13:29:11 | 13:33:49 | - | - | - | **278s** |
| 44 | 276 | 13:42:50 | 13:45:38 | - | - | - | **168s** |
| 45 | 282 | 13:54:51 | 13:58:21 | - | - | - | **210s** |
| 46 | 288 | 14:07:22 | 14:10:29 | - | - | - | **187s** |
| 47 | 294 | 14:20:32 | 14:24:20 | - | - | - | **228s** |
| 48 | 300 | 14:33:41 | 14:37:13 | - | - | - | **212s** |
| 49 | 306 | 14:46:32 | 14:50:14 | - | - | - | **222s** |
| 50 | 312 | 15:01:04 | 15:04:09 | - | - | - | **185s** |
| 51 | 318 | 15:13:14 | 15:16:27 | - | - | - | **193s** |
| 52 | 325 | 15:28:05 | 15:31:08 | - | - | - | **183s** |
| 53 | 332 | 15:42:15 | 15:45:38 | - | - | - | **203s** |
| 54 | 338 | 15:55:02 | 15:58:47 | - | - | - | **225s** |
| 55 | 344 | 16:08:33 | 16:11:49 | - | - | - | **196s** |
| 56 | 351 | 16:22:10 | 16:25:22 | - | - | - | **192s** |
| 57 | 358 | 16:35:28 | 16:38:39 | - | - | - | **191s** |
| 58 | 364 | 16:48:07 | 16:51:51 | - | - | - | **224s** |
| 59 | 371 | 17:01:59 | 17:05:29 | - | - | - | **210s** |
| 60 | 377 | 17:14:17 | 17:17:22 | - | - | - | **185s** |
| 61 | 384 | 17:27:29 | 17:30:50 | - | - | - | **201s** |
| 62 | 390 | 17:40:31 | 17:43:35 | - | - | - | **184s** |
| 63 | 396 | 17:53:14 | 17:56:49 | - | - | - | **215s** |
| 64 | 403 | 18:07:12 | 18:10:51 | - | - | - | **219s** |
| 65 | 409 | 18:19:22 | 18:23:36 | - | - | - | **254s** |
| 66 | 415 | 18:33:30 | 18:37:26 | - | - | - | **236s** |
| 67 | 421 | 18:46:56 | 18:49:55 | - | - | - | **179s** |
| 68 | 428 | 19:01:06 | 19:04:44 | - | - | - | **218s** |
| 69 | 434 | 19:13:44 | 19:17:15 | - | - | - | **211s** |
| 70 | 441 | 19:27:21 | 19:31:35 | - | - | - | **254s** |
| 71 | 447 | 19:41:20 | 19:44:50 | - | - | - | **210s** |
| 72 | 453 | 19:54:49 | 19:58:09 | - | - | - | **200s** |
| 73 | 459 | 20:07:21 | 20:10:56 | - | - | - | **215s** |

Exact old pre-debug trainer checkpoint table from Iris logs:

- Source: trainer logs for
  `/ahmed/iris-rl-e4ms2-500-0327-v2/rl-iris-rl-e4ms2-500-20260328-031315-train`, filtered into
  [iris_rl_e4ms2_500_train_filtered.log](/tmp/iris_rl_e4ms2_500_train_filtered.log)
- This is **exact Iris trainer chronology**, not the earlier lower-bound W&B stall inference.
- Parsed checkpoint-attempt status counts:
  - `77` successful saves
  - `4` `poll_error` checkpoint attempts
  - `1` `restarted_before_saved` checkpoint attempt at step `143`
- Successful old-run saves only:
  - mean **202.2s**
  - median **199s**
  - range **147-280s**
- Old-run successful save decomposition:
  - `start -> step_done`: mean **145.1s**, median **141s**, range **95-197s**
  - `step_done -> end`: mean **57.1s**, median **55s**, range **22-121s**
- Successful-save means by retry segment were all in the same band:
  - initial segment: **199.2s**
  - post-step-60 resume: **199.8s**
  - post-step-136 resume: **204.0s**
  - post-step-274 resume: **198.5s**
  - post-step-399 resume: **210.8s**
- Comparison against the clean run through checkpoint `73`:
  - old successful saves: mean **202.2s**, median **199s**, range **147-280s**
  - clean successful saves: mean **201.9s**, median **198s**, range **131-278s**
  - so the baseline checkpoint cost was already essentially the same before checkpoint debug mode
  - the difference is robustness: the old run had **5 interrupted/incomplete checkpoint attempts** across
    trainer retries, while the clean run had **0 checkpoint failures** through checkpoint `73`
- Important caveat for the table below:
  - row `24` (`step 143`) is **not** a real checkpoint wall time
  - it started and finished the training step, but never logged `Saved` or `PollForError`
  - its `2272s` total simply spans from checkpoint start until the later restart loaded `step-136`

Full old pre-debug trainer checkpoint table from Iris logs:

| # | Attempt | Resume | Step | Status | Start | Step Done | End | Total | Start->Done | Done->End |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | - | 7 | saved | 03:33:56 | 03:35:49 | 03:36:44 | **168s** | 113s | 55s |
| 2 | 0 | - | 14 | saved | 03:46:46 | 03:49:04 | 03:50:03 | **197s** | 138s | 59s |
| 3 | 0 | - | 20 | saved | 03:59:20 | 04:01:20 | 04:02:18 | **178s** | 120s | 58s |
| 4 | 0 | - | 27 | saved | 04:12:20 | 04:14:57 | 04:15:52 | **212s** | 157s | 55s |
| 5 | 0 | - | 34 | saved | 04:26:43 | 04:29:33 | 04:30:21 | **218s** | 170s | 48s |
| 6 | 0 | - | 41 | saved | 04:42:33 | 04:44:49 | 04:45:34 | **181s** | 136s | 45s |
| 7 | 0 | - | 47 | saved | 04:54:38 | 04:57:01 | 04:58:18 | **220s** | 143s | 77s |
| 8 | 0 | - | 53 | saved | 05:07:44 | 05:10:05 | 05:10:49 | **185s** | 141s | 44s |
| 9 | 0 | - | 60 | saved | 05:21:23 | 05:24:02 | 05:25:17 | **234s** | 159s | 75s |
| 10 | 0 | - | 67 | poll_error | 05:35:07 | - | 05:38:02 | **175s** | - | - |
| 11 | 1 | 60 | 65 | saved | 05:53:03 | 05:55:11 | 05:56:20 | **197s** | 128s | 69s |
| 12 | 1 | 60 | 71 | saved | 06:05:48 | 06:07:50 | 06:09:02 | **194s** | 122s | 72s |
| 13 | 1 | 60 | 77 | saved | 06:17:35 | 06:20:33 | 06:21:04 | **209s** | 178s | 31s |
| 14 | 1 | 60 | 83 | saved | 06:30:47 | 06:33:26 | 06:33:48 | **181s** | 159s | 22s |
| 15 | 1 | 60 | 89 | saved | 06:43:44 | 06:46:03 | 06:47:01 | **197s** | 139s | 58s |
| 16 | 1 | 60 | 94 | saved | 06:55:53 | 06:58:41 | 06:59:32 | **219s** | 168s | 51s |
| 17 | 1 | 60 | 99 | saved | 07:09:27 | 07:11:32 | 07:12:41 | **194s** | 125s | 69s |
| 18 | 1 | 60 | 104 | saved | 07:23:18 | 07:25:22 | 07:26:27 | **189s** | 124s | 65s |
| 19 | 1 | 60 | 110 | saved | 07:35:59 | 07:38:04 | 07:39:22 | **203s** | 125s | 78s |
| 20 | 1 | 60 | 116 | saved | 07:48:46 | 07:51:12 | 07:51:49 | **183s** | 146s | 37s |
| 21 | 1 | 60 | 122 | saved | 08:01:25 | 08:04:12 | 08:05:05 | **220s** | 167s | 53s |
| 22 | 1 | 60 | 129 | saved | 08:15:45 | 08:17:55 | 08:18:42 | **177s** | 130s | 47s |
| 23 | 1 | 60 | 136 | saved | 08:29:48 | 08:32:33 | 08:33:42 | **234s** | 165s | 69s |
| 24 | 1 | 60 | 143 | restarted_before_saved | 08:43:23 | 08:46:06 | 09:21:15 | **2272s** | 163s | 2109s |
| 25 | 2 | 136 | 140 | saved | 09:33:13 | 09:35:04 | 09:35:57 | **164s** | 111s | 53s |
| 26 | 2 | 136 | 146 | saved | 09:45:11 | 09:47:13 | 09:48:32 | **201s** | 122s | 79s |
| 27 | 2 | 136 | 152 | saved | 09:58:00 | 10:01:17 | 10:02:30 | **270s** | 197s | 73s |
| 28 | 2 | 136 | 158 | saved | 10:11:28 | 10:14:35 | 10:15:20 | **232s** | 187s | 45s |
| 29 | 2 | 136 | 165 | saved | 10:25:21 | 10:28:13 | 10:29:21 | **240s** | 172s | 68s |
| 30 | 2 | 136 | 172 | saved | 10:40:05 | 10:42:29 | 10:43:13 | **188s** | 144s | 44s |
| 31 | 2 | 136 | 178 | saved | 10:52:39 | 10:55:13 | 10:55:57 | **198s** | 154s | 44s |
| 32 | 2 | 136 | 184 | saved | 11:05:16 | 11:07:17 | 11:08:34 | **198s** | 121s | 77s |
| 33 | 2 | 136 | 191 | saved | 11:18:42 | 11:20:50 | 11:21:41 | **179s** | 128s | 51s |
| 34 | 2 | 136 | 197 | saved | 11:31:49 | 11:33:56 | 11:34:51 | **182s** | 127s | 55s |
| 35 | 2 | 136 | 203 | saved | 11:43:55 | 11:46:04 | 11:46:56 | **181s** | 129s | 52s |
| 36 | 2 | 136 | 209 | saved | 11:56:07 | 11:58:11 | 11:59:07 | **180s** | 124s | 56s |
| 37 | 2 | 136 | 216 | saved | 12:10:19 | 12:12:38 | 12:13:36 | **197s** | 139s | 58s |
| 38 | 2 | 136 | 222 | saved | 12:23:27 | 12:25:37 | 12:26:32 | **185s** | 130s | 55s |
| 39 | 2 | 136 | 228 | saved | 12:36:15 | 12:38:59 | 12:40:14 | **239s** | 164s | 75s |
| 40 | 2 | 136 | 234 | saved | 12:49:14 | 12:51:10 | 12:52:15 | **181s** | 116s | 65s |
| 41 | 2 | 136 | 240 | saved | 13:01:02 | 13:03:38 | 13:04:28 | **206s** | 156s | 50s |
| 42 | 2 | 136 | 245 | saved | 13:13:16 | 13:16:02 | 13:16:46 | **210s** | 166s | 44s |
| 43 | 2 | 136 | 251 | saved | 13:25:53 | 13:28:31 | 13:29:09 | **196s** | 158s | 38s |
| 44 | 2 | 136 | 257 | saved | 13:38:46 | 13:41:07 | 13:41:53 | **187s** | 141s | 46s |
| 45 | 2 | 136 | 262 | saved | 13:50:52 | 13:53:31 | 13:55:32 | **280s** | 159s | 121s |
| 46 | 2 | 136 | 268 | saved | 14:03:56 | 14:06:41 | 14:07:15 | **199s** | 165s | 34s |
| 47 | 2 | 136 | 274 | saved | 14:17:15 | 14:19:29 | 14:20:34 | **199s** | 134s | 65s |
| 48 | 2 | 136 | 281 | poll_error | 14:32:19 | - | 14:34:13 | **114s** | - | - |
| 49 | 3 | 274 | 278 | saved | 14:47:33 | 14:49:16 | 14:50:00 | **147s** | 103s | 44s |
| 50 | 3 | 274 | 283 | saved | 14:58:51 | 15:01:02 | 15:02:04 | **193s** | 131s | 62s |
| 51 | 3 | 274 | 290 | saved | 15:14:18 | 15:16:30 | 15:17:44 | **206s** | 132s | 74s |
| 52 | 3 | 274 | 297 | saved | 15:28:41 | 15:31:48 | 15:32:36 | **235s** | 187s | 48s |
| 53 | 3 | 274 | 304 | saved | 15:43:17 | 15:45:28 | 15:46:41 | **204s** | 131s | 73s |
| 54 | 3 | 274 | 310 | saved | 15:55:50 | 15:58:26 | 15:59:16 | **206s** | 156s | 50s |
| 55 | 3 | 274 | 316 | saved | 16:08:40 | 16:11:23 | 16:12:09 | **209s** | 163s | 46s |
| 56 | 3 | 274 | 322 | saved | 16:21:26 | 16:24:14 | 16:25:04 | **218s** | 168s | 50s |
| 57 | 3 | 274 | 328 | saved | 16:34:17 | 16:36:56 | 16:38:07 | **230s** | 159s | 71s |
| 58 | 3 | 274 | 334 | saved | 16:47:44 | 16:49:49 | 16:51:08 | **204s** | 125s | 79s |
| 59 | 3 | 274 | 340 | saved | 16:59:51 | 17:01:54 | 17:03:17 | **206s** | 123s | 83s |
| 60 | 3 | 274 | 346 | saved | 17:12:13 | 17:14:26 | 17:15:15 | **182s** | 133s | 49s |
| 61 | 3 | 274 | 352 | saved | 17:24:08 | 17:26:35 | 17:27:28 | **200s** | 147s | 53s |
| 62 | 3 | 274 | 358 | saved | 17:37:03 | 17:39:26 | 17:40:20 | **197s** | 143s | 54s |
| 63 | 3 | 274 | 364 | saved | 17:50:35 | 17:53:32 | 17:54:00 | **205s** | 177s | 28s |
| 64 | 3 | 274 | 370 | saved | 18:03:57 | 18:06:27 | 18:07:28 | **211s** | 150s | 61s |
| 65 | 3 | 274 | 376 | saved | 18:18:09 | 18:20:21 | 18:21:08 | **179s** | 132s | 47s |
| 66 | 3 | 274 | 382 | saved | 18:31:20 | 18:33:57 | 18:34:29 | **189s** | 157s | 32s |
| 67 | 3 | 274 | 388 | saved | 18:45:25 | 18:47:45 | 18:48:37 | **192s** | 140s | 52s |
| 68 | 3 | 274 | 394 | saved | 18:57:24 | 18:59:40 | 19:00:47 | **203s** | 136s | 67s |
| 69 | 3 | 274 | 399 | saved | 19:09:41 | 19:11:39 | 19:12:13 | **152s** | 118s | 34s |
| 70 | 3 | 274 | 405 | poll_error | 19:21:50 | - | 19:22:47 | **57s** | - | - |
| 71 | 4 | 399 | 403 | saved | 19:36:02 | 19:37:37 | 19:38:30 | **148s** | 95s | 53s |
| 72 | 4 | 399 | 410 | saved | 19:49:27 | 19:52:19 | 19:53:22 | **235s** | 172s | 63s |
| 73 | 4 | 399 | 416 | saved | 20:02:59 | 20:06:09 | 20:07:26 | **267s** | 190s | 77s |
| 74 | 4 | 399 | 422 | saved | 20:16:50 | 20:19:02 | 20:20:21 | **211s** | 132s | 79s |
| 75 | 4 | 399 | 428 | saved | 20:29:36 | 20:32:37 | 20:33:18 | **222s** | 181s | 41s |
| 76 | 4 | 399 | 435 | saved | 20:43:41 | 20:45:57 | 20:46:45 | **184s** | 136s | 48s |
| 77 | 4 | 399 | 441 | saved | 20:56:08 | 20:58:26 | 20:59:14 | **186s** | 138s | 48s |
| 78 | 4 | 399 | 447 | saved | 21:09:54 | 21:12:04 | 21:13:15 | **201s** | 130s | 71s |
| 79 | 4 | 399 | 453 | saved | 21:22:55 | 21:25:19 | 21:26:43 | **228s** | 144s | 84s |
| 80 | 4 | 399 | 459 | saved | 21:35:57 | 21:38:33 | 21:39:08 | **191s** | 156s | 35s |
| 81 | 4 | 399 | 464 | saved | 21:48:16 | 21:51:25 | 21:52:22 | **246s** | 189s | 57s |
| 82 | 4 | 399 | 470 | poll_error | 22:01:59 | - | 22:05:37 | **218s** | - | - |

Next steps for this run:

1. Keep the tightened monitor running to terminal state; the important remaining unknown is the finish, not checkpoint behavior through mid-run.
2. If this run finishes 500 cleanly, treat the main question as largely answered: heavy debug instrumentation was the dominant cause of the pathological slowdown, and callback deletion is at most a secondary contributor.
3. After a clean finish, run the smallest clean A/B needed to isolate the deletion flag itself:
   `debug_checkpointer=False`, `delete_previous_temporary_checkpoint_after_save=True`.
4. Do not re-enable forced GC / tracemalloc / thread dumps on the mainline 500-step path unless a new regression appears.
5. If a late failure still appears, compare it against the now-complete checkpoint table and ask whether it correlates with a new post-step-351 phase change rather than the old debug-run pattern.
