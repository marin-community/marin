# Tune Megatron and the asynchronous RL loop

This guide preserves the dated measurements in [MarinSkyRL #873](https://github.com/marin-community/MarinSkyRL/pull/873). Rates and qualification limits describe those source revisions. Use the current [Marin SkyRL artifact launcher](skyrl-migration.md) and validate its resolved recipe before running a historical layout.

[Choose the task](skyrl-performance.md), then measure the learner and the full RL loop. BF16 Snowball/Hero evidence covers CoreWeave H100 and GB200/B200, checked on 30 September 2026. Learner-only rates exclude supply, scoring, publication and checkpointing.

**Hero learning is qualified at 4K on non-agentic GSM8K with route replay, frozen query bias and MuonH/AdamH/Adam.** Synthetic long-context AdamW and short staleness-two probes do not extend that qualification. [Learning audit](https://github.com/marin-community/MarinSkyRL/issues/737#issuecomment-5860256429).

## Choose a learner

The fixed bank contains 128 sequences and 116,351 unpadded prompt + response tokens. Rates use the median training time of three warm updates and learner GPU count. HBM is peak allocated GiB across ranks/updates; host memory is peak cgroup GiB per node. Device utilization/MFU was not measured.

TP, PP, EP, CP and DP mean tensor, pipeline, expert, context and data parallelism.

| Model / hardware; measured 26–28 Sept | Learner GPUs; TP / PP / EP / CP | Warm s/update; sequence tok/s/GPU | Peak HBM / host GiB | Supported starting decision |
| --- | --- | --- | --- | --- |
| Snowball H100 | 32; 1 / 2 / 8 / 1 | 5.57; 653 | 50.4 / 120.4 | Efficient tested fixed-bank point; AdamW. [Evidence](#learner-evidence) |
| Snowball GB200 | 16; 1 / 2 / 8 / 1 | 10.45; 696 | 81.6 / 111.0 | Efficient tested point. 32 GPUs takes 6.16 s but 17.9% more GPU-seconds. [Evidence](#learner-evidence) |
| Hero H100 | 96; 1 / 24 / 4 / 1 | 19.97; 60.7 | 75.0 / 449.9 | Fixed-bank efficiency option; **online batch256 OOMed**. Use the measured 128-GPU online layout for the large-batch example. [Evidence](#learner-evidence) |
| Hero GB200 | 48; 1 / 12 / 4 / 4 | 60.36; 40.2 | 128.6 / 445.1 | Efficient tested MuonH point. 64 GPUs at PP16 takes 52.13 s but 15.1% more GPU-seconds. [Evidence](#learner-evidence) |

Use PP to fit layers/state, EP to shard experts, CP to reduce sequence memory and DP to replicate work. Read rank groups from the launch plan: attention/expert DP differ, and Hero can fold EP across CP/DP. The dimensions do not simply multiply to GPU count. Snowball32 at TP1/PP2/CP1 has attention DP16 and expert DP2; Snowball16 has DP8 and expert DP1. Use the provider's valid head and local/global attention geometry. [Grug training](https://github.com/marin-community/MarinSkyRL/blob/3c9ade6b83b2ac22d7720d8c16b26d265da77a96/docs/grug-megatron-training.md); [Megatron parallelism](https://docs.nvidia.com/nemo/megatron-bridge/latest/parallelisms.html).

Match the tested packing, recomputation and optimizer offload settings. Keep micro-forward = micro-train = 1 and set sequence/group limits. [snowball_megatron_full.yaml](https://github.com/marin-community/MarinSkyRL/blob/3c9ade6b83b2ac22d7720d8c16b26d265da77a96/cloud/iris/configs/snowball_megatron_full.yaml) shows source-revision fields but uses a different batch and serving allocation from the worked example. Validate the resolved launcher config.

## Find the full-loop limit

Measure at least two warm ordinary cycles; use more for variable tails. Report startup/evaluation/save separately and include them in task cost. Old `timing/*` spans overlap, so do not sum them. Current `timing/step_wall/*` spans partition the step.

| Observation | First hypothesis | One next change; what would confirm it |
| --- | --- | --- |
| Learner waits; servers have low running counts, low KV, no queue | Group tails, tool waits, admission, scoring or publication starving supply | Inspect group, lease and pause windows. Adjust the limiting admission/worker bound within the allowed age; require higher accepted tokens/GPU-hour without worse quality/drift |
| Learner waits; servers queue and preempt | Serving/cache pressure | Use the [vLLM guide](skyrl-performance-vllm.md#read-the-signals-together). Try one cap or fleet change; lower wait with higher accepted work confirms it |
| Training dominates; memory fit is tight | Learner layout/activation cost | Compare compatible PP/EP/CP or recomputation on the same bank. Require fewer GPU-seconds and memory headroom; offload can move cost to host transfers |
| Scoring dominates | Reference/replay or mesh forward cost | Compare same tokens/routes/weights and measure full-loop benefit; changing routes can fail the numerical gate |
| Publication dominates | Pause, optimizer offload, rank alignment, transfer, reload or drain | Time publication stages separately. Broadcast time alone cannot identify bandwidth; require exact installed weights and an accepted-rate gain |
| Few groups carry mixed rewards; raw reward high but truncation grows | Useful learning work or termination is failing | Inspect complete answers, verifier and informative-group fraction before buying GPUs. Historical Snowball raw reward hid nontermination. [Evidence](https://github.com/marin-community/marin/issues/8936#issuecomment-5581680479) |

### Telemetry you can use

Select the cluster, run and window in [RL Post-training (async)](https://grafana.oa.dev/d/marin-async-rl). See [run selection](https://github.com/marin-community/MarinSkyRL/blob/3c9ade6b83b2ac22d7720d8c16b26d265da77a96/docs/grafana-rl-runs.md) and [metric definitions](https://github.com/marin-community/MarinSkyRL/blob/3c9ade6b83b2ac22d7720d8c16b26d265da77a96/docs/design/async-rl-telemetry.md). Run FineLog from Marin with normal Iris authentication; discover the schema before adapting SQL:

```bash
uv run finelog namespaces cw-us-east-08a
uv run finelog schema cw-us-east-08a telemetry_v1.marinskyrl
uv run finelog query cw-us-east-08a --format table < bounded-query.sql
```

Use `marin` for federated queries or the job's region for recent local data. Compute deltas per run, execution, cluster and physical engine/rank before aggregating. Missing or delayed rows cannot establish idle hardware. [FineLog access and query rules](https://github.com/marin-community/marin/blob/main/lib/finelog/OPS.md).

| Signal / source | Unit and window | Confirmation / contrary evidence |
| --- | --- | --- |
| `async/performance/consumed_loss_tokens` / `cycle_seconds`, step tracker `training_metric_value` | Accepted masked response tokens divided by complete ordinary wall seconds, 2+ warm cycles, then all fleet GPUs | More useful work per GPU-hour at acceptable quality confirms the fleet choice; generated tokens alone do not |
| `wait_for_generation_buffer`, `run_training`, `fwd_logprobs_values_reward`, `sync_weights` phase windows | Seconds and fraction of the same ordinary window | Identifies what to inspect. Overlapping timers do not add to the cycle; use exclusive walls on current code |
| `rollout_groups`, `rollout_group_tokens` by disposition; `rollout_queue_depth` / capacity | Complete groups/tokens and current groups; per update and 5 min | More consumed work with fewer stale/fully-masked groups supports supply tuning; abandoned work raises GPU cost |
| `rollout_staleness_steps` / `consumed_staleness` | Learner step minus generating weight version, per consumed group/token and update | Age within the approved bound supports admission safety; compare quality and mismatch across ages |
| `policy/mismatch/staleness0/*`, pooled absolute log-ratio tails; `policy/ppo_clip_ratio` | Same-token `log p_trainer - log p_vLLM`, quantiles and clipped token fraction per update | Age-zero isolates numerical mismatch; pooled values also contain policy drift. Growth after tuning contradicts numerical safety |
| `cuda_memory_observation`; Iris task cgroup memory | Allocated/reserved/free bytes around each policy phase; host MiB, GiB = MiB/1,024; peak per rank/node | Headroom through publication/backward confirms fit; NVML minus PyTorch reserved can expose non-PyTorch memory |
| Forward/backward scheduler, optimizer, communication/barrier rank spans | CPU dispatch wall seconds per warm update; corroborate with GPU trace when needed | Large waits suggest imbalance/communication; dispatch spans overlap and cannot prove kernel compute utilization |
| Reward, stop reason, length limits, retained evaluations | Complete-answer counts, truncation fraction and held-out score at fixed evaluation settings | Useful answers maintained across seeds support a change; raw reward alone does not establish quality |

<details>
<summary>A bounded FineLog server query to join with learner waits</summary>

Hero GB200 step45 waited 408.02 seconds while running requests declined, queues stayed near zero and KV stayed low. That weakens a saturated-server explanation. Adapt the run, window and region after checking your schema.

```sql
SELECT date_bin(INTERVAL '1 minute', to_timestamp_millis(timestamp_ms)) AS minute_utc,
       name, count(*) AS samples, avg(value) AS mean_per_engine,
       max(value) AS peak_per_engine
FROM "telemetry_v1.marinskyrl"
WHERE run_id = 'hero-learning-gb200-01a0dca4-a3'
  AND timestamp_ms >= 1790526804761 AND timestamp_ms < 1790527212781
  AND name IN ('num_requests_running', 'num_requests_waiting', 'kv_cache_usage_perc')
GROUP BY minute_utc, name
ORDER BY minute_utc, name
```

For cumulative vLLM counters, scan one sample before the window, take `LAG(value)` per full series, discard negative reset deltas and divide summed deltas by elapsed seconds. Sum native Rigging `work_completed` deltas directly. These gauge averages describe the fleet; inspect each engine's maximum for skew or KV pressure.

</details>

## Worked task: short Snowball math RL on H100

For short single-turn GSM8K with eight responses per group, evaluate **32 learner + 8 serving H100s (P32/I8)**. Optimize accepted work per all GPU-hours while checking fixed-token alignment and completed answers. This is a [26 September measurement](https://github.com/marin-community/MarinSkyRL/issues/737#issuecomment-5848726769).

**Historical launcher gap, checked at the source revision.** At `14597ea844`, [async_rl.py](https://github.com/marin-community/marin/blob/14597ea8441ff629cf936db583f571b67391d5fe/experiments/post_training/async_rl.py#L84-L87) fixes 128 prompts × four answers and rejects role-plan changes through `--set`. It emits `entrypoint: fully_async` and `trainer.fully_async`; the [pinned runtime](https://github.com/marin-community/marin/blob/14597ea8441ff629cf936db583f571b67391d5fe/lib/marin/src/marin/external_dependencies.py#L83-L89) accepts `standard` with `trainer.rollout_buffer`. That historical pairing failed preflight. Current Marin artifacts use the vendored standard runtime and rollout buffer. Author and validate a current recipe and role allocation before using 256 × eight; the measurements below do not qualify that new recipe.

1. Match the Snowball Grug-67B-A2B BF16 layout:

   - Learner: four eight-H100 nodes, TP1/PP2/EP8/CP1, packed samples, equal microbatches of one, AdamW at 1e-6, frozen query bias and optimizer offload during rollouts.
   - Server: one eight-H100 node, TP1/PP1/DP8/EP8, non-eager batch-invariant vLLM, 1,024 sequences / 4,096 batched tokens, memory utilization 0.90 and prefix cache off.

2. Match the task: at most 256 prompt + 3,840 response tokens = 4,096 total; 256 groups × eight responses/update. Historical admission used 512 workers, 256 buffered groups and staleness one. Current `trainer.rollout_buffer` uses `max_staleness_steps`, `batch_policy` and `max_in_flight`, with no direct worker/buffer equivalents. Remeasure supply after migration.
3. Check warm cycles two and three, excluding the cold update and final evaluation. They delivered **1,802,197 accepted loss-masked response tokens / 313.7906 s / 40 GPUs = 143.58 tok/GPU-s**, about 516,898 tok/GPU-hour. Wait was 0.081 and 0.684 s; per-engine median running 7.5 and 7.0; median queues and KV preemptions zero. Keep I8 for this window and inspect training, scoring and publication.
4. Sustained wait **with busy, queued/preempting servers** justifies a cap or I16 comparison using accepted work per all GPU-hours. With idle servers, inspect group tails, admission and pauses first. Earlier I16 saved 5.8% wall time but cost 17% more task GPU-hours; add servers only for a deadline or measured efficiency gain. [Earlier allocation study](https://github.com/marin-community/marin/issues/8936#issuecomment-5581680479).

The run was `/romain/snowball-integrated-h100-01a0dca4-a3`, with MarinSkyRL `0d80999cda9258c7094c720ab66db59f454ee3cc` and vLLM `25f0fc1aae71cccb6da65046cff13145512a6946`. Its config and counters are in `s3://marin-us-east-02a/marin/users/romain/hero-learning-01a0dca4/snowball-integrated-h100-a3/report.json`. Two warm cycles on an older runtime cannot predict total short-run cost; include startup and evaluations.

## Performance and numerical alignment

Hero's fixed-weight gate permits fewer than 0.1% of same-token trainer/serving probability ratios outside [0.8, 1.2]. Replay the captured routes with identical tokens, positions, masks and weights. Native-route/sentinel timing banks cannot qualify numerical agreement or learning. [Numerical decision](https://github.com/marin-community/MarinSkyRL/issues/737#issuecomment-5835032707).

| Choice | Measured result | Starting decision / remaining gap |
| --- | --- | --- |
| Captured versus native expert routes | H100 5 versus 114 outliers / 50,962 tokens; native fails the gate. Replay scoring 5.147 versus 4.146 s when native replay controller is removed | Keep replay. Controller removal is an additional change; this is scoring cost, not whole-RL cost. [Matched study](https://github.com/marin-community/MarinSkyRL/issues/737#issuecomment-5848726769) |
| Learner batch invariance OFF | H100 fixed-bank warm update 18.538→18.473 s; both captured-route gates pass | Keep ON for the qualified online path; safety/accepted throughput of OFF is unknown. Same source study |
| GB200 learner fused→flash | Warm 63.303→61.180 s (3.4%); same-token replay gates both pass | A learner-only option; online safety and accepted throughput unmeasured. [Evidence](#learner-evidence) |
| Publication NCCL bundle OFF | GB20048 fixed bank: ON costs 6.7% learner time; OFF has no safe online publication result. H100 OFF failed at 96 and 128 GPUs | Keep ON on tested H100 layouts. Matched 128-GPU OFF step0 used 16.59 GiB more non-PyTorch memory; no warm OFF rate. [Evidence](#learner-evidence) |
| S=1→S=2 GB200 admission package | Workers 32→48 and buffer16→32 also change; accepted rate 75.46→89.56 tok/s over 56 GPUs | Short combined-setting result, not a staleness-only or long-run learning claim. [Online evidence](#online-evidence) |
| GB200 broadcast→expert_block | 89.56→153.05 accepted tok/s; four-cycle publication 737.93→117.27 s. Fixed-weight, sampled readback and separate full byte replay passed | Trial revisions differ from draft #860's combined branch; GPU validation there is still required. Responses differ by 5.44%; +70.9% is observed, not a universal transfer speedup. [Online evidence](#online-evidence) |

Recomputation saves activation memory through extra compute; optimizer offload uses host memory and transfer time. Keep forward/train microbatches equal because changes can alter kernels/numerics. Check training and publication memory, checkpoint compatibility, fixed-token mismatch and online clipping before accepting a speed gain. [Current Grug mechanism](https://github.com/marin-community/MarinSkyRL/blob/3c9ade6b83b2ac22d7720d8c16b26d265da77a96/docs/grug-megatron-training.md).

### Learner evidence

<details>
<summary>Fixed-bank reports and qualification limits</summary>

Bank SHA256: `6d97026e4235babacdfe1d04d4e53d9e7deef4c68e1c96a6be73a12ced09c896`. Prefix **H** = `s3://marin-us-east-02a/marin/users/romain/hero-learning-01a0dca4/`; **P** = `s3://hero-checkpoints/marin/users/romain/hero-perf-01a0e406/`. Append the report key; CoreWeave access is required. Reports contain configs, source and input digests.

| Point | Iris job; source SHA | Report key; SHA256 |
| --- | --- | --- |
| Snowball H10032 | `/romain/snowball-learner-h100-32g-01a0dca4-a1`; `fa7059545a8a6ec62456fda3bdb9ce05dae103f8` | H `snowball-learner-h100-32g-a1/report.json`; `6f0e7faac71c545eb58395846b316b7fe7b988ce007b05e10271227c0fbb5b74` |
| Hero H10096 | `/romain/hero-learner-h100-96g-01a0dca4-a2`; `80907a6de65c49cee29ca3b13c2fee27d2a05556` | H `hero-learner-h100-96g-a2/report.json`; `7d20654dda9bcb4435459e89a895806e89958de778aa4a3dd783e60bf9bf1e73` |
| Hero GB20048/64 | `/romain/hero-gb200-learner-48g-01a0e406-a1` / `64g` counterpart; `ca91727c765e6f79399b8844b195108cc78a6ee1` | P `gb200-learner-48g-a1/report.json` / `gb200-learner-64g-a1/report.json`; hashes `14078f3e73dc17ab109a717df74fe0af769a39dfd137ef4e8f06a1597fcd611b` / `5fc6aa81919460a21e4576e00abdc8aa71500ca965a6545913eba2518260ead8` |
| Snowball GB20016/32 | `/romain/gb200-snowball-learner-16g-01a0e406-a1` / `32g` counterpart; same GB200 source | P `gb200-snowball-learner-16g-a1/report.json` / `gb200-snowball-learner-32g-a1/report.json`; hashes `dc5ad08a13cdd37b9a9982a3c45d56cb8696edf9e52fc450e126a43a3eccd63f` / `d926dea95c1b497e58161c91f4e498361a78b451c3b1fe474ca04f2fda2a774f` |
| GB200 fused/flash | `/romain/hero-gb200-tim-fused-48g-01a0e406-a2` / `/romain/hero-gb200-tim-flash-48g-01a0e406-a1`; same fixed bank | P `gb200-tim-fused-48g-a2/report.json`, `gb200-tim-flash-48g-a1/report.json`; SHA256 `fc76835736d86baacbcc1f8bfc0d38c973eb4bc72db0dbb2b5e362bbecfeca51` / `a482ca6521a353cd511dd99cce4721850f52d6c5204041dcf48a95c9b3a64405` |
| GB200 NCCL ON/OFF | `/romain/hero-gb200-nccl-on-48g-01a0e406-a2` / `/romain/hero-gb200-nccl-off-48g-01a0e406-a1`; source `1bc2affa3d54da08285faed2b2750785082f3272` | P `gb200-nccl-on-48g-a2/report.json`, `gb200-nccl-off-48g-a1/report.json`; SHA256 `d87a98d01940c2b4091c5c089756e604c093efda7b3879d2fa3450c993aa735c` / `6cf46d33907e72c5977b7945156abce01e7341c8ea042e6693c7f00144352956` |
| H100128 NCCL ON/OFF | same 16 nodes; OFF failed backward after step0 | H100 performance prefix `s3://marin-us-east-02a/marin/users/romain/hero-perf-01a0e406/`, keys `h100-nccl-on-128g-a1/report.json` / `h100-nccl-off-128g-a1/report.json` |

Hero uses MuonH/AdamH/Adam; Snowball uses AdamW. Warm training excludes scoring, offload outside the timer, serving and publication. Rank dispatch spans overlap. One placement and three warm updates cannot establish minimum fleet size or variability.

NCCL flags were tested as one bundle: `NCCL_LAUNCH_MODE=GROUP`, `NCCL_COLLNET_ENABLE=0`, `NCCL_NVLS_ENABLE=0`, `NCCL_P2P_NET_DISABLE=1`, `NCCL_MIN_NCHANNELS=1`, `NCCL_MAX_NCHANNELS=1`, `NCCL_PROTO=Simple`, `NCCL_ALGO=allreduce:tree`, `NCCL_NTHREADS=1`, `NCCL_SOCKET_NTHREADS=1`. The ON/OFF switch belongs to the experiment; individual flag effects are unknown.

</details>

### Online evidence

<details>
<summary>Complete-cycle measurements and source integration status</summary>

| Fleet | Warm accepted response work / complete cycle wall | Bound and durable source |
| --- | --- | --- |
| Hero H100128+32, steps2–5 | 8,859,198 / 4,975.25 s = 1,780.65 tok/s, 11.13 per all GPUs | `/romain/hero-integrated-h100-01a0dca4-a8`, source `3d455ebf648fb3a58764f20afa5923a7afffa0c5`; H prefix `h100-integrated-batch512-noprof-a8/report.json`. Batch512×8, staleness1, cap64/512. Wait14.9%, full training53.1%, publication18.2% |
| Hero GB20048+16, 29 ordinary updates | 1,429,495 / 19,419.66 s = 73.61 tok/s | `/romain/hero-learning-gb200-01a0dca4-a3`, source `d9c9997ea13c314fd2ddd2c698b8d2d50320e2a0`; `s3://hero-checkpoints/marin/users/romain/hero-learning-01a0dca4/gb200-learning-a3/report.json`; SHA256 `e7e6e7cc6b8d8938e21cd6700f426ee7ca1815d6a864151dcc8aed04277c079a`. Batch16×8; wait57.9%, publication28.3% |
| Hero GB20048+8 S=2 cap16, steps34–37 | 161,621 / 1,804.63 s = 89.56 tok/s, 1.599 per all GPUs | `/romain/hero-gb200-feed-48p8s-s2-01a0e406-a1`, source `8d71544751a41d978bb476a0c49e9e30af21921a`; P prefix `gb200-feed-48p8s-s2-a1/report.json`. Batch16×8, workers48, buffer32. Maximum clip0.02441%, age2; wait40.6%, publication40.9% |
| Same fleet, expert_block | 170,418 / 1,113.45 s = 153.05 tok/s | `/romain/hero-gb200-expert-block-48p8s-s2-01a0e406-a4`, source `e26957594e84c67c88888b7c0653100905fb3d77`; P `gb200-expert-block-48p8s-s2-a4/report.json`; SHA256 `d3eab5bdde398985bb49f5c1775c8c096debd71d25fd10e9bb2c1e776884d346`. Max clip0.02955%; sampled weights exact |

Use the P/H prefixes above. A4 passed five updates and final evaluation; after its head pod was deleted, Iris completion was marked manually. Separate `/romain/hero-gb200-expert-block-verify-48p8s-s2-01a0e406-a6` succeeded 14/14 and checked full receiver-parameter bytes plus trainer DP replicas at step32, with no updates or evaluation. Source `97bae9fbf285d95caa569a01d17065d0435c34e9`; P `gb200-expert-block-verify-48p8s-s2-a6/report.json`; SHA256 `ca5c3022ecf6bf777a6b0bf66dc3db402e0428bd57350a84337b9f6921fdf65e`. Per-receiver byte counts were not persisted.

The online cap comparisons are `s3://marin-us-east-02a/marin/users/romain/hero-perf-01a0e406/h100-online-128seq-a2/report.json` and P `gb200-feed-48p8s-s2-maxseq32-a3/report.json`. The H100 retry also raised the pause deadline from 120 to 300 s; four warm steps with different generated outputs limit causal inference.

On 30 September, [Hero architecture #792](https://github.com/marin-community/MarinSkyRL/pull/792) and [vLLM #77](https://github.com/marin-community/vllm/pull/77)/[#78](https://github.com/marin-community/vllm/pull/78) were merged. [Runtime wheels #799](https://github.com/marin-community/MarinSkyRL/pull/799), [expert-block Hero integration #860](https://github.com/marin-community/MarinSkyRL/pull/860), and [sparse publication #701](https://github.com/marin-community/MarinSkyRL/pull/701) remained drafts. Their combined runtime still needs GPU qualification.

</details>

## Measurements that could change a recommendation

| Gap | Narrow proposed probe | Cost and decision value |
| --- | --- | --- |
| Validate rollout-buffer mapping for the Snowball worked task | Same 4K bank, P32/I8, two warm cycles plus fixed-token gate; inspect resolved admission semantics | 40 H100s × roughly 30 min = 20 GPU-hours, including estimated startup. Checks current supply/accepted rate; record actual cost |
| Hero expert-block combined branch | Step32 restore, exact full-byte publication replay, then four ordinary matched cycles on P48/I8 | 56 GB200s × roughly 1 hour = 56 GPU-hours, including restore/eval. Checks release readiness and integrated publication savings |
| Hero/Snowball isolated GB2008 serving | Fixed 4K bank at two client loads on a reliably constrained rack; verify placement before loading | 8 GPUs × roughly 30 min = 4 GPU-hours plus startup variance. Checks whether 16 GPUs can shrink to eight; failed placements gave no sizing data |
| Hero long-response or multi-turn >4K RL | First qualify fixed-token routes/positions at one intended length, then a bounded complete task with tools/tails | Cost unknown: learner fit/runtime untested. Serving 256K short outputs and synthetic 65K AdamW updates cannot predict this rate |

Probe budgets are **estimates**; no work is scheduled. [Synthetic full-Hero capacity](https://github.com/marin-community/MarinSkyRL/blob/e3f5fff039f9b965d3f014584f8abd34164c0381/docs/grug-megatron-training.md#full-hero-capacity) passed 65K save/restore on 256 H100 or 64 GB200 with AdamW. MuonH long-context learner fit, online quality, sustained decode and 128K/256K trainer rates remain unknown.
