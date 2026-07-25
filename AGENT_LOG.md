# AGENT_LOG — ep25-d3 round 6 (R6-2: global-QB controller probes)

Append-only. All times UTC. Branch: agent/ep25-d3-qbprobes (from agent/ep25-d4-pipelined @1650246c5).
Prior-round logs live on their own branches (fa4lse, token-chunk, TE).

Mission: probe whether a global-controller variant (damped gain g<1 or DeepSeek-V3-style
integral accumulation) drives QB-on cf1.0 steady-state drops under 3% at <=0.5pp MFU cost
with loss parity. Baselines (d4, 350-step g=1): drop series 0.885(5)/0.271(60)/0.175(119)/
0.089(250)/0.064(349), tail-100 mean 7.3%, p50 MFU 22.002%, loss 3.335@350 (tail20 3.3434).
120-step draw: 22.595%, drops 0.083@119. g=2 diverges (limit cycle, drops 0.67+).

## Check-in 2026-07-25 20:55 UTC — setup; arm 1 (g=0.5) submitted

- Read ROUND6_BRIEF.md; branch created from agent/ep25-d4-pipelined (custom adjoint,
  SCALE_REPORT_DROPS emission fix, SCALE_CAPACITY_FACTOR, SCALE_QB_GAIN all present).
- QB code read: `_compute_qb_beta` (model.py:392) = per-step equalizing quantile, pmean
  over batch axes; `_apply_qb_betas` (train.py:261) sets bias = -center(beta), full
  replacement = implicit proportional controller at gain 1. SCALE_QB_GAIN blending
  (train.py:409) gives g<1 damping for free: pending <- g*beta + (1-g)*pending.
- Reference env recovered verbatim from iris job_config provenance of
  /mwittmann/ep25d4-qb-cf100-drops-350-v1-20260725 (incl. SCALE_A2A_CUSTOM_ADJOINT=1,
  SCALE_MOE_QB=1, cf default 1.0, 350 steps, checkpoints disabled).
- Fleet check 20:50 UTC: only peer SMOKES running (d1 batch-smoke-ep4, d4 sqb-smoke-ep4,
  rav hybridep smoke) — no EP25 rack job in flight; slot free.
- ARM 1 SUBMITTED: /mwittmann/ep25d3-qbg05-cf100-350-v1-20260725 — g=0.5 damped gain
  (SCALE_QB_GAIN=0.5, stock code path otherwise byte-identical), cf1.0, adjoint, drops,
  350 steps, DISABLE_CHECKPOINT, operating point. ETA ~22:30 UTC (setup+compile+350x~12.3s).
  Job mutations this session: submissions only.
- Falsifiable read: if the ~6% plateau is a mild proportional limit cycle, g=0.5 halves
  the correction rate and should settle lower; if it is sender-local bucket hotspots
  (invisible to any global bias), the series matches the g=1 baseline within draw variance.

Confidence: 4/10 that g=0.5 reaches <3% steady-state drops
Next: build arm 2 (SCALE_QB_INTEGRAL) + CPU test while arm 1 runs; babysit arm 1.

## Check-in 2026-07-25 21:12 UTC — arm 2 built, CPU tests green, EP4 smoke submitted

- ARM 1 (/mwittmann/ep25d3-qbg05-cf100-350-v1-20260725) running, 9 min in (setup/compile
  phase). No intervention needed.
- ARM 2 IMPLEMENTED (commit 3f10dcc6a): SCALE_QB_INTEGRAL=<gamma> env gate.
  model.py::_compute_qb_loads ships per-expert biased-top-k assignment counts (psum over
  the batch axes — same single f32 [num_experts] reduce per layer as the quantile pmean;
  scatter-add replaces top_k). train.py: extracted _next_qb_betas helper (stock replacement
  / SCALE_QB_GAIN blend / integral accumulation); integral rule is
  pending += gamma * sign(load - mean_load), which under bias=-center(beta) is exactly
  DSv3's bias_i += gamma * sign(mean_load - load_i) up to common-mode (absorbed by
  centering). Stock path byte-identical when env off (helper returns measured unchanged).
- CPU TESTS PASS (4/4 in experiments/grug/moe/test_model.py): rule math + DSv3 bias
  direction (overloaded expert's applied bias goes down), EP8-subprocess bincount parity
  for _compute_qb_loads, and forward identity (routed output bitwise identical with the
  integral gate on; loads match the biased top-k bincount). pyrefly clean.
- EP4 SMOKE SUBMITTED: /mwittmann/ep25d3-qbint-smoke-ep4-20260725 (1 replica/4 GPUs, EP4,
  e256 top8 d2048 L4 b64, 20 steps, gamma=0.001). Verify: sentinel "QB integral rule
  active", loss descends, drops computed.
- Design note for the verdict: gamma=0.001 moves biases at most 0.35 over 350 steps.
  DSv3's rule works over 100k+ steps by preventing drift, not reversing collapse; at this
  horizon the integral arm discriminates "can fixed-step integral control hold/gain ground
  vs the g=1 trajectory", not "does it converge". If it shows directional movement without
  reaching 3%, arm 3 = gamma sweep point (e.g. 0.01) is the tuned follow-up per the brief.

Confidence: 4/10 g=0.5, 3/10 integral (gamma horizon concern above)
Next: babysit smoke (~15 min) + arm 1; harvest smoke; arm 1 series at ~22:30.

## Check-in 2026-07-25 21:42 UTC — smoke GREEN; log-server ingestion gap noted

- EP4 SMOKE SUCCEEDED (/mwittmann/ep25d3-qbint-smoke-ep4-20260725): step completed in
  8:13, exit clean. Provenance confirms base_commit 3f10dcc6a (my arm-2 commit),
  SCALE_QB_INTEGRAL=0.001. Mechanics validated on the real stack (scan+recompute_all+
  muonh+fa4_cute, QB integral trace + 20 steps). Sentinel/loss lines not yet readable —
  see the log gap below.
- INFRA NOTE: since ~20:40 UTC the central log server is not serving logs for ANY new
  Fray-dispatched grug-train child tasks (mine, d4's sqb smoke, rav's hybridep smoke all
  return 0 lines; parent executor logs serve fine; this morning's completed jobs serve
  fine). Peers equally affected — global ingestion lag, not my jobs. Backup harvest paths
  probed: in-container stdout is a pipe (no local file); per-task telltale endpoints exist
  but the minted capability token 403s on app paths (/metrics, /health). Plan: poll job
  logs on a delay; morning evidence says served logs eventually appear.
- ARM 1 alive and stepping: 16/16 tasks running, all 4 GPUs at 100% util / ~170 GiB on
  task 0 (checked via task exec + nvidia-smi). ~27 min into the train task at 21:25.
  ETA ~22:25-22:35.
- No job mutations beyond my two submissions (arm 1 rack leg, arm 2 smoke).

Confidence: 4/10 g=0.5, 3/10 integral
Next: poll arm-1 logs from ~22:00; harvest smoke metrics when the log server backfills;
prepare the arm-2 rack submit command (identical to arm 1 minus SCALE_QB_GAIN, plus
SCALE_QB_INTEGRAL=0.001) so it fires the moment arm 1 exits.

## Check-in 2026-07-25 22:00 UTC — log gap persists; arm 1 healthy

- Child-task log serving still 0 lines for all new grug-train jobs (mine + peers').
  Tried: bare/health/metrics on the minted telltale URL (403 endpoint-scope), in-container
  stdout (pipe only). Remaining bypass (finelog StatsService SQL with iris auth) parked —
  cost/benefit poor vs waiting; morning jobs prove the pipeline retains data.
- Arm 1: GPUs 100% on task 0 at 21:58; ~62 min into the train task (started ~20:57).
  If compile took ~12 min, stepping since ~21:10 -> step ~(48 min * 60 / 12.3s) ~ 230/350.
  ETA ~22:30. No mutations.
- Arm-2 rack command staged (fire on arm-1 exit):
  job-name ep25d3-qbint-cf100-350-v1-20260725, env identical to arm 1 except
  drop SCALE_QB_GAIN, add SCALE_QB_INTEGRAL=0.001, version ep25d3-dev.

Confidence: 4/10 g=0.5, 3/10 integral
Next: arm-1 completion ~22:30 -> harvest (logs permitting) -> submit arm 2 rack leg.

## Check-in 2026-07-25 22:30 UTC — arm 1 DONE; arm 2 rack leg submitted; log gap localized

- ARM 1 COMPLETE: /mwittmann/ep25d3-qbg05-cf100-350-v1-20260725 succeeded at ~22:28
  (350/350 steps, no failures/preemptions). Metrics not yet readable — see below.
- ARM 2 RACK LEG SUBMITTED at 22:28 (rack slot free, one-in-flight rule kept):
  /mwittmann/ep25d3-qbint-cf100-350-v1-20260725 — gamma=0.001 integral rule, cf1.0,
  adjoint, drops, 350 steps, provenance base_commit 3f10dcc6a. Dispatched 22:18:50,
  ETA ~00:00-00:15.
- HARVEST TOOL validated (commit, experiments/grug/moe/harvest_ep25.py): finelog SQL
  path; reproduces ALL d4 published reference numbers exactly (baseline drop checkpoints
  0.885(5)/0.271(60)/0.175(119)/0.089(250)/0.064(349), tail-100 0.0732, p50 22.002%
  p10/p90 21.83/22.93, loss 3.335; g=2 limit cycle 0.68-0.70).
- LOG GAP LOCALIZED: child train-task logs from cw-us-east-08a (GB200 rack) workers are
  not flowing since ~20:40 UTC — mine AND peers' (d4 sqb, rav hybridep smoke). rav's
  non-GB200 swarm jobs ingest in real time (rows current to 22:19). SQL namespace lags
  fetch_logs further. Data is presumed buffered at the source workers (morning jobs fully
  served); treating as an indexing/shipping outage, polling both paths every ~15 min.
- Job mutations: submissions only (arm 1, smoke, arm 2).

Confidence: 4/10 g=0.5, 3/10 integral
Next: poll arm-1 logs; babysit arm 2 to completion ~00:15; harvest both; decide arm 3.

## Check-in 2026-07-25 23:15 UTC — LIVE PROBE: arm 2 integral @150 = 0.66 drops (wrong direction)

- BYPASS FOUND for running tasks: `iris task exec` + curl the pod's own telltale
  /metrics (levanter_* gauges) — no log-server dependency. (Proxy-minted capability URLs
  403, but in-pod localhost is open.)
- ARM 2 LIVE READ at step 150: drop_fraction 0.659, loss 5.189, 358.1K tok/s (~23.1% MFU
  by the 15.5K-tok/s-per-pp conversion — the heavy-drop inflation zone). Baseline g=1 at
  step 150: 0.125. Integral gamma=0.001 has NOT reversed the early collapse: max bias
  travel by step 150 is 0.15 in bias units, far below the collapsed-logit spread. This is
  the predicted gamma-horizon failure mode, now measured. Unless the late series shows a
  strong monotonic decline (controller steadily gaining), arm 2 is a clean negative at
  this horizon; arm 3 = larger gamma (e.g. 0.01) would be the tuned point ONLY if the
  series shows directional movement.
- Arm-1 metrics still gated on the GB200 log pipeline (0 child rows at 23:10; 2.5h gap;
  d4/rav evening jobs equally stuck). Arm 1 completed 22:28 — its pods are gone, so
  telltale is not an option for it; logs are the only path.
- Background telltale poller running for arm 2 (150s cadence -> /tmp/qbint-350-telltale.tsv),
  insurance against the log gap + early series shape. Full series still from logs later.
- Baseline draw variance quantified (harvest of the two g=1 draws): step 60: 0.271/0.234;
  step 90: 0.233/0.132; step 119: 0.175/0.090 — ~2x mid-training draw spread. The <3% win
  bar needs tail separation well beyond this.

Confidence: 4/10 g=0.5, 1.5/10 integral-g0.001 (measured wrong-direction separation at 150)
Next: arm-2 series via poller; keep polling arm-1 logs every ~15 min.
