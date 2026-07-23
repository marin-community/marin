# Plan: root-cause and fix the vLLM DP-rank logprob variance (Snowball 67B parity gate)

Status: v2, revised after four independent agent reviews (`.agents/tmp/{CLAUDE,CODEX,GEMINI,KIMI}_PLAN_REVIEW.md`).
Budget cap: 2× 8xH100 nodes concurrent, production priority band.

## TL;DR

The Snowball parity gate's rank sentinel fails because the same prompt returns materially
different next-token distributions depending on which vLLM data-parallel rank serves it:
top-token probability span 0.1699 across 8 ranks at 15,025 tokens (sequential replays;
the one measured concurrent wave spanned 0.076), 0.283 at 8,192, and a single rank moved
0.2788 between launches — far enough that two ranks' greedy tokens flipped. Worst rank vs
golden is 0.1632 against a 0.075 tolerance; neither of the two evaluated references
(frozen golden, fresh Levanter-GPU) passes. Within one captured launch and request history, responses repeat bit-exactly — but
review showed that evidence is confounded by prefix caching (enabled by default), so
whether a fresh recomputation is deterministic is the first thing this plan measures.

Seven gates: G0 desk audit (largely complete — the review cycle verified the key source
facts), G1 model-free collective microreproducer + controlled baseline, G2 layer-boundary
trace + weight audit through the server's dev RPC endpoint (no fork branch needed), G3
targeted interventions on the branch the trace selects, G4 mechanism confirmation on
captured activations, G5 fix (user checkpoint), G6 independent-launch validation with an
explicit tolerance-margin analysis. Worst case ≈ 9 node-hours ≈ 72 H100-hours including
20% contingency; never more than 2 nodes.

Three review findings reshaped v1: the `naive` all2all arm and the TP=8/DP=1 arm were
impossible at the pinned rev (one silently aliases the baseline, the other cannot start);
`VLLM_BATCH_INVARIANT=1` unconditionally pins NCCL env (so reduce-scatter is effectively
always Ring — supporting the leading mechanism while eliminating NCCL selection as the
launch-variance explanation); and a broad knob matrix cannot separate combine-order
effects from local-kernel effects — only a pre-combine activation trace can, so tracing
moved ahead of ablations.

## Problem statement

`test_snowball_export_matches_representative_goldens[vllm-gpu]` replays one long prompt
(`knowledge-longbench-02`, 15,025 tokens) once per DP rank and asserts per-observation
`max_probability_error <= 0.075`
([snowball.py](../../tests/cluster/vllm/snowball.py) line 19,
[backend_parity.py](../../tests/cluster/vllm/backend_parity.py) lines 30–34).

Measured top-token probability per rank (golden p = 0.557848), sequential single-rank
replays at full length in job `snowball-experiment-b-vllm-bd7f6354` — note this is the
replay protocol, not the gate's concurrent wave; the concurrent-wave span in the one
launch where it was measured (Experiment A, different launch) was 0.076:

| rank | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| p(423) | 0.494724 | 0.515144 | 0.473948 | 0.531029 | 0.394602 | 0.514611 | 0.534107 | 0.564495 |

Worst rank error vs golden 0.163246; vs fresh Levanter-GPU 0.107914. The 64-case
representative wave passes reliably (5/5 cold runs, #7354) but scores each case on exactly
one rank, so it cannot see rank spread.

"Resolved" means: (a) mechanism identified with boundary-level evidence, (b) an
intervention lands after which the gate workload is rank-stable and launch-stable,
(c) 3/3 green independent cold launches, (d) findings captured in the standing report.

**Margin constraint.** Both GPU backends sit ~0.055 below the TPU-derived golden on this
prompt. Even a perfect rank/launch fix leaves the sentinel ~0.055 from golden with only
~0.02 headroom under 0.075. Root-causing that offset is out of scope (golden provenance,
tracked in the [report](../../docs/reports/snowball-gpu-golden-logprob-parity.md)), but G6
explicitly measures residual headroom and escalates if it is too thin for a durable gate.

## Established facts

### Observed (measured; job names + raw files in the report and `scratch/LOG-attempts.md`)

| Fact | Evidence |
|---|---|
| Rank span 0.1699 at 15,025 tokens; widest spans at 8,192: 0.283 (unchunked), 0.174 (chunked) | Experiment B, jobs `snowball-experiment-b-vllm-bd7f6354` / `-de845a00`; spans re-derived from raw logs during review |
| Span present at 128 tokens (0.043–0.068) | same |
| Launch movement 0.2788 (rank 0, 8,192 tokens, same flags — `VLLM_BATCH_INVARIANT=1` confirmed in the harness for both) | 0.621766 vs 0.343003 across the two jobs above |
| In the second launch the argmax itself flipped at 8,192: ranks 0 and 3 return greedy 426 (p = 0.388673, 0.408742) — the gate's greedy and gap assertions are implicated, not only `max_probability_error` | re-derived from job `-de845a00` raw log |
| Cross-launch movement grows with length: 6/8 ranks bit-identical across the two launches at 128 tokens, none at 8,192 | re-derived from both raw logs |
| Node identity was not logged for the historical launch pair; the two jobs ran minutes apart and are plausibly same-node (overlapping worker pid ranges, per review), so the movement cannot be attributed to node heterogeneity — nor the opposite. G1's effective-config record closes this gap going forward | Kimi review §fact 8 |
| Within-launch repeats bit-exact after the first request; first request differs (0.461436 then 7× 0.464546) | Experiment A ([_experiment_rank_spread.py](../../tests/cluster/vllm/_experiment_rank_spread.py)). **Cache-confounded**: repeats hit the prefix cache, so this establishes cache-hit stability, not recomputation determinism. A2's rank-0 value equals the repeat value exactly — itself a cache hit. G1 re-measures with caching off |
| Levanter, same-node `data=8` sharding, 7 identical rows bit-identical | Experiment B Levanter arm, job `snowball-experiment-b-levanter-9f283c4b`; one run |
| Experiment B's ladder is cache-clean (left-truncations are suffixes, not prefixes) — the span and launch-movement facts above stand | truncation code in `scratch/experiment_b.py` |
| Gate asymmetry: the sentinel case also runs in the representative wave, so 1 of its 8 sentinel observations is served from prefix cache by construction | [test_snowball_backend_parity.py](../../tests/cluster/vllm/test_snowball_backend_parity.py) lines 268–281 + caching default below |

### Verified at source (marin-community/vllm @ `afb26719`; review-checked)

| Fact | Source |
|---|---|
| Serving config: DP=8, EP on, RunAI distributed loader, `--max-num-seqs 1`, `max_num_batched_tokens=512`, FLASH_ATTN, `VLLM_BATCH_INVARIANT=1` | test lines 53–58, 189–208, 306–319 |
| MoE: 256 experts, top-4, 32 local/rank, `placement=linear`, `all2all=allgather_reducescatter` | `GrugMoE effective config` log line (`grugmoe.py:135`) |
| EP combine = all-gather tokens → local experts over the full gathered batch (non-local rows zeroed) → `ncclReduceScatter` on bf16 tensors; no fixed-order or fp32-output path exists in the model code | `all2all.py:85-123`, `fused_moe.py:1708-1709`, `cuda_communicator.py:338-371`, `pynccl.py:257-320` |
| `VLLM_BATCH_INVARIANT=1` reaches much further than dense ops: it overwrites NCCL env unconditionally (channels=1, `NCCL_PROTO=Simple`, `NCCL_ALGO=allreduce:tree`, NVLS/CollNet/P2P-net disabled → reduce-scatter restricted to Ring), pins the MoE grouped-GEMM config (fixed tile sizes, `SPLIT_K=1`, tuned-config lookup skipped), gates MoE kernel selection to batch-invariant ones, and pins FlashAttention to `num_splits=1` | `batch_invariant.py` `override_envs_for_invariance()`; `fused_moe.py:1031-1033,1212-1219`; `modular_kernel.py:577-578`; `flash_attn.py` (per review) |
| The EP combine has no fixed-order variant at this rev — every available backend sums inside NCCL; a deterministic combine is new code, not a config switch | `all2all.py`, `cuda_communicator.py` |
| NCCL reduces bf16 with fp32 internal accumulation and rounds on store; non-local expert rows are zeroed, so ≤4 of the 8 partials per token are nonzero | `fused_moe.py:415-419`; NCCL semantics (per review; measured directly by the G1 microreproducer) |
| Prefix caching defaults on (`enable_prefix_caching: bool = True`) | `cache.py:92` |
| `POST /collective_rpc` (named method on every worker) and `POST /reset_prefix_cache` exist behind `VLLM_SERVER_DEV_MODE=1` | `entrypoints/serve/dev/{rpc,cache}/api_router.py`, `api_server.py:197-200` |
| `naive`/`pplx` all2all backends were removed — silently rewritten to `allgather_reducescatter` | `parallel.py:441-447` |
| GrugMoE requires TP=1 on this path (`NotImplementedError` otherwise) — a TP-sharded no-EP control is impossible | `grugmoe.py` `_raise_for_unsupported_modes` |
| Idle-rank dummy tokens do enter the all-gather with DP-synchronized padded sizes, and dummy tokens get routed | `dp_utils.py:47-89`, `gpu_model_runner.py:4052-4064` |
| Fork-pin override works at runtime (`vllm_fork_ref()` reads the module global at serve time; `VLLM_USE_PRECOMPILED=1` set for MARIN_FORK) — fallback instrumentation path | `vllm_server.py:25,121`, `tpu_vllm_pins.py` |

### Hypothesized (what the gates test)

Destination-dependent ring reduce-scatter accumulation order; router top-4 flips
amplifying small perturbations across the 26 layers; local-kernel position sensitivity;
the rest of the hypothesis list below.

## Hypotheses

**H1 — EP combine order (leading).** Ring reduce-scatter accumulates each destination
rank's chunk in a rotated rank order, so the same token's expert partials sum in a
different order per destination. The per-op seed is small — fp32 internal accumulation
over ≤4 nonzero partials, with bf16 rounding only on store — so the observed 0.17 span
must be carried almost entirely by discrete router top-4 flips compounding across the 26
layers. That makes magnitude plausibility part of the hypothesis: the microreproducer
must show per-op deltas that, times a plausible flip rate, can reach the observed span,
or H1 is at best incomplete. Narrowed pre-registered predictions:
(i) pre-combine partials identical across ranks while post-combine outputs differ per
destination; (ii) a fixed-order combine collapses the spread while leaving pre-combine
partials unchanged; (iii) the effect follows the logical destination chunk under rank
permutation. Explicit non-predictions (characterization only, not evidence): span growth
with length, monotone span growth with EP width, isolated == concurrent.

**H2 — loader/placement integrity (prior: low, now source-grounded).** The distributed
RunAI path has each tensor read once and broadcast, so bytes are identical across ranks
and casts are elementwise — no per-rank numeric channel, only an integrity channel
(missing/duplicated slice). Corrected test (v1's per-rank checksum comparison was invalid
— ranks own different experts by design): replicated tensors (dense, embeddings, router,
norms, LM head) checksummed by canonical name must agree across ranks; expert tensors
checked as (name, global expert ID, slice) against the authoritative source slice; every
expert loaded exactly once; NaN/uninitialized scan. The `{"distributed":false}` A/B must
stay on the RunAI loader (`--load-format runai_streamer`; the default loader rejects the
key) and budgets a much slower load (every rank streams the full checkpoint).

**H3a — cross-rank composition.** Peer activity changes all-gather sizes and per-expert
segment shapes (verified: dummy tokens are gathered and routed). Tested with a
peer-content matrix, not idle-vs-busy alone.

**H3b — local kernel position/shape sensitivity (prior lowered by review).** After
all-gather, rank r's tokens sit at offset r of the gathered batch; grouped-GEMM tile
boundaries and `moe_sum` could make partials position-dependent before any collective.
The invariance pinning (fixed tiles, `SPLIT_K=1`, one attention split) closes most of this
channel in the gate config, so the pre-registered expectation for isolated-vs-concurrent
is a **null**; a measured difference would point outside the current model and is exactly
what would make it interesting. One verified confound must be recorded per mode: equal
per-rank sizes take a single `ncclReduceScatter` while uneven sizes take a grouped
per-root `ncclReduce` sequence — isolated and concurrent modes may run different combine
variants, and the comparison is only interpretable with the variant logged.
Indistinguishable from H1 by outputs alone — this is why the trace precedes the ablation
matrix.

**H4 — launch-scoped state.** Split into testable sub-axes: CUDA-graph capture/compile/
autotune (arm: `--enforce-eager`), cuBLASLt algorithm selection and workspace state,
Triton JIT state, the even-vs-uneven combine-variant selection, per-job `uvx` dependency
drift (the fork installs fresh from git each job), and physical GPU / rank→device mapping
(recorded via worker probe: GPU UUID, PCI bus ID). NCCL algorithm selection is eliminated
(env pinned — verified above). The historical movement is plausibly a same-node effect
(facts table), so launch state, not node hardware, is the default suspect.

**H5 — request history / warm-path state.** Default-on prefix caching explains the
Experiment A anomalies and contaminates the gate itself (asymmetry fact above); Triton JIT
compilation observed firing seconds before the first measured request is a second
candidate for the first-request offset (0.461436 → 0.464546). Controls: caching off for
measurement arms, `/reset_prefix_cache` where the gate config must stay on, one discarded
warmup request per server start, one gate-exact-history arm for relevance.

**H6 — attention/LM-head numerics.** Kept open; the trace decides whether it gets any
budget.

These are not mutually exclusive; gates attribute variance to boundaries rather than
selecting a single winner by elimination.

## Instrumentation channel

`VLLM_SERVER_DEV_MODE=1` + a `PYTHONPATH`-injected `sitecustomize.py` that attaches debug
methods to the worker class at import time; results retrieved as JSON via
`POST /collective_rpc` (string method name dispatch on each worker). This needs no fork
branch and no GitHub write, and the HTTP data path is immune to the ~49k-char Iris log
truncation (sharp edge #17). The channel is smoke-tested in G1 before anything depends on
it: a probe method returning worker env (`NCCL_*`, effective flags), GPU UUID, and PCI bus
ID — which doubles as the effective-configuration record every arm must log. Fallback if
injection fails: env-gated debug branch on the fork via the verified pin override — that
path pushes to GitHub and requires Romain's approval first.

## Gate overview

| Gate | Question | Resource | Est. | Exit |
|---|---|---|---|---|
| G0 | Are all arms real contrasts? Schema + scripts ready? | desk | ~2 h | arm list verified; endpoints defined; scripts written |
| G1 | Is a fresh recompute deterministic? Does the symptom reproduce under controlled history? What does the collective do in isolation? | 1 node | ~1.5 h | determinism verdict + controlled baseline + bands |
| G2 | At which boundary do ranks first diverge? Are the weights right? | 1 node | ~1.25 h | branch selection (combine / local kernel / loader / attention) |
| G3 | Does the targeted intervention move the traced boundary and the endpoint? | 1 node | ~1.5 h | ≥2 bracketed treatment effects or branch falsified |
| G4 | Does the mechanism reproduce on captured activations? | 1 node | ~0.75 h | boundary-level confirmation |
| G5 | Fix (user checkpoint: direction + scope) | 1 node | ~1 h + PRs | fix merged; same-node validation green |
| G6 | 3/3 independent cold launches green with margin? | 2 nodes | ~1.5 node-h | gate green, margin measured, sentinel stays blocking |

Every GPU arm keeps the gate harness, model export, and request protocol fixed except for
its pre-registered variable, and logs the effective-configuration record (worker env,
resolved backends, cache/graph state, GPU UUIDs, `NCCL_DEBUG=INFO` selection lines).

## G0 — desk audit (COMPLETE, 2026-07-21)

Findings (each verified in the fork clone at `afb26719`):

- **CLI spellings verified**: `--no-enable-prefix-caching` (`BooleanOptionalAction`,
  `arg_utils.py:346,1140`), `--expert-placement-strategy`, `--enforce-eager`, and
  `--moe-backend` is a direct flag (`arg_utils.py:1454`; values include `triton`,
  `triton_unfused`, `emulation`). `auto` resolves via `select_unquantized_moe_backend`
  (`oracle/unquantized.py:151`), which `info_once`-logs "Using … Unquantized MoE backend
  out of potential backends: […]" — that log line plus the probe's per-layer kernel-class
  record answer the contrast-arm-existence question at G1.
- **Injection channel upgraded**: vLLM has a *supported* mechanism —
  `--worker-extension-cls` (`worker_base.py:255-287`) dynamically bases a user class into
  the Worker; `run_method` getattr (`serial_utils.py:486`) + `WorkerWrapperBase.__getattr__`
  (`:327`) make its methods `/collective_rpc`-callable. No sitecustomize needed; only
  `PYTHONPATH` so the module resolves. **Caveat**: the DP client broadcasts the RPC to all
  engines but returns only the first engine's result (`core_client.py:1440-1449`) — so
  RPC is trigger-only and per-rank data travels via a node-local side-channel directory
  (all DP workers are local processes).
- **NCCL-contrast hook window found**: the extension module is *imported* during
  `init_worker`, before `init_device` → before `init_batch_invariance()`
  (`gpu_worker.py:1173`, inside `init_worker_distributed_environment`, before
  process-group creation). A module-level wrap of `override_envs_for_invariance` (gated
  by `MARIN_PROBE_NCCL_ENV_JSON`) is therefore the only — and a sufficient — place for
  the G3 combine-branch env contrast. Job-level env cannot contrast NCCL (overwritten).
- **Combine variant per mode resolved**: `reduce_scatterv` takes a single
  `ncclReduceScatter` when per-rank sizes are all equal, else grouped per-root
  `ncclReduce` (`cuda_communicator.py:365-368`, `pynccl.py:285-320`). Sizes come from DP
  metadata; `should_dp_pad = cudagraph_mode != 0 or should_ubatch` (`dp_utils.py:152`),
  so unpadded prefill chunks → **isolated mode (1 active rank + idle dummies) runs the
  grouped-ncclReduce variant while a lockstep concurrent wave runs single
  ncclReduceScatter**. Mode comparisons are only interpretable with the per-call sizes
  log the probe installs; the microreproducer exercises both variants. `--enforce-eager`
  (G3) also forces mode 0 → changes the decode-step variant; log it there.
- **Wheel fallback is cheap**: `determine_wheel_url` (setup.py) honors
  `VLLM_PRECOMPILED_WHEEL_COMMIT`, else uses the base commit in upstream main — a
  Python-only debug branch off the pinned rev resolves to the same precompiled wheel; no
  source build.
- **Scripts written and CPU-dry-run** (committed under `tests/cluster/vllm/`):
  `_marin_rank_probe.py` (worker extension: env record, sizes log, in-server
  microreproducer, G2 trace arm/disarm; checked by `_marin_rank_probe_check.py`),
  `_experiment_c_rank_variance.py` (G1 harness: S1 battery + A–A′–A″ + gate-exact S4 +
  cross-job arm; `--dry-run` validates fixture, wave order, and companion selection).
  Result schema documented in the harness docstring (per-block JSON between
  `EXPERIMENT_C_JSON_BEGIN/END` markers; sizes logs trimmed to 200+50 lines/rank; trace
  capped at 64 calls with last-row `.pt` slices).
- **Reduced raw-data fixture (not committed — data lives in finelog / a marin bucket)**:
  `_experiment_ab_reduce_fixture.py` distils 40 groups / 256 observations from the three
  raw job logs; the small summarized tables are inlined in the report, the fixture itself
  is not committed.
- **Harness API note**: the parity test now serves through
  `VllmEngineConfig`/`VllmLauncherType`/`VllmSource` (`marin.inference.config`); the
  harness mirrors that (experiment B used the pre-refactor API). Env propagation into the
  server subprocess re-verified on this branch (`_vllm_env()` starts from `os.environ`;
  launcher overlay touches only `VLLM_USE_PRECOMPILED`/`AWS_CONFIG_FILE`/sampler).

Deviation from v2 text: the microreproducer runs *in-server* via the worker extension on
the production communicator (`get_dp_group().reduce_scatterv`), not as a separate
no-model job — strictly more faithful (exact communicator, env, stream) and free, since
G1 needs the server anyway. Its "demote H1 on null" exit is unchanged.

Exit met: every retained arm is a verified contrast; every prediction names a measured
endpoint.

## G1 — microreproducer + controlled baseline (1 node, ~1.5 h)

Order matters; earlier items gate later ones.

1. **Collective microreproducer** (no model load, minutes): rank-distinct bf16 operands
   `v_r`, each replicated across destination chunks so every destination receives the same
   mathematical sum through the production reduce-scatter path (pynccl, matched NCCL env);
   compare per-destination results against a fixed-order fp32 reference; `NCCL_DEBUG=INFO`
   records the selected algorithm. Magnitudes matched to typical hidden-state scale, and
   re-matched to traced reality after G2. Two pre-registered exits: (i) a null at
   gate-like message sizes **demotes H1** regardless of later knob results — the knob
   matrix cannot rescue a mechanism whose primitive is absent; (ii) a positive must also
   pass magnitude plausibility (per-op delta × plausible router-flip rate can reach the
   observed span) or H1 is recorded as incomplete. A synthetic negative still does not
   *refute* H1 outright (operands are synthetic), which is why the demotion is a
   reweighting, not a kill.
2. **Injection smoke**: probe method via `/collective_rpc` returns each worker's effective
   env and GPU identity. Failure → surface the fork-branch fallback decision immediately.
3. **Controlled Snowball baseline**, caching off, one discarded warmup request per server
   start (policy pre-registered):
   - **Fresh-recompute determinism**: same rank, same prompt, 8 sequential requests with
     cache reset between — the single most load-bearing measurement in the plan.
   - C1: same prompt, all 8 ranks concurrent, 2 rounds.
   - Isolated vs concurrent vs staggered (~200 ms offsets) waves, with the combine
     variant logged per mode (G0 item — modes may exercise different collectives).
   - Wave-realistic mode: sentinel on one rank while the other seven serve distinct
     wave cases — the composition the gate actually runs; expected null under the pinned
     config, worth having on record.
   - Target-rank permutation with physical-GPU identity recorded.
   - Lengths {128, 2048, 8192, 15025} + 3 additional >2,048-token prompts from the
     64-case set (cheapest generality check; everything so far rests on one prompt).
   - A–A′–A″: three server restarts of the gate config in the same job/node — the
     within-node launch band. An engineering screen, not a statistical bound (n=3).
   - One gate-config baseline in a **second job** (can run on node 2 concurrently): the
     cross-job band that every cross-job comparison (EP=4 arm, G6) needs — the historical
     0.2788 movement was cross-job, and the within-job band cannot speak to it.
   - One gate-exact-history arm (caching on, 64-wave order then sentinel): relevance
     anchor C4.

**Endpoints** (pre-registered): primary S = span of p(423) per (condition, length);
secondary D = max pairwise |Δp| over tokens present in both ranks' top-50 (coverage
flagged when top-50 sets differ); bitwise equality where predicted; golden
`max_probability_error` at full length. Effects count only if they exceed 2× the A–A′–A″
band at the same length; "collapse" = S < 0.02 at **every** measured length (8,192
included — it is the widest, and a screen at 15,025 alone could pass a knob that fails at
8,192). The 0.02 target is a screening threshold, deliberately well inside the 0.075 gate
budget minus the ~0.055 offset; G6 owns the final margin question.

**Branch rules**: symptom must reproduce twice under controlled history — otherwise stop
and debug provenance. If fresh recomputes are **not** bit-stable, the target changes from
"rank symmetry" to "per-computation determinism" — pause for a re-plan with Romain (the
intervention set is different). If the A-band is a large fraction of the baseline span,
magnitude verdicts are declared void and only bitwise/structural tests count thereafter
(pre-registered pivot).

## G1 — RESULTS (executed 2026-07-21/22; jobs `snowball-experiment-c-main-b20426d2`,
`-crossjob-d7c849b9`; 4 server launches + 1 cross-job launch, ~1.4 node-hours)

Every pre-registered exit is met, and the picture is sharper than the plan assumed:
**two separable mechanisms feed one amplifier**, and the gate's pass/fail is decided by
which discrete state a launch lands in.

### Verdicts against the pre-registered branch rules

| Rule | Result |
|---|---|
| Fresh-recompute determinism (load-bearing) | **Bit-stable in every session** — 8/8 repeats identical at 15,025 and 8,192 in S1, 4/4 in S2/S3/XJ, caching off. No pivot: the rank-symmetry framing stands, and prefix caching turns out to have confounded the *evidence* without changing the *fact* |
| Symptom reproduces twice under controlled history | Yes — 5 launches, every one shows the spread |
| Injection smoke | Passed on all 8 workers in both jobs; the fork-branch fallback (and its GitHub-write approval) is off the table |
| Microreproducer null → demote H1 | **Positive**, not null (below) |
| A-band comparable to baseline span → magnitude verdicts void | Triggered in spirit: the launch band is the same order as the rank span, so both are now first-class targets rather than signal-vs-noise |

### The per-rank answer is a pure function of rank

At 15,025 tokens, S1's per-rank values are **bit-identical** across isolated replay,
concurrent wave (2 rounds), staggered wave (~200 ms offsets), reverse and random service
order, and the wave-realistic composition (sentinel on one rank, seven different long
prompts on the others — Δ = 0.000000 for the sentinel-carrying rank in all 4 rounds).
Nothing about concurrency, peer content, timing, or service order moves a single bit.

**Superseded by G2** (kept for the record): v1 of this section read the above as
falsifying H3a/H3b, on the assumption that a concurrent wave produces equal DP sizes and
therefore a different local batch layout. The trace showed the engines **serialize** these
requests — every one of these modes ran the same `[1,…,n,…,1]` layout — so the comparison
never varied position and could not test H3b. What these modes do establish is that
request timing, peer content and service order are irrelevant *given* the scheduling that
actually occurs.

### Rank spread, launch movement, and the bands

| Session | S at 15,025 | S at 8,192 | worst error vs golden | gate verdict |
|---|---|---|---|---|
| S1 (first launch in job) | 0.111884 | 0.234087 | 0.098835 | fail |
| S2, S3 (bit-identical to each other) | 0.071126 | 0.286958 | 0.070121 | pass |
| S4 (gate-exact, caching ON) | 0.071126 | — | 0.070121 | pass |
| XJ (second job, second node) | 0.099165 | 0.221011 | 0.078494 | fail |

Bands (per-rank |Δp|, the units G3/G6 treatment effects must beat):
within-node S1→S2 **0.046777** at 15,025 (0.301409 at 8,192); S2→S3 **0.000000**
(bit-identical launches); cross-job S1→XJ **0.061380**, S2→XJ 0.026833. All are one to
two orders of magnitude above G6's pre-registered targets (span < 0.02, movement < 0.01).

Argmax flips across ranks at 8,192 in every session, and two of the three extra long
prompts also flip argmax across ranks — the phenomenon is not specific to the sentinel.

### The launch states are discrete and they recur exactly

Per-rank values reproduce **bit-exactly across jobs, nodes, and days**: XJ's rank 0 at
8,192 is 0.621766, the same value job `bd7f6354` produced a day earlier (and its ranks 1,
3, 6 match too), while S2/S3's rank 0 is 0.343003 — exactly job `de845a00`'s value. The
0.278763 S2→XJ movement at 8,192 is the same number the plan recorded as the historical
launch movement. Continuous floating-point noise does not reproduce six decimals across
machines; a **discrete choice** does. Each (rank, prompt) has a small recurring set of
outcomes, and a launch draws one per rank.

This is the flakiness mechanism, and it explains #7354's history: the representative wave
scores each case on one rank and passes 5/5, while the sentinel's verdict depends on the
draw. Five launches produced **three distinct states**; the sentinel's worst per-rank error
exceeds 0.075 in two of them (S1 0.098835, XJ 0.078494) and stays under in the third
(S2/S3/S4, 0.070121 — including S4, which ran with caching on in gate-exact request
order). S2, S3 and S4 are the same state, so this is three independent draws, not five.

### Microreproducer: H1's primitive confirmed, magnitude deferred

On the production EP communicator, with all eight destinations receiving the *same*
mathematical sum: **8 distinct output checksums**, bit-stable across 16 iterations,
deviating from a fixed-order fp32 reference by ≤1 bf16 ulp on ~52% of elements (max
1.25e-01 at operand scale 1, scaling exactly with magnitude — consistent with single
rounding, not accumulation blowup). All three fixed-order fp32 reference orderings agree
**bitwise with each other**, so a fixed-order fp32 combine is destination-independent by
construction: the proposed fix collapses the primitive.

So the combine is deterministic-but-destination-dependent, at ~1 ulp per op. The 0.07–0.29
spans therefore require amplification, and the discreteness above says what kind: router
top-4 flips. G2 tests that directly.

### Revised mechanism

1. **Within a launch, across ranks**: ring reduce-scatter accumulates each destination's
   chunk in a rotated order fixed by rank identity → ~1 ulp per-rank perturbation →
   near-critical routing decisions flip → discrete output differences. (H1, confirmed at
   the primitive; boundary evidence pending G2.)
2. **Across launches, same rank**: a launch-scoped discrete choice — first-launch-in-job
   S1 differs from warm S2/S3, which are bit-identical to each other — shifts the same
   perturbation and flips different decisions. (H4, newly evidenced.)

Both feed the same amplifier. **A fixed-order combine addresses (1) but not (2)** — which
changes G5's scope and is the main thing to weigh at the checkpoint.

### Caveats and follow-ups

- S1 ran the full battery while S2/S3 ran the lite one, so "first launch differs" is
  confounded with "different preceding requests". Within-launch history is demonstrably
  irrelevant (all S1 modes bit-identical), which makes launch state the live explanation,
  but three *identical* batteries would settle it. XJ is cold yet closer to warm S2/S3,
  so cold-vs-warm caching is not a sufficient account on its own.
- The combine-size log filled its budget during warmup (one 15k prefill ≈ 780 calls), so
  the concurrent wave's size vectors went unrecorded. Fixed: per-mode rotation.
- Two probe defects found and fixed: an exception in a `collective_rpc` method kills the
  worker and the whole serve (probes now report failures as data — this cost the first
  attempt two jobs), and the microreproducer and trace hooks initially targeted the DP
  group where production combines over EP.

## G2 — boundary trace + weight audit (1 node, ~1.25 h)

Via the injection channel, caching off, on the smallest prefix that reliably shows spread
(start at 128; escalate if needed). Three layers (first, middle, last MoE layer), final
token, all 8 ranks. Boundaries, per Codex's decomposition:

1. pre-dispatch hidden state + router logits + selected expert IDs;
2. gathered copy for each source rank;
3. local expert GEMM output per source chunk;
4. locally weighted/summed partial immediately before reduce-scatter;
5. post-combine output;
6. post-residual / post-MoE;
7. LM-head logits + log-softmax (only if all earlier boundaries agree).

Per boundary: max/mean absolute difference, L2, and a checksum (exact equality), indexed
by (token, source chunk, destination rank, expert ID, GPU UUID). Output size capped
(pre-registered). Plus the H2 weight audit as specified above.

Branch table:

| First divergence | Verdict | Next |
|---|---|---|
| Boundary 4→5 (partials equal, combined differ) | H1 confirmed at boundary | G3 combine branch |
| Boundary 2→4 (equal inputs, differing partials) | H3b local kernel | G3 kernel branch |
| Weights differ in audit | H2 | loader fix (skip to G5) |
| Before MoE (attention) | H6 | attention trace before any more arms |
| Everything equal yet outputs differ | harness defect | stop, debug |

## G2 — RESULTS (executed 2026-07-22; job `snowball-experiment-c-trace-*`)

**Branch selected: combine.** The trace places first divergence exactly at the first MoE
combine, and shows the amplification path the magnitude argument needed.

Method correction found in the first attempt: the engines **serialize** these requests.
During what the harness submits as a concurrent wave, the DP size vector is
`[1,1,1,1,128,1,1,1]` — one rank holds a real chunk, the other seven contribute a single
dummy token each. A shared capture therefore compared one rank's real tokens against
seven ranks' dummies, and call N did not denote the same layer on every rank. The trace
now arms once per serving rank, so capture r holds exactly rank r's own prefill; the
comparison is the same computation, same weights, same tokens, on different ranks.

Result at 128 tokens, comparing the serving rank's own entries across all 8 captures:

| call (MoE layer) | pre-dispatch hidden | post-combine | distinct top-4 expert sets |
|---|---|---|---|
| 1 | **1 (all ranks identical)** | **8 (all ranks differ)** | 1 |
| 2–13 | 8 | 8 | 1 |
| 14 | 8 | 8 | **3** |
| 17–20, 24, 26 | 8 | 8 | 2 |
| 25 | 8 | 8 | **5** |

Reading the chain:

1. Everything upstream of the first MoE — embedding, attention, dense projections, norms —
   is **bitwise identical across all 8 ranks** (call 1 pre-dispatch, one distinct value)
   **at 128 tokens**. H6 and any pre-MoE origin are ruled out for a single-chunk prompt.
   (Superseded in part: the 2,048-token trace, recovered later, is 2-distinct at the first
   MoE input — a second, chunk-related upstream divergence that this 128-only reading
   missed. See the findings issue / report.)
2. The very first MoE combine produces **8 distinct outputs** for the same tokens. The
   divergence is created there, at the magnitude the microreproducer measured (~1 ulp).
3. It propagates: from call 2 on, the hidden state entering every later MoE differs per
   rank.
4. By layer 14 it **flips the router's top-4 expert selection** for the final token, and by
   layer 25 the 8 ranks choose 5 distinct expert sets. This is the amplification step the
   plan required H1 to demonstrate: a 1-ulp perturbation becomes a discrete change in
   which experts run, and discrete routing differences produce the 0.04–0.29 spans.

Together with G1's microreproducer — the same collective returns 8 different results for
one mathematical sum, while all three fixed-order fp32 references agree bitwise — H1 is
confirmed end to end for the rank dimension: **destination-dependent reduce-scatter →
~1 ulp per-rank perturbation → router flips → discrete output differences.**

The weight audit (H2) was not run: it tests an integrity failure that cannot produce a
first-divergence at the combine with identical inputs, and the trace already localizes
the mechanism. It stays available if G3 falsifies the combine branch.

Not addressed by this chain: the launch-scoped movement (G1's S1 vs S2/S3). The combine's
ring order is fixed by rank topology and cannot vary across launches of the same
configuration, so a second mechanism remains live for that dimension.

## G3 — targeted interventions (1 node, ~1.5 h; only the selected branch)

Common protocol for every arm: verified effective config; B–A–A–B same-node baseline
bracketing; treatment effect vs local baseline median; ≥2 treatment + ≥2 baseline
observations before any conclusion; screen at 128, confirm at {8,192, 15,025}.

- **Combine branch**: injected fixed-order combine (all-gather partials + ordered fp32
  local sum — ~20 Python lines, acceptable at `--max-num-seqs 1` throughput). Direct
  causal test: spread collapses, pre-combine partials unchanged. Optional NCCL contrast
  only if G0 showed the env survives `override_envs_for_invariance()` and the effective
  setting is observable in the probe.
- **Kernel branch**: `--enforce-eager`; moe-backend contrast if an alternative exists;
  batch-position permutation (reorder peers so the target's gathered offset changes);
  `VLLM_BATCH_INVARIANT=0` as a broad diagnostic — re-registered prediction: values will
  shift (it changes dense kernels *and* unpins NCCL env); the informative readout is
  whether fresh-recompute stability degrades.
- **Loader branch**: `{"distributed":false}` A/B after canonical weight checks.
- **Characterization (budget-permitting, secondary evidence only)**:
  `--expert-placement-strategy round_robin`; EP=4 on a 4-GPU job (dispersion measured as
  per-rank std and mean pairwise |Δ|, not span — span shrinks mechanically with fewer
  ranks; cross-job band applies; 128-token startup smoke first). The 2-GPU arm from v1 is
  dropped (memory-tight, weakest inference).

Kill: if the branch's interventions produce no effect beyond band under verified
application, return to G2 with the next boundary candidate rather than widening the
matrix.

## G3 — RESULTS (executed 2026-07-22; jobs `snowball-experiment-c-fixedcombine-*`,
`-tracefixed-*`)

**The combine fix works and is not sufficient.** It is the sole cause of the first-layer
divergence; a second source survives it and dominates long prompts.

### Intervention: destination-independent combine

All-gather every rank's partial, sum in rank order in fp32, slice this rank's chunk — so
every destination performs identical arithmetic. Bracketed B–A–A–B inside one server, in
two independent launches. Bracketing is clean: `baseline_pre` reproduces `baseline_post`
bit-exactly, and the two treatment sweeps reproduce each other bit-exactly.

| length | F1 baseline → treatment | F2 baseline → treatment |
|---|---|---|
| 128 | 0.042688 → **0.006272** (−85%) | 0.042688 → **0.006272** (−85%) |
| 8,192 | 0.200678 → 0.171888 (−14%) | 0.270564 → 0.157229 (−42%) |
| 15,025 | 0.066786 → 0.063381 (−5%) | 0.080416 → 0.034474 (−57%) |

The pre-registered collapse criterion (S < 0.02 at **every** length) is met only at 128.
Note also that the treatment moved the sentinel *away* from the TPU golden at 15,025
(worst error 0.0665 → 0.0859 in F1): better-defined arithmetic is not the same as closer
to a golden produced on other hardware.

At 128 tokens both launches produce bit-identical baselines and bit-identical treatments,
so the launch-state variability does not touch the short case at all — it is specific to
the long, multi-chunk prompts.

### Trace with the fix installed: where the residual lives

Repeating the per-rank capture with the fixed combine:

| call (MoE layer) | pre-dispatch hidden | post-combine | vs. baseline trace |
|---|---|---|---|
| 1 | 1 | **1 (all 8 ranks identical)** | was 8 |
| 2 | 1 | **2** | was 8 |
| 3+ | 2 | 2 | was 8 |

Two things follow. First, the destination-dependent reduce-scatter was the **sole** cause
of the first-layer divergence — removing it makes the first MoE output bitwise identical
on all eight ranks. Second, a smaller residual appears at the second MoE layer, and it is
not an eight-way spread: checksums show **rank 7 alone differs; ranks 0–6 agree exactly.**

Rank 7 is the terminal position in the gathered batch. With one serving rank and seven
idle ranks contributing a dummy token each, the serving rank's 128 rows sit at offset r
with (7−r) rows trailing; only r = 7 has no trailing rows. Offsets 0–6 give identical
results and offset 7 does not, which is the signature of a tile/padding boundary in the
local MoE path rather than of anything in the collective.

### Correction to the G1 write-up

G1 recorded H3b (local kernel position sensitivity) as falsified because isolated,
concurrent, staggered and wave-realistic modes agreed bit-for-bit. G2 showed why that
inference was wrong: **the engines serialize these requests**, so all four modes ran the
same `[1,…,128,…,1]` layout and none of them varied the position. That experiment never
tested H3b. The tracefixed result now provides direct evidence for it.

### Revised mechanism (three components)

1. **Combine destination-dependence** (H1) — confirmed at the primitive, at the boundary,
   and by intervention. Dominant at short lengths; removable in software.
2. **Gathered-batch position sensitivity in the local MoE path** (H3b) — survives the fix,
   visible as the terminal-position rank diverging; grows with prompt length, since every
   chunk of a multi-chunk prefill repeats the exposure.
3. **Launch-scoped state** (H4) — moves per-rank values across launches of the same
   configuration, only for multi-chunk prompts; untouched by the combine fix.

All three feed the same amplifier: near-critical router decisions, where a ~1 ulp
perturbation flips which experts run and moves the output distribution by 0.03–0.29.

## G4 — mechanism confirmation (1 node, ~0.75 h)

Capture real pre-combine partials from a divergent layer (G2 tooling) and replay through:
the production combine path; a fixed-order fp32 sum + cast; the candidate fixed combine.
Confirm the intervention changes the first mismatching boundary and preserves earlier
ones. A tiny random-weight GrugMoE repro is built only after the mechanism is known, as a
maintainable regression asset — a random model's failure to reproduce refutes nothing
(router margins and scales differ).

## G5 — fix [USER CHECKPOINT — REACHED 2026-07-22, awaiting Romain]

The diagnosis is complete enough to choose a direction, and the choice is materially
different from what v2 anticipated. **A deterministic combine alone will not make the gate
reliable**: it removes the first-layer divergence entirely but leaves a position-sensitive
residual (0.006 at 128 tokens, 0.16–0.17 at 8,192) and does nothing about launch-scoped
movement. Options, with what each is known to buy:

| Option | Evidence | Cost | Leaves unfixed |
|---|---|---|---|
| **A. Deterministic combine only** | −85% at 128, −5…−57% at 8,192/15,025 (launch-dependent); two launches, bracketed | ~20 lines in the fork; all-gather instead of reduce-scatter, so world_size× the combine traffic (measured acceptable at `--max-num-seqs 1`, unmeasured at production concurrency) | position residual, launch movement — gate still fails at 8,192 |
| **B. A + chase the position sensitivity** | Localized to the terminal gathered-batch position; not yet localized to a specific kernel | 1–2 more gates; likely an upstream vLLM MoE issue | launch movement |
| **C. A + B + launch-state pinning** | H4 not yet localized to a sub-axis | Unknown; the sub-axis hunt is open-ended | — |
| **D. Change what the gate asserts** | The gate's own assumption — one cold launch, one rank per case, compared to a frozen cross-hardware golden — is what makes any of this fatal | Small | the numerics (deliberately) |

My recommendation is **D as the immediate move, with A landed alongside it**, and B/C only
if you want the numerics themselves fixed. Reasons: the sentinel is already ~0.055 from
the TPU golden before any of this (a GPU-vs-TPU offset, out of scope here), leaving ~0.02
of headroom that no realistic combination of A/B/C reliably buys back; and the discreteness
result means the gate is not measuring a stable quantity — it is sampling one of a handful
of launch states, three of the five measured states failing and two passing. A gate that
asserts rank-consistency and launch-consistency separately, against bounds frozen from
measured launches, would have caught this class of problem earlier and would not be a
coin flip.

Whichever direction you pick, the remaining work needs your go-ahead: everything so far is
diagnosis, and no fork or gate change has been written.

Original option list (still applicable to whichever branch is chosen):

- **Combine fix (H1)**: deterministic-combine mode in the fork as an explicit server
  option (not an env var), used by the parity gate. Explicit scope decision: test-only
  mode (keep a production-default canary run in G6) vs production default (perf
  assessment needed beyond the gate's `--max-num-seqs 1`).
- **Kernel fix (H3b)**: pin the resolved MoE backend / kernel config; upstream if the
  defect is generic.
- **Loader fix (H2)**: correct the distributed loader.
- **Gate hygiene (any branch)**: resolve the sentinel cache asymmetry (sentinel wave
  before representative waves, or caching off in the gate job) — decided here, validated
  in G6.
- **Fallback** if the mechanism is inherent FP non-associativity with no acceptable
  deterministic mode: a sentinel redesign memo — rank-consistency and golden-correctness
  asserted separately, any bound frozen from independent launches, presented as a decision
  rather than auto-applied.

Mechanics: fork PR via `refresh-tpu-vllm-forks` overlay, marin pin bump + gate config PR,
upstream vLLM issue if not fork-specific. All GitHub writes happen at or after this
checkpoint. Same-node bracketed validation before G6.

## G6 — independent validation (2 nodes, ~1.5 node-hours)

Three cold launches (distinct jobs, ≥2 distinct nodes), production defaults + the landed
fix: full parity workload, plus a stratified short/medium/long prompt subset replayed on
**every** rank (the representative wave alone cannot detect rank-specific regressions).
Pass, pre-registered:

- every observation meets `assert_matches` at 0.075;
- sentinel per-rank span < 0.02 in each launch; cross-launch per-rank movement < 0.01 at
  15,025;
- representative wave unchanged-green;
- **margin check**: residual sentinel error headroom (0.075 − observed worst error) must
  exceed 2× the observed residual cross-launch band. The arithmetic is knife-edge by
  construction: offset ~0.055 + span/2 at the 0.02 target + 0.01 movement ≈ 0.075, so the
  span/movement criteria alone can pass a gate that still fails. If the margin check
  fails, escalate with a pre-registered menu: tighten the span/movement targets, rescore
  the sentinel against a Levanter-GPU reference (mean delta +0.0003 on this prompt), or
  pull golden provenance into scope — Romain's decision.

Node caveat: the ≥2-distinct-nodes requirement is coverage, not the measured axis — the
historical launch movement was plausibly same-node, so cross-job repetition is the part
of G6 that probes it.

Then: sentinel stays blocking, report updated with the mechanism section, harness + data
fixture committed, threads on #7354 closed out (comment text approved per standing rules).

## Budget

| Gate | Node-hours | GPU-hours |
|---|---|---|
| G1 | 1.5 | 12 |
| G2 | 1.25 | 10 |
| G3 | 1.5 | 12 (EP=4 arm counts 0.375 node-h at 4 GPUs) |
| G4 | 0.75 | 6 |
| G5 | 1.0 | 8 |
| G6 | 1.5 | 12 |
| Subtotal | 7.5 | 60 |
| +20% contingency | **9.0** | **72** |

Peak concurrency ≤ 2 nodes; wall time is lower than node-hours where G6/characterization
arms parallelize. The loader A/B (G3, loader branch) budgets 2–3× the normal load time —
with `distributed:false` every rank streams the full checkpoint itself. Per-launch overhead re-derived from measured data (server startup
~135 s, gate test body ~273 s per #7354; weight load tens of seconds with the distributed
streamer) — v1's "model load dominates at ~8 min" overstated per-arm cost; the dominant
costs are job provisioning and long-prompt request rounds.

## Risks and kill criteria

- **Injection smoke fails** → instrumentation needs the fork debug branch → requires
  approval; surfaced at G1, not discovered mid-G2.
- **Microreproducer negative** ≠ H1 refutation (synthetic operands); it only reweights
  toward the trace.
- **Trace shows pre-MoE divergence** → stop all MoE-path arms; H6 branch with a re-plan.
- **Fresh recomputes unstable** → determinism-first re-plan with Romain (different
  intervention space; this plan's rank-symmetry framing would be wrong).
- **Symptom absent under controlled history** → stop; provenance/harness debugging.
- **A-band comparable to baseline span** → magnitude verdicts void; bitwise/structural
  only (pre-registered pivot).
- **Iris log truncation / `tail` buffering** (sharp edges #17, #18): all bulk data over
  the HTTP channel; logs carry only markers and summaries.
- **4-GPU arm feasibility**: 128-token startup smoke before spending; drop on OOM.
- **Global kill**: two consecutive gates producing no verified discrimination → stop and
  re-plan with Romain rather than spending the remaining budget.

## Deliverables

1. Mechanism note with boundary-level evidence (report update).
2. Fix PR(s) — fork + marin pin/config — or the fallback decision memo.
3. G6 validation evidence (3 launches, linked jobs) including the margin analysis.
4. Committed harness, microreproducer, trace tooling, and reduced data fixture.
5. Upstream vLLM issue if warranted.

## Prior work

- Report: [snowball-gpu-golden-logprob-parity.md](../../docs/reports/snowball-gpu-golden-logprob-parity.md).
- Experiments A/B: [_experiment_rank_spread.py](../../tests/cluster/vllm/_experiment_rank_spread.py),
  `scratch/experiment_b.py`, `scratch/LOG-attempts.md`; repeat-run history in #7354.
- Plan reviews (v1 → v2): `.agents/tmp/CLAUDE_PLAN_REVIEW.md` (prefix-cache confound,
  batch-invariant NCCL pinning, arm validity, G4 construction),
  `.agents/tmp/CODEX_PLAN_REVIEW.md` (trace-first restructure, endpoint definitions,
  weight-audit correction, budget audit), `.agents/tmp/GEMINI_PLAN_REVIEW.md` (margin
  analysis, injection channel, NCCL_DEBUG verification),
  `.agents/tmp/KIMI_PLAN_REVIEW.md` (invariance-pinning depth incl. MoE tiles, combine
  dtype and magnitude analysis, greedy flips, plausibly-same-node launch pair, cross-job
  band, wave-realistic mode). Every review claim adopted here was independently
  re-verified against the pinned source or raw logs; two review claims were rejected on
  verification (a TP=8 reference arm — the model hard-rejects TP≠1 — and a
  cache-sharing concern about the length ladder, which uses suffixes, not prefixes).
