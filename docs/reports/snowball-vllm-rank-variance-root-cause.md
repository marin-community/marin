# Root cause of vLLM per-rank logprob variation on a 67B MoE under DP+EP

Date: 2026-07-22 · Cluster: `cw-us-east-02a` (8×H100) · vLLM fork `afb2671946` · Companion to [the earlier descriptive report](snowball-gpu-golden-logprob-parity.md), which characterised the symptom and identified no cause.

Status: the mechanism results are **replicated** (every headline measurement reproduces across at least two independent server launches, and several reproduce bit-exactly across jobs, nodes, and days). The remediation results are **exploratory**: one intervention, measured in two launches, on one prompt family.

## TL;DR

Serving a 67B mixture-of-experts model with vLLM under data parallelism of 8 plus expert parallelism, the same prompt returns measurably different next-token distributions depending on which data-parallel rank serves it. We traced the mechanism end to end.

- **At a single-chunk prompt, divergence is created entirely by the expert-parallel combine.** For a 128-token prompt everything upstream of the first MoE layer — embedding, attention, dense projections — is *bitwise identical* on all eight ranks, and the first MoE combine returns eight distinct outputs for the same tokens. At multi-chunk prompts a second divergence appears upstream of the combine (see below), so this clean isolation is length-specific.
- **The cause is destination-dependent reduce-scatter.** A model-free reproducer on the production communicator gives eight different results for one mathematically identical sum, stable across 16 repetitions, differing from a fixed-order fp32 reference by ≤1 bf16 ulp on 52% of elements. All three fixed-order reference orderings agree bitwise with each other.
- **The amplifier is expert routing.** That ~1 ulp perturbation flips the router's top-4 expert selection by layer 14; by layer 25 the eight ranks choose five distinct expert sets. Discrete routing changes are what turn a rounding difference into a 0.03–0.29 probability difference.
- **Launch-to-launch variation is a separate mechanism.** The collective's per-rank output is bit-identical across launches, jobs and nodes, so it cannot explain launch variation. Per-rank model outputs nonetheless move up to 0.30 between launches of an identical configuration — and they move between a *small set of recurring discrete values*, reproducing bit-exactly across jobs and days.
- **A deterministic combine helps and is not sufficient.** It eliminates the first-layer divergence entirely and removes 85% of the rank spread at 128 tokens, but between 5% and 57% at 8,192 and 15,025 tokens depending on the launch. With the collective ruled out, the residual localises to a single rank — the terminal position in the gathered batch — implicating position sensitivity in the local MoE kernel.
- **Consequence for testing:** a gate that serves one cold launch and compares per-rank outputs to a frozen golden is sampling one of a few discrete states. Of three distinct states observed, two exceed the 0.075 bound and one passes.

Terms: *data-parallel (DP) rank* — one of eight worker processes, each holding a slice of the experts, addressed by an `X-data-parallel-rank` header. *Combine* — the collective that sums each token's expert outputs after expert-parallel dispatch. *Rank spread* (S) — max minus min, across the eight ranks, of the probability assigned to one fixed token.

## Setup

Model: 67B-A2B MoE (~2B active), 256 experts, top-4 routing, 32 experts per rank, 26 layers, bf16. Serving flags match the parity gate exactly: `--data-parallel-size 8 --enable-expert-parallel --max-num-seqs 1 --max-num-batched-tokens 512`, FlashAttention backend, `VLLM_BATCH_INVARIANT=1`, `VLLM_USE_FLASHINFER_SAMPLER=0`. Prefix caching defaults on and was disabled in every arm except the gate-exact session S4, which reproduces the gate's own request history.

Expert-parallel path at this revision: all-gather the tokens, run local experts over the full gathered batch (non-local rows zeroed), then `ncclReduceScatter` on bf16 tensors. There is no fixed-order or fp32-output variant in the model code, and the `naive`/`pplx` all-to-all backends have been removed and silently alias to `allgather_reducescatter`.

`VLLM_BATCH_INVARIANT=1` reaches further than dense-op determinism: it overwrites the NCCL environment unconditionally (single channel, `NCCL_PROTO=Simple`, `NCCL_ALGO=allreduce:tree`, NVLS/CollNet/P2P-net disabled, which restricts reduce-scatter to Ring), pins the MoE grouped-GEMM tile configuration with `SPLIT_K=1` and skips tuned-config lookup, and pins FlashAttention to a single split. So the measurements below are already under vLLM's strongest determinism setting.

### Instrumentation

Measurement needs per-rank internals from a live server without modifying the fork. We used vLLM's supported worker-extension mechanism: `--worker-extension-cls` mixes a user class into the Worker at init, and its methods become callable on every rank through `POST /collective_rpc` under `VLLM_SERVER_DEV_MODE=1`. The module ships to the node at runtime and is added to `PYTHONPATH`; no fork branch, no rebuild.

Two constraints shaped the design, both learned by breaking them:

- The DP client broadcasts an RPC to all engines but **returns only the first engine's result**, so RPC is a trigger, not a data path. Per-rank data is written to a node-local directory and collected by the harness.
- An exception escaping a `collective_rpc` method **kills the worker process and the whole server**. Probe methods therefore return their traceback as data and never raise; a CPU-only check asserts that contract.

Endpoints were pre-registered before the first GPU run: primary S = rank spread of p(token 423); secondary D = max pairwise |Δp| over tokens present in both ranks' returned top-50; bitwise equality where predicted; and the gate's own statistic, max over the golden's 25 tokens of |p_observed − p_golden|.

## Chronology, including what failed

| # | Job | Purpose | Outcome |
|---|---|---|---|
| 1 | `main-c13d5d45`, `crossjob-2f72ea73` | first controlled baseline | **Both failed** after model load: a pynvml `str`/`bytes` mismatch in the probe raised, which killed the workers. Cost ~8 H100-minutes and produced no data. Fix: probes report failures as data. |
| 2 | `main-b20426d2` | 4 launches: full battery, two repeats, gate-exact | Succeeded — G1 results below |
| 3 | `crossjob-d7c849b9` | second job, second node | Succeeded — cross-job band |
| 4 | `trace-4e8f82aa` | boundary trace, wave-based | Ran, but the design was **invalid** (below). Superseded. |
| 5 | `trace-e15f0c4e` | boundary trace, per serving rank | Succeeded — G2 results |
| 6 | `fixedcombine-2b0db3ad` | deterministic-combine intervention, 2 launches | Succeeded — G3 results |
| 7 | `tracefixed-5d61f23f` | boundary trace with the intervention installed | Succeeded — residual localisation |

Roughly 4 node-hours (32 H100-hours) in total, against a 9 node-hour budget.

## Results

### The per-rank answer is deterministic and reproducible

With prefix caching off, repeated fresh recomputations of the same prompt on the same rank are **bitwise identical**: 8/8 repeats in the full battery at both 15,025 and 8,192 tokens, 4/4 in each of the other three sessions. This mattered because the earlier report's stability evidence had been collected with prefix caching on, where repeats are cache hits; the underlying fact survived the confound, but it was unmeasured until now.

The per-rank value is also invariant to request context. Isolated sequential replay, a concurrent wave (two rounds), a staggered wave with ~200 ms offsets, reverse and random service order, and a "wave-realistic" composition (sentinel on one rank, seven different long prompts on the others) all return **bit-identical per-rank values** — for the wave-realistic case, Δ = 0.000000 for the sentinel-carrying rank in all four rounds.

That invariance is weaker evidence than it first appears, and we misread it initially. See "Corrections" below.

### Rank spread, by launch and length

Spread of p(423) across the eight ranks, and the gate's own worst-case statistic:

| session | S @ 15,025 | S @ 8,192 | worst error vs golden @ 15,025 | vs 0.075 bound |
|---|---|---|---|---|
| S1 (first launch in job) | 0.111884 | 0.234087 | 0.098835 | fail |
| S2 | 0.071126 | 0.286958 | 0.070121 | pass |
| S3 (bit-identical to S2) | 0.071126 | 0.286958 | 0.070121 | pass |
| S4 (caching on, gate-exact order) | 0.071126 | — | 0.070121 | pass |
| XJ (second job, second node) | 0.099165 | 0.221011 | 0.078494 | fail |

Five launches, **three distinct states**: S2/S3/S4 are one state, and S2 and S3 are bit-identical to each other. Two of the three states exceed the bound.

Spread is present at every length tested and is not monotone in length: 0.041857 at 128, 0.039831 at 2,048, 0.234087 at 8,192, 0.111884 at 15,025 (S1). At 8,192 the *greedy token itself* differs across ranks in every session — ranks return 423 or 426 — so the failure mode is not confined to probability tolerance.

Three additional long prompts from the same 64-case set were checked to test generality beyond the one sentinel. All three show cross-rank disagreement (D = 0.050097, 0.052299, 0.021598 at 23,113 / 14,734 / 11,291 tokens), and two show the greedy token differing across ranks. The phenomenon is not specific to the sentinel prompt.

### Launch variation is discrete and recurs exactly

Per-rank values move between launches of an identical configuration. Per-rank |Δp|:

| comparison | @ 15,025 | @ 8,192 |
|---|---|---|
| S1 → S2 (same job, same node) | 0.046777 | 0.301409 |
| S2 → S3 (same job, same node) | **0.000000** | **0.000000** |
| S1 → XJ (different job and node) | 0.061380 | 0.211697 |
| S2 → XJ | 0.026833 | 0.278763 |

Two launches can be bit-identical (S2, S3) and two can differ by 0.30. The values also **recur exactly across jobs, nodes and days**. At 8,192 tokens, the cross-job launch returned 0.621766 on rank 0 — the same value a job produced the previous day — and also matched that job on ranks 1, 3 and 6. The S2/S3 state returned 0.343003 on rank 0, matching a *different* prior job. Continuous floating-point noise does not reproduce six decimal places across machines; a small set of discrete outcomes does.

This is the flakiness mechanism. The gate's 64-case wave scores each case on a single rank and passes reliably; the sentinel replays one prompt across all eight ranks and its verdict depends on which state the launch lands in.

### The collective is destination-dependent — and launch-invariant

A model-free reproducer runs inside the live worker on the production expert-parallel communicator, with the same environment and stream. Every rank generates all ranks' operands from fixed seeds, replicated across destination chunks, so all eight destinations receive the *same mathematical sum*. Modes cover both production collective variants (equal sizes → single `ncclReduceScatter`; uneven sizes → grouped per-root `ncclReduce`), dense and 4-of-8-sparse partials, at three operand magnitudes.

Every mode, every magnitude:

- **8 distinct output checksums** across the 8 destinations, for one mathematical sum.
- **Bit-stable across 16 repetitions** within a rank — deterministic, not noisy.
- Deviation from a fixed-order fp32 reference of at most 1 bf16 ulp: max |out − ref| = 1.25e-01 at operand scale 1, 1.0 at scale 8, 4.0 at scale 32, scaling exactly with magnitude, which is the signature of single-rounding rather than accumulated error. 684,812 of 1,310,720 elements (52%) differ in the dense case; 431,924 (33%) in the sparse case.
- All three fixed-order fp32 reference orderings (rank order, reversed, ring-rotated) agree **bitwise with each other**, so a fixed-order fp32 combine is destination-independent by construction.

The eight per-rank checksums are **identical in every session and in the second job on a different node**. The collective's destination dependence is perfectly reproducible, which excludes it as the source of launch-to-launch variation.

### Where divergence enters the model

The boundary trace captures, on every rank, the hidden state entering MoE dispatch, the selected expert IDs, the gathered batch, the pre-combine partials, and the combined output. Comparing the serving rank's own prefill across eight captures at 128 tokens (26 MoE layers, one prefill chunk):

| MoE layer | distinct pre-dispatch hidden states | distinct post-combine outputs | distinct top-4 expert sets |
|---|---|---|---|
| 1 | **1** | **8** | 1 |
| 2–13 | 8 | 8 | 1 |
| 14 | 8 | 8 | **3** |
| 17–20, 24, 26 | 8 | 8 | 2 |
| 25 | 8 | 8 | **5** |

At 128 tokens, everything before the first MoE is bitwise identical on all eight ranks, which excludes attention, embeddings and dense projections as the origin *at a single-chunk prefill*. The first combine produces eight distinct outputs. From layer 2 on, every subsequent MoE sees a different input per rank. By layer 14 the perturbation flips which experts the router selects, and by layer 25 the eight ranks select five distinct expert sets.

This clean isolation holds only for a single-chunk prompt. The same trace at 2,048 tokens (four prefill chunks) shows the pre-dispatch hidden already 2-distinct at the first MoE — ranks 3 and 4 versus the other six — *before any combine*. So a multi-chunk prefill carries a second, chunk-related divergence upstream of the combine, on top of the combine's own eight-way split; the 3,4 grouping does not match the 4+4 PCI-socket split. The single-chunk case isolates the combine mechanism cleanly; the multi-chunk case does not, and the combine fix's shrinking effect at longer lengths (below) is consistent with that upstream source.

A 1-ulp difference cannot by itself move a probability by 0.1; a change in which experts run can, and that is the path these rows trace.

### Intervention: a destination-independent combine

We replaced the combine with one that all-gathers every rank's partial, sums in rank order in fp32, then slices this rank's chunk — so every destination performs identical arithmetic. It moves world_size times the data and is not intended to be efficient; at `--max-num-seqs 1` it is affordable. Measured B–A–A–B inside a single server, in two independent launches. Bracketing was clean: `baseline_pre` reproduced `baseline_post` bit-exactly, and the two treatment sweeps reproduced each other bit-exactly.

| length | launch F1 | launch F2 |
|---|---|---|
| 128 | 0.042688 → **0.006272** (−85%) | 0.042688 → **0.006272** (−85%) |
| 8,192 | 0.200678 → 0.171888 (−14%) | 0.270564 → 0.157229 (−42%) |
| 15,025 | 0.066786 → 0.063381 (−5%) | 0.080416 → 0.034474 (−57%) |

The 128-token case is bit-identical between the two launches in both baseline and treatment, so launch variation does not touch the single-chunk case at all — it is specific to multi-chunk prompts.

Two further observations. The treatment moved the sentinel *away* from the TPU-derived golden at 15,025 in F1 (worst error 0.066453 → 0.085918): better-conditioned arithmetic is not the same as agreement with a golden generated on other hardware. And the residual spread still differs between launches (0.171888 vs 0.157229 at 8,192), consistent with the launch mechanism being untouched.

### Where the residual lives

Repeating the boundary trace with the intervention installed:

| MoE layer | distinct pre-dispatch hidden | distinct post-combine | baseline comparison |
|---|---|---|---|
| 1 | 1 | **1** | was 8 |
| 2 | 1 | **2** | was 8 |
| 3+ | 2 | 2 | was 8 |

The first-layer divergence is gone: with a destination-independent combine, the first MoE output is bitwise identical on all eight ranks. This confirms the collective was the *sole* cause of that divergence.

The residual is not an eight-way spread. Checksums show **rank 7 alone differs; ranks 0–6 agree exactly**. In these captures the serving rank contributes a real chunk while seven idle ranks contribute one dummy token each, so the serving rank's rows sit at offset r in the gathered batch with (7−r) rows trailing — and only r = 7 has no trailing rows. Offsets 0 through 6 give identical results and offset 7 does not. That pattern points to a tile or padding boundary in the local MoE path, not to anything in the collective. It is not a socket split: the eight GPUs divide 4+4 by PCI bus ID, which does not match the 7-vs-1 grouping.

## Interpretation

Three mechanisms, one amplifier.

1. **Combine destination-dependence.** Ring reduce-scatter accumulates each destination's chunk in a rotated order fixed by rank identity, and the bf16 store rounds at each hop. Confirmed at the primitive (model-free reproducer), at the boundary (first-layer divergence), and by intervention (divergence removed). Deterministic and launch-invariant. Dominant at short lengths.
2. **Chunk/position sensitivity in the local path.** Two signatures: with the combine made deterministic, the terminal gathered-batch position is the only rank that still diverges (128 tokens); and without any fix, a 2,048-token prompt is already 2-distinct at the first MoE input, upstream of the combine. Grows with prompt length, plausibly because every chunk of a multi-chunk prefill repeats the exposure. Not yet localised to a specific kernel.
3. **Launch-scoped state.** Moves per-rank values by up to 0.30 across launches of an identical configuration, only for multi-chunk prompts, and between a small set of recurring discrete values. Not the collective (which is launch-invariant). Not yet localised; cold-versus-warm compilation caches is the obvious hypothesis but does not fit cleanly, since the cross-job launch was cold yet closer to the warm in-job launches than to the cold one.

The amplifier is common to all three: this model has expert-routing decisions close enough to their decision boundary on long prompts that a 1-ulp perturbation flips them, and a flip changes which experts contribute to a token by a discrete amount. That explains both the magnitude (0.03–0.29, far beyond rounding) and the discreteness (values recurring exactly across machines).

## Corrections to our own earlier claims

Recorded because both affected conclusions we had already written down.

**We reported position sensitivity as falsified; it is not.** The argument was that isolated replay and concurrent waves produce different gathered-batch layouts yet identical outputs. The trace showed the premise is false: **vLLM serialises these requests.** During what the harness submits as a concurrent wave, the DP size vector is `[1,1,1,1,128,1,1,1]` — one rank with a real chunk, seven contributing a single dummy token. Every mode we compared ran the same layout, so none of them varied position. What those modes do establish is narrower: request timing, peer content and service order are irrelevant *given the scheduling that actually occurs*.

**Our first boundary trace design was invalid for the same reason.** It captured one synchronized wave and compared call N across ranks, assuming call N was the same layer everywhere. Because the engines serialise, a shared capture compared one rank's real tokens against seven ranks' dummy tokens. The fix was to arm the trace once per serving rank, so each capture holds exactly that rank's own prefill. Every trace entry now records the DP size vector so alignment is verified rather than assumed.

## Consequences for the parity gate

The gate serves one cold launch, replays one prompt across eight ranks, and compares each to a frozen golden at a 0.075 bound. Three findings bear on it directly.

The quantity being asserted is not stable across launches: the same configuration yields at least three states, two of which fail. Separately, both GPU backends sit ~0.055 below the TPU-derived golden on this prompt for reasons outside this investigation, leaving roughly 0.02 of headroom — smaller than the launch movement we measured. And the deterministic combine, the one intervention we have measured, does not recover that headroom on long prompts.

Rank consistency and golden correctness are different properties, and a bound frozen from measured launches would test the first directly without a golden at all. The remediation options and their measured costs are laid out in the project plan; the choice among them belongs to the maintainers.

## Limitations

- One model, one checkpoint, one deployment topology. Nothing here establishes behaviour at other expert counts, world sizes, or routing configurations.
- Headline spreads come from one prompt family. The three extra prompts confirm the phenomenon exists elsewhere but do not characterise its size distribution.
- The intervention was measured in two launches at three lengths, with no throughput measurement at production concurrency. Treat its numbers as exploratory.
- Mechanisms 2 and 3 are localised but not identified. We have the position signature and the discreteness signature, not the responsible kernel or the launch-scoped variable.
- Session S1 ran a longer battery than S2/S3, so "first launch differs" is confounded with "different preceding requests". Within-launch history is demonstrably irrelevant, which makes launch state the live explanation, but three identical batteries would settle it.
- The golden's provenance remains unverified, and the ~0.055 offset it implies is out of scope here.

## Reproduction

All code is committed alongside this report: the worker-extension probe (`tests/cluster/vllm/_marin_rank_probe.py`) and its CPU-only contract check, the harness (`_experiment_c_rank_variance.py`, with arms for baseline, cross-job, trace, fixed-combine and traced-fixed-combine), the analyzer that computes every table above from raw job output (`_experiment_c_analyze.py`), and the reducers that distil a table-regenerating fixture from the raw logs. The raw data itself lives in finelog (`iris job logs /romain/<job>`); the reduced fixtures are hosted, not committed, since they are measurement data rather than code.

Three practical notes for anyone instrumenting vLLM this way. Probe methods must never raise, or they take the server down with them. Per-rank data cannot come back through the RPC response, because the DP client returns only the first engine's result. And emitted result blocks need chunking: a per-rank trace block exceeds the job log's ~49k-character line limit and arrives as unparseable JSON, which is why the analyzer counts and reports dropped blocks rather than presenting partial coverage as complete.
