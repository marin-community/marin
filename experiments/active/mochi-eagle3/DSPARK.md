# Mochi DSpark qualification design

Approved design, 2026-11-23. Running Mochi .123/.125 remain untouched.
Code lands through `~/.local/bin/loop-push`. The 128-GB200 training pair remains on hold.

## Approved serving gate: execution status

The coordinator approved one 16-GB200 EP16/TP1/BF16 gang on cw-us-east-08a at
INTERACTIVE priority, the existing 521M draft at four proposals, strict
correctness first with immediate stop on failure, then matched plain/DSpark
measurements at 16/32/48 running requests per engine.

Access was restored after the earlier sandbox failures. In-region CPU staging
`/benfeuer/mochi-dspark-stage-20261123a` succeeded. The verified-copy completion
manifest pins the checkpoint from `fb241421/2026.11.22.4`; its weights SHA-256 is
`84ba9697317b3e96a9a8416a83f11884c5cc88f8e92228aae414f7123f6178e1`.
The serving job `/benfeuer/mochi-dspark-gate-20261123a` is registered with the
coordinator and awaits successful source landing before submission.

The runtime is the released `47a8e415` wheel with the digest-pinned ShortConv
and cache-spec Python overlay from `c83c9865`. PR140 remains open; this is a
private experiment runtime and does not promote a reviewed production release.
The original EAGLE calibration hash and full-precision bounds remain frozen.
Plain checks also qualify the runtime change before candidate scoring.

The EAGLE streaming and metrics harness now supports prefetched exact-token
windows. Every scrape must contain all 16 ranks at the requested occupancy;
missing ranks, underload, preemptions or natural early stopping invalidate a cell.
The focused suite passes 27 tests. The wider landing run reported 2,030 passes,
74 failures and 18 skips. The untracked copy of this note triggered provenance
guards in several launcher tests; this note is now included in the source.
Remaining wider-suite failures are being isolated before landing.
No serving or training job has been submitted at this checkpoint.

## Decision and evidence

Use the Mochi-trained **521M one-layer DSpark** from [#12000](https://github.com/marin-community/marin/pull/12000),
frozen during qualification, starting with **four proposals**. Reuse that checkpoint
and exporter before retraining. The serving integration in
[#11975](https://github.com/marin-community/marin/pull/11975) applies; its 2,560-wide
Snowball weights do not fit 6,144-wide Mochi.

Pilot target: the adopted BF16 Hero SFT **step 400407**, tokenizer
`marin-community/marin-tokenizer-tools@3f6879acdaf3dae7874b2d48c7d11b9d8c6faf09`.
Use the regional target URI below and fresh clones of
[`mochi-throughput-train-track1.yaml` at `520b8829a8d5`](https://github.com/marin-community/marin/blob/520b8829a8d5c4da04ef65eebbca64bc8f1f9f55/experiments/post_training/long_loop/rounds/mochi-throughput-train-track1.yaml).
This is the .125 recipe template, not its live policy or allocation. The eventual
full training pair needs a separate capacity window: 48 policy plus 80 generation
GB200s per arm, run sequentially. The initial serving gate uses only one EP16 gang.

The coordinator reports tuned acceptance around 2.53 tokens/round. The accessible
#11975 body records untuned KV contention and tuned total-wall-time improvements.
These Snowball results do not forecast Mochi speed. Its long-loop port remains gated.

#12000 reports Mochi length-4 throughput of 1.471× at C1 but 0.917× at C8, with
1.87–1.91 tokens/round. Those DP8/EP8 client-concurrency tests used ≤1,536 input
and 64 greedy output tokens; they do not establish 16–48 running/engine or long
thinking performance. First-token agreement leaves continuation parity unqualified.

Evidence read: available #11975 conversation, #12000 author results, and local
source heads `c5ac5571ef6a` / `e27c2b131fc1` respectively. The web snapshot of
#11975 predates its local head; newer comments could not be retrieved through the
disconnected connector/restricted shell. The 2.53 figure remains coordinator-supplied.

## Candidate and Mochi compatibility

The primary candidate is
`s3://marin-us-east-02a/marin/checkpoints/dspark/mochi/fb241421/2026.11.22.4/hf`
([Filer](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fcheckpoints%2Fdspark%2Fmochi%2Ffb241421%2F2026.11.22.4%2Fhf)).
Verify manifest, hashes and config before staging in 08a. The 766M two-layer sibling
is a fallback experiment.
Target URI:
`s3://marin-us-east-08a/marin/models/mochi-sft/hero-sft-fullab-step393k-v2/step-400407/2af3f828aa-cpu-a1`.
Do not stage its large 02a teacher copy.

The default has one Qwen3 full-attention layer, width 6,144, FFN 4,096,
12 KV heads of dimension 128, 32k mapped vocabulary, rank-256 Markov head and
confidence head. Training block size 8 differs from proposal count 4. Feature
taps are completed blocks 2/24/45; tap 48 feeds the frozen final gated logits head.
Preserve embedding linkage, integer maps and stop IDs. The 520.9M exported weights
need about 0.97 GiB BF16; measure any additional embedding allocation.

Mochi stays GrugMoe, **TP1/BF16, DP16/EP16**. Each rank's dense draft must have EP
disabled while the target retains EP16. The November 18 `47a8e415bff1` fork supports
DSpark and dense drafts beside EP targets. Reported Mochi tests additionally used
ShortConv/cache Python changes, including [vLLM #140](https://github.com/marin-community/vllm/pull/140);
the base wheel alone is insufficient evidence. Pin a complete reviewed runtime,
verify rejection restoration, and qualify the actual deployed runner/cache path.
DSpark still requires correct Grug ShortConv state and final accepted-token routes.

Assistant-chat/tool training leaves long thinking, later turns and 64k–96k response
positions unqualified. Later RL policies may lower acceptance. If adaptation is
justified, reuse #12000's trainer (`77eaa2390cab`), gated head and grouped holdouts,
with exact generated thinking tokens. EAGLE features have different taps and
normalization and cannot be substituted.

## Draft length and KV budget

Start at **4**; consider 2 after CPU/runtime validation. Defer 8 until measured
headroom and benefit justify it. Longer drafts expand verification, speculative
cache reservations and workspaces; shortening them does not remove history KV.

An architecture estimate for one draft layer is
`2(K,V) × 2(BF16 bytes) × 12 KV heads × 128 = 6 KiB/context token`.
With no sharing between requests, incremental draft KV per engine is:

| Context per request | 16 running | 32 running | 48 running |
| --- | ---: | ---: | ---: |
| 8,192 tokens | 0.75 GiB | 1.50 GiB | 2.25 GiB |
| 65,536 tokens | 6 GiB | 12 GiB | 18 GiB |
| 98,304 tokens | 9 GiB | 18 GiB | 27 GiB |

Add weights, target KV/ShortConv state, lookahead, allocator padding and workspaces;
these estimates do not prove capacity. Two layers double this draft-KV term.
Match the recipe's HBM fraction and context/sequence/batch limits in both arms
(0.90 only if its receipt agrees). Record KV capacity, context mix, queues and
preemptions. Do not lower concurrency, truncate thinking or change precision/TP
to make an arm fit.

## Expected benefit

Working hypothesis: roughly **1.8–2.2 tokens/round** for the existing Mochi draft,
with unmeasured speed at 16–48 requests/engine; regression remains plausible.
An exploratory 1.0–1.3× speed range is not a prediction or a pass. EAGLE achieved
1.67 tokens/round yet only 0.66× speed at load 16. Snowball's tuned 2.53 and
Mochi DSpark's single-client improvement cannot establish loaded speed or a
20-minute 96k response.

## Qualification sequence

1. **Preflight and source gate.** Inspect current main/branches and reuse the
   Snowball port's default-off binding. Pin target, tokenizer, draft/config/maps,
   wheel/patch hashes, runner, scheduler, prefix/Mamba caches and graph settings.
   Keep compilation and CUDA graphs enabled. Extend kernel and compiled model
   parity tests through every rejection position for four proposals plus the
   verifier row, including ragged and mixed prefill/decode batches. Verify exact
   emitted-token/route alignment and EOS/tool boundaries against non-speculation.
   Any precision or TP change requires a logprob-mismatch check first.
   A wheel/runner change also needs an old/plain versus new/plain logprob check.

2. **Plain controls before DSpark.** Reuse `mochi_eagle3_evaluate.py`'s identical-
   prefix scoring, sampled checks and final plain-teacher rescore. Use held-out
   canonical tasks excluded from both draft-training corpora and model-selection
   sets, plus long-thinking/late-position fixtures. Retain the full-precision EAGLE
   limits from its stored calibration; rounded here:
   fixed max logprob error 0.243823, top-1 mismatch ≤1/98, expert-choice mismatch
   ≤2.8723%; autoregressive limits 0.559715 / 0.261437 for known-token/top-1
   logprob errors, top-1 mismatch ≤7/750, expert-choice mismatch ≤4.0781%.
   Expert-choice error is the fraction of eight reference IDs absent from the
   candidate per layer, averaged within each canonical prompt, then equally over
   prompts. Retain slot-order diagnostics. Require the baseline noise upper bound
   (prompt mean + three standard errors + one-slot floor) ≤5%. For sampled
   logprob/top-1 mass, average seeds 17/23/29 within prompts and compare prompt
   means against the plain mean ±three prompt standard errors, with natural
   stopping. Independent plain repeats across batches and
   allocations must pass before candidate scoring; final plain rescoring follows
   the candidate. EAGLE's fresh plain repeat failed 0.259718 > 0.243823: if this
   recurs, stop as unqualified. Do not raise bounds or retry for a lucky pass.

3. **Serving at real load.** Reuse the EAGLE streaming harness, rank-pinned clients
   and labeled occupancy gauges on an isolated 16-GB200 gang. Test actual
   **16/32/48 running per engine**, not just global client counts; one-request
   results are diagnostic. Use the real rollout prompt/length mix and long
   streams or the exact-token window prototype to avoid the 256-token request
   waves that missed EAGLE's high-load targets. Qualify that measurement protocol
   before comparing arms. Counterbalance fresh plain/DSpark repeats on the same
   allocation; include all tuning trials. Report per-request decode rate, TPOT
   p50/p95, TTFT, rollout-group tails, errors, KV/preemptions and acceptance by
   context/response position. Acceptance is `1 + accepted_tokens / draft_rounds`;
   proposal acceptance is separately `accepted_tokens / proposed_tokens`.
   Unachieved occupancy or changed budgets cannot support a matched speed ratio.
   This serving gate must pass ≥1.3× at 16/32 and ≥1.0× at 48 before step 4.

4. **Matched training-load gate.** After separate authorization for an isolated
   training gate, clone the intended long-loop recipe into plain and frozen-
   DSpark arms with identical target/data, TP1/BF16, load, token budgets, tools,
   staleness, evaluation and weight-sync cadence. Never modify .123/.125. Require
   improved matched update/rollout wall time and policy agreement. Recompute
   mismatch/KL against the recorded serving-policy version;
   use equally sized prompt-group samples with comparable stamp coverage. Freeze
   the allowed KL/mismatch upper bound from plain controls (mean + three group
   standard errors) before the candidate; require group p95 ≤plain p95.
   Insufficient same-policy coverage leaves the training gate unqualified.
   Report startup-inclusive total wall time separately from steady
   updates, retaining verifier/evaluation/checkpoint waits and failures. A faster
   isolated decode or higher acceptance does not pass this gate.

5. **Adoption.** Register every future Iris job with the coordinator before
   submission; stage large artifacts through an 08a CPU job directly into regional
   S3. No weights/features on the Mac. Report one line per milestone and measured
   acceptance, latency and wall time before wiring. Only after correctness,
   loaded performance and reviewed wheel adoption pass, add an opt-in immutable
   Mochi draft binding through `loop-push`; keep the existing rounds unchanged.

Source details: [offline recipe at the inspected head](https://github.com/marin-community/marin/blob/e27c2b131fc1c5aa3101b4f99229add0cb83299d/docs/recipes/offline-dspark.md),
[draft config construction](https://github.com/marin-community/marin/blob/e27c2b131fc1c5aa3101b4f99229add0cb83299d/lib/marin/src/marin/training/drafting/dspark_data.py),
[EAGLE result](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-08a%2Fmarin%2Fusers%2Fbenfeuer%2Fmochi-eagle3%2F2026.11.23.1%2Fresult.json)
(`s3://marin-us-east-08a/marin/users/benfeuer/mochi-eagle3/2026.11.23.1/result.json`).
