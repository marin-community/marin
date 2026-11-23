# Mochi DSpark qualification design and result

## Serving gate result, 2026-11-23

**Unqualified: the final autoregressive top-1 mismatch check failed.** The gate
stopped before any performance cells, as required. There is no measured
acceptance length, per-request speedup or TPOT at 16/32/48 running per engine.
No draft binding was wired, no RL job launched, and running .123/.125 were untouched.

| Final independent plain-teacher check | Plain | DSpark4 | Frozen upper bound |
| --- | ---: | ---: | ---: |
| Top-1 mismatch | 6/746 = 0.80429% | **8/748 = 1.06952%** | **7/750 = 0.93333%** |
| Maximum known-token logprob error | 0.259173 | 0.276580 | 0.559715 |
| Maximum top-1 logprob error | 0.179650 | 0.184409 | 0.261437 |
| Equal-prompt expert-choice mismatch | 0.025010 | 0.024322 | 0.040781 |

The 30-case ShortConv sequential oracle passed on all four tasks, covering
FP32/BF16, ragged verifier rows and rejection/acceptance through all five rows
for four proposals. Initial and independent plain controls passed. DSpark's
98 fixed-prefix comparisons passed (maximum logprob error 0.119401, zero top-1
mismatches); its sampled-distribution check passed. The final teacher check above
failed only the top-1 mismatch fraction. The full-precision bounds and calibration
hash were unchanged; there was no retry to obtain a passing sample.

Offline inspection found two byte-identical token prefixes with different top-1
IDs in the retained plain-teacher responses. Five of the eight DSpark mismatches
cluster in one original prompt group. This motivates a batch/cache/numerical
control investigation; it neither identifies the cause nor overrides the failure.

The measured runtime memory budgets were:

| Per GPU/engine at HBM 0.90 | Plain | DSpark4 |
| --- | ---: | ---: |
| Model loading allocation, GiB | 86.42 | 87.52 |
| CUDA graphs, GiB | 1.46 | 3.06 |
| Available KV cache, GiB | 58.45 | 54.32 |
| KV token capacity per engine | 687,808 | 587,205 |

DSpark reduced the token capacity by 14.6%. These are startup measurements,
not acceptance or speed measurements. Compilation and PIECEWISE CUDA graphs
remained enabled throughout; serving stayed TP1/BF16 with DP16/EP16.

The registered jobs were CPU staging `/benfeuer/mochi-dspark-stage-20261123a`
(succeeded), serving `/benfeuer/mochi-dspark-gate-20261123a` (setup-only failure
on an optional `worker_region=None` value), and replacement
`/benfeuer/mochi-dspark-gate-20261123b` (strict qualification failure, all four
tasks terminal). The replacement enforced a hard US-EAST-08A scheduling constraint
and target cluster 08a. Only one 16-GB200 gang was allocated at a time, at
INTERACTIVE priority, excluding r136, with zero automatic retries. There is no further
serving or training job running for this gate.

Code landed only through `loop-push`; the measured source was
`26fa9a19b9f859b07364563b0dac92d6b05b0795`. The runtime was the released `47a8e415`
wheel with digest-pinned ShortConv/cache Python files from `c83c9865`, in a private
isolated environment. PR140 remains open; this does not promote a reviewed
production release. Runtime receipts retain exact dependencies, including
Torch 2.14.1+cu132, Transformers 5.16.1 and Speculators 0.8.0+marin.1.
The staged 521M `fb241421` weights SHA-256 is
`84ba9697317b3e96a9a8416a83f11884c5cc88f8e92228aae414f7123f6178e1`.

The 29 focused gate/launcher tests passed before the first serving submission;
27 serving-focused tests passed after the optional-metadata fix. Required
mechanical lint/type checks passed. The broader affected run reported 2,030
passes, 74 failures and 18 skips before the untracked-design provenance guard was
fixed. Two SFT failures reproduced on the clean prototype base; the full broader
suite was not rerun to green. This is not a claim that all repository tests pass.

[Machine-readable result](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-08a%2Fmarin%2Fusers%2Fbenfeuer%2Fmochi-dspark%2F2026.11.23.1%2Fserving-gate-result.json)
(`s3://marin-us-east-08a/marin/users/benfeuer/mochi-dspark/2026.11.23.1/serving-gate-result.json`; SHA-256
`a1d591b68b583c9113ae2d73bab8056c6ea7af0febcfd0d3199209203e5e12fc`).
Raw scores and logs remain under `s3://marin-us-east-08a/marin/users/benfeuer/mochi-dspark/2026.11.23.1/evaluation-b/`; the result links the
retained mismatch diagnostics. Results and datasets are stored in S3, not committed.

## Next cycle: staged comparison and RL-pair launch plan

No additional compute is authorized by this plan. The `fb241421`/four-proposal gate stopped on final top-1 mismatch (8/748, above the frozen 7/750 fraction); no speed cells ran. Any failed correctness check blocks tuning and training; retain the failed samples and diagnose the runtime without raising the frozen bounds. Production adoption also requires the reviewed runtime and its consumer checks.

### Resolve correctness before tuning

The final plain control passed at 6/746 mismatches. Retained teacher responses vary on two byte-identical prefixes, and five of the eight candidate mismatches cluster in one prompt group. These observations do not identify a cause. Preserve the exact failing prefixes and served/teacher scores. Before any new registered diagnostic GPU job, predeclare matched plain/plain and plain/DSpark batches, cache state, query shapes and rejection-depth coverage. Investigate batch/cache/numerical variation and accepted-state alignment under TP1/BF16 with graphs enabled. Keep the frozen thresholds; do not retry unchanged settings for a lucky pass. Tuning, checkpoint selection and RL remain blocked until this gate passes.

### Match the training request and context mix

The planned performance payload retained temperature 0.7, top-p 1, natural EOS, exact token IDs and routed experts, but omits the training request's `logprobs: 0`. Its six cap-conditioned groups use 8,573–12,733-token contexts and 90-second censored decode windows, rather than complete 98,304-token responses. That protocol would measure isolated decode; this failed gate produced no ratios. The next comparison must request the chosen-token logprobs, retain the same protocol qualification, and freeze a representative prompt mix with canonical prompt hashes checked against both drafter training/model-selection sets. Include a bounded mixture of early and >64k late contexts; do not fill all 48 slots with 100k contexts that cannot fit the measured KV budget.

### Compare checkpoints before expanding the tuning matrix

The [newer #12000 export](https://github.com/marin-community/marin/pull/12000#issuecomment-7113277871) is `s3://marin-us-east-02a/marin/checkpoints/dspark/mochi/e43c1b81/2026.11.22.4/hf` ([Filer](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fcheckpoints%2Fdspark%2Fmochi%2Fe43c1b81%2F2026.11.22.4%2Fhf)). Its config SHA-256 is identical to `fb241421`: `737eb5cb3bba8746ba70b723b8d25fb86afc835599d5c97f351c66cc3ac2fe9e`. Both weight files are 1,041,946,938 bytes; the newer source ETag is `6cd5263b085d1b182da94fec502a7ba1-20`. Its producer metadata SHA-256 is `cc1ad36a59aa63f2ee59be6bfc99cfd3b6508d6fd7e1b38efa2165732728898e`. It records 8,040 steps, validation loss 0.600466 and trainer acceptance length 2.72867 at block size 8. These are not measurements at Mochi's rollout concurrency.

After approval, register an 08a CPU staging job using the existing `fsutil verified-copy` path. Verify producer/config hashes and source identity, then require the destination completion manifest and a full streamed weight SHA-256 before serving. Keep weights off the Mac. Compare old and new checkpoints at four proposals on one isolated EP16/16-GB200 INTERACTIVE gang, with the same runtime, target, prompt inventory, seeds and common HBM budget. Run the unchanged strict controls first, then matched plain/old/new cells at 16/32/48. Counterbalance repeats; report complete trial history. Inspect and record vocabulary-map identity in-region; matching config and file size alone do not isolate training-token count from map/data changes. The previous short greedy C8 result does not establish a real-load winner.

For the checkpoint that passes correctness and improves request decode rate, test proposals 2/3/4. Parameterize the existing harness, preserving the sequential ShortConv oracle for each verifier depth. Keep TP1/BF16, prefix-cache path, max sequences and sampler fixed. Start at common HBM 0.90; consider a matched lower-HBM arm only as an explicit separate configuration. Record target/draft/graph memory and actual KV token capacity, and invalidate any cell with missing ranks, underload or preemptions. Current measured capacity is 687,808 tokens/engine plain versus 587,205 with four proposals; the latter gives only about 12,233 tokens per request at 48-way occupancy without sharing. Rank configurations by measured request speed and TPOT, rather than acceptance alone. Budget and register each future gang separately; no extra serving job is launched here.

### Prepare the RL pair for the user's next-day decision

Clone the frozen `.125` Track1 recipe into new immutable plain/DSpark versions and regional output roots. Preserve the adopted Mochi step-400407 target, tokenizer, task pool, TP1/BF16 serving, CP/PP/EP training topology, GS8/batch64, context and 98,304-response cap, staleness, weight-sync cadence, checkpointing and evaluation. Change only the qualified draft binding. Add that binding only after reporting serving acceptance and speed; do not edit or resume .123/.125.

The existing role plan is 48 policy GPUs plus five EP16 rollout engines (80 GPUs): **128 GPUs per arm**. Run the arms serially to keep the maximum allocation at 128; concurrent arms would need 256 and separate approval. The current recipe allows 32 updates with a 10-hour training budget. Propose an explicit 12-hour job cap per arm, including startup/drain, for a maximum 3,072 GPU-hours across the serial pair; the user decides that budget and priority before launch. A shortened run that cannot complete the required matched updates remains unqualified.

Use the existing artifact entry point, not a new submission path:

```bash
uv run python -m experiments.post_training.long_loop.round_runner \
  --round <new-immutable-plain-or-dspark-round.yaml> \
  --source-revision <loop-push-landed-sha> \
  --validation-output <arm-validation-directory> \
  --version <matching-new-round-version> --through rl
```

Before requesting launch approval, render both plans, strictly compose the native SkyRL documents, verify the only treatment difference is the draft, and seal runtime/target/data/draft hashes. The SkyRL runtime must contain the same qualified ShortConv fixes in its own isolated environment; the serving wrapper alone does not establish that. Keep probabilistic drafting and standard rejection sampling. Qualify policy logprobs and routes against the recorded policy version before rollout and after weight sync. Register coordinator and child job IDs before compute. Add `--run` only after the separate approval.

Require at least six completed updates per arm, with update one excluded from steady metrics and two regular checkpoint/evaluation intervals covered, plus matched policy-stamp coverage, improved median update/rollout-group time without a worse group p95, stable fresh-policy mismatch/KL under the plain-control bounds, and no quality or preemption regression. Report startup-inclusive wall time, steady update time, acceptance, request TPOT, group tails, evaluation/checkpoint waits and failures separately. Isolated decode speed does not pass the training gate. At 7 tokens/s, 98,304 tokens in 20 minutes requires about 11.7x speedup; acceptance length alone cannot support that forecast.

## Original approved design

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
