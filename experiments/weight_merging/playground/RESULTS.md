# Verified playground results

Status: experiments are unfinished. The curves and new independent-sample donor gates are verified; far-donor controls remain in progress. Math passes the joint-precision gate and code does not, so the two-donor grid has not run. No two-donor method ranking is established.

All curves below use eager vLLM execution, output cap **8192**, context **12288**, the full 1319-item GSM8K test and 257-item sanitized MBPP test, and greedy plus sampled seeds 17/29. Each gain averages three decodes per item before a paired item bootstrap (5000 resamples, seed 290). Intervals are pointwise exploratory 95% CIs, without correction for recipe selection. Each precision has its own Instruct control. Separate BF16 and FP32 tables change both weight casting and compute; their difference does not isolate either cause.

MBPP scores use the frozen strict extractor and Docker sandbox. Scoring used one scorer per exclusive healthy host. Both generation and correctness of shared controls matched exactly across the scoring shards. Earlier shared-host grades are excluded from these tables. Truncation and lengths below are greedy counts/means; unclosed thinking counts all three decodes.

The original math RFT donor used duplicated responses because seeded `n>1` requests in vLLM 0.6.6.post1 copied the same seed to each child. Its evaluations use `n=1` and remain valid. New training uses separately seeded `n=1` requests, a diversity probe, and exact response deduplication; its fresh graph-mode gates use their own controls and are reported separately below.

The main observations are:

- Partial Aristaeus transfer eliminates its unclosed thinking on MBPP while retaining a modest GSM gain. Strict code effects vary, with several intervals including zero.
- Shrinking the original math RFT update to c=0.25 or 0.5 improves GSM more than the full donor. Clean MBPP does not establish a regression at those points.
- Single-donor TIES is magnitude pruning, without sign election. Density 0.3 retains almost all Aristaeus update energy, so the lower-density controls matter. No single method wins generally.

## Aristaeus, BF16: max_tokens=8192

| Candidate | GSM greedy /1319 | MBPP greedy /257 | GSM gain pp [CI] | MBPP gain pp [CI] | Truncated GSM / MBPP | Mean tokens GSM / MBPP | Unclosed think MBPP /771 |
|---|---:|---:|---:|---:|---:|---:|---:|
| instruct | 946 | 157 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 3 / 0 | 249.1 / 133.6 | 0 |
| aristaeus | 994 | 15 | +2.63 [+1.01, +4.27] | -52.14 [-56.94, -47.21] | 43 / 168 | 512.2 / 5581.0 | 700 |
| aristaeus-average-c0.25 | 956 | 154 | +1.31 [+0.03, +2.58] | +0.13 [-2.20, +2.46] | 9 / 1 | 280.7 / 132.2 | 0 |
| aristaeus-average-c0.5 | 961 | 149 | +1.11 [-0.33, +2.55] | -3.11 [-6.23, -0.26] | 8 / 0 | 284.2 / 88.5 | 0 |
| aristaeus-average-c0.75 | 972 | 157 | +1.95 [+0.40, +3.46] | -2.20 [-5.84, +1.43] | 21 / 3 | 375.9 / 286.2 | 0 |
| aristaeus-dare_linear-c0.25 (d=0.3) | 940 | 158 | -0.18 [-1.34, +0.99] | +0.26 [-1.69, +2.20] | 8 / 0 | 274.8 / 119.1 | 0 |
| aristaeus-dare_linear-c0.5 (d=0.3) | 951 | 157 | +0.88 [-0.40, +2.15] | +0.00 [-2.46, +2.46] | 10 / 0 | 287.9 / 96.5 | 0 |
| aristaeus-dare_linear-c0.75 (d=0.3) | 951 | 156 | +0.61 [-0.81, +2.02] | -1.56 [-4.15, +1.04] | 11 / 1 | 300.2 / 112.4 | 0 |
| aristaeus-dare_linear-energy0.5-c0.5 | 940 | 154 | +0.28 [-0.83, +1.36] | +0.39 [-1.30, +2.08] | 9 / 1 | 281.1 / 168.8 | 0 |
| aristaeus-dare_linear-energy0.7-c0.5 | 940 | 157 | -0.03 [-1.09, +1.04] | +0.00 [-1.82, +1.82] | 7 / 1 | 270.6 / 151.3 | 0 |
| aristaeus-dare_linear-energy0.9-c0.5 | 934 | 159 | -0.23 [-1.31, +0.91] | +0.00 [-2.08, +1.95] | 5 / 0 | 261.7 / 123.9 | 0 |
| aristaeus-magnitude_topk-c0.25 (d=0.3) | 951 | 156 | +1.21 [-0.10, +2.50] | +0.13 [-2.33, +2.59] | 12 / 1 | 306.1 / 133.4 | 0 |
| aristaeus-magnitude_topk-c0.5 (d=0.3) | 972 | 154 | +1.19 [-0.23, +2.58] | -1.82 [-4.93, +1.17] | 7 / 2 | 279.8 / 144.9 | 0 |
| aristaeus-magnitude_topk-c0.75 (d=0.3) | 964 | 151 | +1.72 [+0.18, +3.29] | -4.15 [-7.78, -0.65] | 21 / 2 | 377.9 / 250.8 | 0 |
| aristaeus-magnitude_topk-energy0.5-c0.5 | 949 | 158 | +0.86 [-0.51, +2.22] | +0.26 [-2.08, +2.46] | 13 / 2 | 306.7 / 167.0 | 0 |
| aristaeus-magnitude_topk-energy0.7-c0.5 | 952 | 155 | +1.54 [+0.10, +2.91] | -0.52 [-2.85, +1.69] | 14 / 0 | 315.5 / 103.9 | 0 |
| aristaeus-magnitude_topk-energy0.9-c0.5 | 963 | 155 | +1.01 [-0.30, +2.35] | -0.78 [-3.50, +1.82] | 12 / 2 | 305.1 / 146.1 | 0 |

Artifact prefix: `s3://marin-rno2a/scratch/karan/task290/remote/aristaeus-bf16-final-v1/`. Its `done.json` authenticates the audit or publication manifest. The math audit records raw-source SHA256s and both-control equality across three exclusive hosts; Aristaeus publication manifests link the independently audited scoring shards.

## Aristaeus, FP32: max_tokens=8192

| Candidate | GSM greedy /1319 | MBPP greedy /257 | GSM gain pp [CI] | MBPP gain pp [CI] | Truncated GSM / MBPP | Mean tokens GSM / MBPP | Unclosed think MBPP /771 |
|---|---:|---:|---:|---:|---:|---:|---:|
| instruct | 939 | 156 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 6 / 0 | 264.9 / 137.2 | 0 |
| aristaeus-average-c0.25 | 951 | 155 | +1.59 [+0.33, +2.81] | -1.17 [-3.50, +1.17] | 10 / 0 | 284.0 / 103.9 | 0 |
| aristaeus-average-c0.5 | 973 | 155 | +2.07 [+0.63, +3.51] | -2.33 [-5.45, +0.65] | 6 / 0 | 271.1 / 100.6 | 0 |
| aristaeus-average-c0.75 | 966 | 159 | +1.82 [+0.25, +3.36] | -3.76 [-7.00, -0.52] | 16 / 0 | 346.1 / 202.1 | 0 |
| aristaeus-dare_linear-c0.5 (d=0.3) | 951 | 159 | +1.14 [-0.08, +2.40] | +0.13 [-2.46, +2.72] | 11 / 0 | 290.6 / 102.5 | 0 |
| aristaeus-dare_linear-energy0.5-c0.5 | 929 | 154 | -0.03 [-0.99, +0.94] | +0.00 [-1.43, +1.43] | 9 / 0 | 279.7 / 130.9 | 0 |
| aristaeus-dare_linear-energy0.7-c0.5 | 940 | 156 | +0.35 [-0.66, +1.36] | -0.13 [-1.82, +1.56] | 10 / 0 | 290.5 / 125.8 | 0 |
| aristaeus-dare_linear-energy0.9-c0.5 | 942 | 157 | +0.38 [-0.68, +1.49] | -0.39 [-2.33, +1.43] | 8 / 0 | 280.9 / 127.2 | 0 |
| aristaeus-magnitude_topk-c0.5 (d=0.3) | 973 | 155 | +2.07 [+0.63, +3.51] | -2.33 [-5.45, +0.65] | 6 / 0 | 271.1 / 100.6 | 0 |
| aristaeus-magnitude_topk-energy0.5-c0.5 | 955 | 160 | +1.57 [+0.25, +2.88] | -0.52 [-2.85, +1.82] | 8 / 1 | 275.0 / 136.8 | 0 |
| aristaeus-magnitude_topk-energy0.7-c0.5 | 958 | 156 | +1.54 [+0.20, +2.93] | -0.52 [-2.85, +1.69] | 6 / 1 | 265.6 / 138.4 | 0 |
| aristaeus-magnitude_topk-energy0.9-c0.5 | 950 | 154 | +1.14 [-0.23, +2.53] | -1.30 [-4.15, +1.43] | 9 / 1 | 290.6 / 115.0 | 0 |

Artifact prefix: `s3://marin-rno2a/scratch/karan/task290/remote/aristaeus-fp32-final-v2/`. Its `done.json` authenticates the audit or publication manifest. The math audit records raw-source SHA256s and both-control equality across three exclusive hosts; Aristaeus publication manifests link the independently audited scoring shards.

## Original math RFT, FP32: max_tokens=8192

| Candidate | GSM greedy /1319 | MBPP greedy /257 | GSM gain pp [CI] | MBPP gain pp [CI] | Truncated GSM / MBPP | Mean tokens GSM / MBPP | Unclosed think MBPP /771 |
|---|---:|---:|---:|---:|---:|---:|---:|
| instruct | 939 | 156 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 6 / 0 | 264.9 / 137.2 | 0 |
| math_rft | 949 | 153 | +0.58 [-1.09, +2.25] | +0.65 [-2.59, +3.63] | 9 / 0 | 240.3 / 128.2 | 0 |
| math-rft-average-c0.25 | 983 | 153 | +3.29 [+1.84, +4.68] | +0.78 [-1.56, +3.24] | 8 / 0 | 254.1 / 149.7 | 0 |
| math-rft-average-c0.5 | 976 | 151 | +3.13 [+1.72, +4.57] | +0.52 [-2.33, +3.37] | 9 / 0 | 248.9 / 147.7 | 0 |
| math-rft-average-c0.5-rounded-donor | 993 | 152 | +3.41 [+1.97, +4.85] | +0.52 [-2.08, +3.11] | 8 / 0 | 255.6 / 150.0 | 0 |
| math-rft-average-c0.75 | 960 | 152 | +2.32 [+0.76, +3.87] | +1.04 [-1.95, +3.89] | 13 / 0 | 267.3 / 138.0 | 0 |
| math-rft-dare_linear-d0.01188502227437019-c0.5 | 969 | 155 | +2.30 [+1.09, +3.49] | +0.91 [-0.65, +2.46] | 5 / 0 | 247.4 / 152.2 | 0 |
| math-rft-dare_linear-d0.022387211385683378-c0.5 | 967 | 155 | +2.60 [+1.34, +3.82] | +0.91 [-0.91, +2.72] | 5 / 0 | 247.1 / 149.1 | 0 |
| math-rft-dare_linear-d0.04731512589614803-c0.5 | 985 | 155 | +3.06 [+1.74, +4.37] | +0.78 [-1.17, +2.72] | 5 / 0 | 244.7 / 151.3 | 0 |
| math-rft-dare_linear-d0.3-c0.5 | 984 | 153 | +3.49 [+2.07, +4.88] | +0.78 [-1.69, +3.24] | 11 / 0 | 266.5 / 149.6 | 0 |
| math-rft-magnitude_topk-d0.01188502227437019-c0.5 | 964 | 156 | +2.68 [+1.36, +3.99] | +1.95 [+0.00, +3.89] | 3 / 0 | 244.0 / 153.1 | 0 |
| math-rft-magnitude_topk-d0.022387211385683378-c0.5 | 975 | 153 | +2.70 [+1.39, +4.04] | +1.04 [-1.04, +2.98] | 7 / 0 | 266.7 / 152.6 | 0 |
| math-rft-magnitude_topk-d0.04731512589614803-c0.5 | 989 | 155 | +3.18 [+1.87, +4.55] | +0.78 [-1.30, +2.98] | 10 / 0 | 279.8 / 152.1 | 0 |
| math-rft-magnitude_topk-d0.3-c0.5 | 996 | 153 | +3.31 [+1.90, +4.75] | +0.52 [-2.20, +3.11] | 10 / 0 | 268.2 / 150.0 | 0 |

Artifact prefix: `s3://marin-rno2a/scratch/karan/task290/remote/math-fp32-own-donor-audit-v1/`. Its `done.json` authenticates the audit or publication manifest. The math audit records raw-source SHA256s and both-control equality across three exclusive hosts; Aristaeus publication manifests link the independently audited scoring shards.

### Why FP32 average and top-k c=0.5 rows match

The two FP32 rows are genuine identical observed behavior, rather than copied statistics: their canonical item/token/text/finish payload SHA256 is `4d070783c50190ff1619b10e398545d93c98a242bd5001bebd6a5ad0ffad949b` for all 4728 outputs. Raw file hashes differ because candidate labels differ. Their BF16 runs differ on 2689/4728 token sequences.

Reconstruction from the pinned inputs and shared frozen arithmetic gives different tensor-content hashes in both precisions. There are 36,350,708 differing FP32 entries, but the difference norm is only 0.03534% of the average update norm. After BF16 casting, 2,796,645 entries differ; 2,761,459 (98.742%) are exact BF16 midpoints in the average merge, and 99.591% of the changes move toward the donor. The norm-matching scale 0.500000031216137 becomes 0.5000000596046448 in FP32, tipping midpoint ties. **c=0.5 in BF16 is a tie-dominated special case in this comparison.** The delta is 85.67% exactly zero globally, but tensor-local top-k can still remove nonzeros in individual tensors.

No historical per-candidate checkpoint exports exist because serving loaded merged tensors in place. These are reconstructed sorted named-tensor content hashes, not hashes captured during serving. The raw marker/file hashes, canonical payload hashes, all four reconstructed weight hashes, exact midpoint counts and source hashes are in `s3://marin-rno2a/scratch/karan/task290/remote/aristaeus-duplicate-audit-v1/audit.json`, authenticated by its `done.json`. The matching FP32 rows provide no evidence of a top-k score advantage.

## Controlled precision comparisons

| Comparison, fixed FP32 serving | GSM gain pp [95% CI] | Strict MBPP gain pp [95% CI] |
|---|---:|---:|
| Original math c=0.5, rounded donor minus FP32 donor | +0.28 [−0.81, +1.36] | 0.00 [−1.43, +1.43] |
| Aristaeus c=0.5, BF16-rounded final weights minus FP32 final weights | −0.13 [−0.33, +0.05] | Pending direct item-bootstrap contrast |

These comparisons round different objects: the first rounds the donor before delta arithmetic; the second rounds the completed merge. Neither establishes a GSM effect. BF16 final casting erased most of the small math update, which is why the math curve uses FP32 throughout.

The Aristaeus top-k densities 0.04731512589614803, 0.022387211385683378 and 0.01188502227437019 retain about 90%, 70% and 50% of its squared update L2. At the same numeric densities, math retains about 26.84%, 15.80% and 9.88%. Math density 0.3 retains about 78.01%. Sparse labels c0.25/c0.5/c0.75 target the L2 norm of that coefficient times the full donor update. All added energy-density rows and math sparse rows target c=0.5. Energy labels identify densities chosen from Aristaeus top-k retention; DARE at those densities does not claim the same retained energy. A shared density does not imply shared retained energy.

Aristaeus is `EphAsad/Aristaeus` @ `f6e918c83cda9aa8ec57ab8d57e9a557b8190b00`, a public full SFT from the pinned Instruct anchor on OpenThoughts3/Stratos. The full donor endpoint was evaluated in BF16 only; the FP32 curve cannot establish its retained fraction of a full FP32 donor gain or its full-donor FP32 code behavior. In BF16 serving, FP32 merge weights are cast before computation; FP32 serving keeps FP32 merge weights and computation. Instruct on-disk weights are BF16, so its FP32 baseline upcasts the same exact stored values.

## Independent k8 training and admission

The pinned Instruct parent generates training candidates from 1024 filtered GSM8K-train prompts and 365 filtered MBPP full-training prompts. The fixed sampler generates eight separately seeded n=1 requests per prompt. A controlled 20-prompt test found mean distinct completions of 1.00/8 for seeded n=8, 7.95/8 for unseeded n=8, and 7.85/8 for independently seeded n=1. The production probe passed at 7.9/8 against a minimum of five. Global prompt indices preserve request seeds under chunking and resume.

Generation produced 11112 candidates from 1389 prompts. Answer verification and sandbox execution, including code challenge assertions absent from the prompt, preceded training eligibility. Exact duplicate prompt-response pairs were removed. Each domain selected 1024 distinct training pairs and 128 validation pairs after a source-question split and test-question overlap filtering. Within each source-question split, verified length-eligible pairs are shuffled with seed 290 and the first requested count is selected; selection is over response pairs, not balanced across prompts. Selected training pairs cover 603 math prompts and 213 code prompts; 213 of 215 eligible code training prompts were represented. Execution checks establish consistency with available assertions, not correctness on every possible input.

Both new donors start from Instruct, with full FP32 parameters/exports, BF16 autocast, microbatch 1 × accumulation 8, LR 5e-6, cosine decay, 5% warmup, zero weight decay, clipping 1, maximum training length 1024, and 100 maximum updates. Both use Torch 2.6.0, Transformers 4.57.6, bitsandbytes 0.45.5 and Liger 0.6.4 with fused linear cross entropy. A checkpoint smoke verified exact optimizer state and FP32 weight reload on this stack. Validation/save occur at steps 50 and 100. Math selects step100 (loss .156467 versus .161719 at50); code selects step50 (.108353 versus .109742 at100). Thus the selected exports saw 800 and 400 examples, respectively: equal training budgets did not yield equal exported update counts. Both selected exports match the saved checkpoint weight hashes.

Admission requires a positive own-domain item-mean gain in both serving precisions and a paired 95% interval above zero in at least one. Each precision has freshly generated Instruct and donor baselines. BF16 serving casts the FP32 donor/merge weights to BF16 and uses BF16 computation; FP32 serving keeps FP32 weights and computation. Instruct starts from BF16 on-disk weights, so its FP32 baseline upcasts the same exact weight values. Controlled donor-rounding rows instead round the donor before arithmetic and serve the resulting merge in FP32. Final-weight-rounding controls round the completed merge and upcast it before FP32 serving. These gates use graph mode; the older curves below use eager mode and are never mixed with them. Math passes; code does not. The code result is insufficient evidence of improvement, not evidence of a harmful donor.


| Donor / own domain | BF16 gain pp [95% CI] | FP32 gain pp [95% CI] | Admission |
|---|---:|---:|---|
| math_rft / GSM8K | +3.77 [+2.27, +5.23] | +1.97 [+0.35, +3.64] | Pass |
| code_rft / MBPP | +0.00 [-3.24, +3.11] | +0.52 [-2.98, +4.02] | Fail |

Instruct precision controls change GSM greedy totals 940→939 but flip correctness on 91 items and token sequences on 834/1319. MBPP totals 157→156 hide 9 correctness flips and 72/257 sequence changes. This is a precision noise reference, not an uncertainty estimate for the paired treatment effects.

### New graph-mode gate rows: max_tokens=8192

| Precision / candidate | GSM greedy /1319 | MBPP greedy /257 | Truncated GSM / MBPP | Mean tokens GSM / MBPP |
|---|---:|---:|---:|---:|
| BF16 / instruct | 940 | 157 | 3 / 0 | 252.5 / 134.6 |
| BF16 / math_rft | 963 | 154 | 7 / 3 | 253.6 / 240.2 |
| BF16 / code_rft | 948 | 153 | 3 / 0 | 286.6 / 129.1 |
| FP32 / instruct | 939 | 156 | 6 / 0 | 264.9 / 137.2 |
| FP32 / math_rft | 943 | 159 | 6 / 0 | 219.2 / 139.9 |
| FP32 / code_rft | 927 | 154 | 5 / 0 | 296.2 / 113.7 |

The new donors are very near Instruct: relative model-wide L2 distances 0.00069594 (math) and 0.00059958 (code). At c=0.5, BF16 output casting preserves only 12.45%/11.03% of FP32-changed entries, with error norms 87.92%/89.32% of the FP32 update. Counts of surviving entries and update-norm error measure different aspects of the cast. Keeping merges in FP32 avoids erasing these small updates.


### New-donor paired comparisons

| Donor / precision | Greedy W/L; exact sign p | Pooled W/L; descriptive sign p |
|---|---:|---:|
| math_rft / bfloat16 | 127/104; 0.1476 | 460/311; 8.9872e-08 |
| math_rft / float32 | 133/129; 0.85301 | 466/388; 0.0083786 |
| code_rft / bfloat16 | 15/19; 0.60759 | 56/56; 1 |
| code_rft / float32 | 21/23; 0.8804 | 69/65; 0.79563 |

Admission concerns the per-item three-decode average. The greedy math comparisons alone do not have sign-test p<0.05; the bootstrap admission result does not establish a greedy-only improvement.


Admission audit: `s3://marin-rno2a/scratch/karan/task290/remote/independent-k8-admission-v1/`. Both clean source audits, raw hashes, driver hash and final done marker were verified. Code has exactly equal pooled wins/losses in BF16; a floating-point residue near zero is treated as zero using the exact paired counts.

<!-- far-arm-start -->
## Far donors under the anchor config: verified partial results

The planned far arm is still in progress. This completed Math group uses eager BF16 serving, max_tokens=8192 and context=12288, with the full panel and greedy+seeds17/29. Its raw matrices, frozen strict scorer, exclusive healthy host, fresh execution timing and exact report recomputation are independently verified. The two full Math controls are shown beside the completed Math merge. “Native” refers to RoPE theta=10000; remaining serving settings use the shared configuration. All merges use anchor theta=1000000. These results measure far donors under the anchor config, so Math merges reflect a configuration mismatch as well as weight changes.

| Candidate (max_tokens=8192) | GSM greedy /1319 | Strict MBPP greedy /257 | GSM gain pp [95% CI] vs Instruct | MBPP gain pp [95% CI] vs Instruct | Truncated GSM / MBPP | Mean tokens GSM / MBPP |
|---|---:|---:|---|---|---:|---:|
| instruct | 946 | 157 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 3 / 0 | 249.1 / 133.6 |
| math-native | 1116 | 30 | +12.00 [+10.13, +13.92] | -47.73 [-53.31, -42.41] | 4 / 13 | 348.4 / 787.1 |
| math-anchor-rope | 570 | 12 | -27.45 [-29.85, -24.97] | -56.16 [-61.35, -51.23] | 40 / 158 | 544.3 / 5178.7 |
| far-math-average-c0.1 | 743 | 139 | -15.09 [-17.06, -13.07] | -11.67 [-15.69, -7.65] | 91 / 3 | 779.7 / 178.4 |

Math alone loses 546 greedy GSM answers when only its RoPE setting changes to the anchor value (1116→570), and MBPP truncation rises from 13 to 158 items. The c=0.1 merge is below Instruct on both benchmarks under this configuration. Gains above use per-item averages of three decodes and pointwise bootstrap intervals; they are not greedy-only effects. The direct paired change from native to anchor RoPE is GSM −39.45 pp [−41.70, −37.23] and MBPP −8.43 pp [−11.54, −5.58], after averaging the three decodes within each item. The c=0.1 transfer is +12.36 pp [10.03, 14.68] on GSM relative to the full Math donor under anchor RoPE, while remaining below Instruct. These contrasts reuse existing outputs; source validation confirms the two full-donor controls differ only in name and RoPE setting. Direct-contrast proof: `remote/far-math-rope-audit-v1/` under the common artifact root.

Verified source: `s3://marin-rno2a/scratch/karan/task290/remote/far-scored-g1-audit-v1/`. Other far controls remain pending; the final Math table will repeat this same native/anchor control pair beside each Math merge.
<!-- far-arm-end -->
