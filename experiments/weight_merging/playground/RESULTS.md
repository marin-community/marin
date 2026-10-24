# Verified playground results

Status: experiments are unfinished. These tables cover completed, verified curves; the new independent-sample donor gates and far-donor controls remain pending. No two-donor method ranking is established.

All curves below use eager vLLM execution, output cap **8192**, context **12288**, the full 1319-item GSM8K test and 257-item sanitized MBPP test, and greedy plus sampled seeds 17/29. Each gain averages three decodes per item before a paired item bootstrap (5000 resamples, seed 290). Intervals are pointwise exploratory 95% CIs, without correction for recipe selection. Each precision has its own Instruct control. Separate BF16 and FP32 tables change both weight casting and compute; their difference does not isolate either cause.

MBPP scores use the frozen strict extractor and Docker sandbox. Scoring used one scorer per exclusive healthy host. Both generation and correctness of shared controls matched exactly across the scoring shards. Earlier shared-host grades are excluded from these tables. Truncation and lengths below are greedy counts/means; unclosed thinking counts all three decodes.

The original math RFT donor used duplicated responses because seeded `n>1` requests in vLLM 0.6.6.post1 copied the same seed to each child. Its evaluations use `n=1` and remain valid. New training uses separately seeded `n=1` requests, a diversity probe, and exact response deduplication; its fresh graph-mode gates cannot reuse these eager controls.

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
| aristaeus-dare_linear-c0.25 | 940 | 158 | -0.18 [-1.34, +0.99] | +0.26 [-1.69, +2.20] | 8 / 0 | 274.8 / 119.1 | 0 |
| aristaeus-dare_linear-c0.5 | 951 | 157 | +0.88 [-0.40, +2.15] | +0.00 [-2.46, +2.46] | 10 / 0 | 287.9 / 96.5 | 0 |
| aristaeus-dare_linear-c0.75 | 951 | 156 | +0.61 [-0.81, +2.02] | -1.56 [-4.15, +1.04] | 11 / 1 | 300.2 / 112.4 | 0 |
| aristaeus-dare_linear-energy0.5-c0.5 | 940 | 154 | +0.28 [-0.83, +1.36] | +0.39 [-1.30, +2.08] | 9 / 1 | 281.1 / 168.8 | 0 |
| aristaeus-dare_linear-energy0.7-c0.5 | 940 | 157 | -0.03 [-1.09, +1.04] | +0.00 [-1.82, +1.82] | 7 / 1 | 270.6 / 151.3 | 0 |
| aristaeus-dare_linear-energy0.9-c0.5 | 934 | 159 | -0.23 [-1.31, +0.91] | +0.00 [-2.08, +1.95] | 5 / 0 | 261.7 / 123.9 | 0 |
| aristaeus-magnitude_topk-c0.25 | 951 | 156 | +1.21 [-0.10, +2.50] | +0.13 [-2.33, +2.59] | 12 / 1 | 306.1 / 133.4 | 0 |
| aristaeus-magnitude_topk-c0.5 | 972 | 154 | +1.19 [-0.23, +2.58] | -1.82 [-4.93, +1.17] | 7 / 2 | 279.8 / 144.9 | 0 |
| aristaeus-magnitude_topk-c0.75 | 964 | 151 | +1.72 [+0.18, +3.29] | -4.15 [-7.78, -0.65] | 21 / 2 | 377.9 / 250.8 | 0 |
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
| aristaeus-dare_linear-c0.5 | 951 | 159 | +1.14 [-0.08, +2.40] | +0.13 [-2.46, +2.72] | 11 / 0 | 290.6 / 102.5 | 0 |
| aristaeus-dare_linear-energy0.5-c0.5 | 929 | 154 | -0.03 [-0.99, +0.94] | +0.00 [-1.43, +1.43] | 9 / 0 | 279.7 / 130.9 | 0 |
| aristaeus-dare_linear-energy0.7-c0.5 | 940 | 156 | +0.35 [-0.66, +1.36] | -0.13 [-1.82, +1.56] | 10 / 0 | 290.5 / 125.8 | 0 |
| aristaeus-dare_linear-energy0.9-c0.5 | 942 | 157 | +0.38 [-0.68, +1.49] | -0.39 [-2.33, +1.43] | 8 / 0 | 280.9 / 127.2 | 0 |
| aristaeus-magnitude_topk-c0.5 | 973 | 155 | +2.07 [+0.63, +3.51] | -2.33 [-5.45, +0.65] | 6 / 0 | 271.1 / 100.6 | 0 |
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

## Controlled precision comparisons

| Comparison, fixed FP32 serving | GSM gain pp [95% CI] | Strict MBPP gain pp [95% CI] |
|---|---:|---:|
| Original math c=0.5, rounded donor minus FP32 donor | +0.28 [−0.81, +1.36] | 0.00 [−1.43, +1.43] |
| Aristaeus c=0.5, BF16-rounded final weights minus FP32 final weights | −0.13 [−0.33, +0.05] | Pending direct item-bootstrap contrast |

These comparisons round different objects: the first rounds the donor before delta arithmetic; the second rounds the completed merge. Neither establishes a GSM effect. BF16 final casting erased most of the small math update, which is why the math curve uses FP32 throughout.

The Aristaeus top-k densities 0.04731512589614803, 0.022387211385683378 and 0.01188502227437019 retain about 90%, 70% and 50% of its squared update L2. At the same numeric densities, math retains about 26.84%, 15.80% and 9.88%. Math density 0.3 retains about 78.01%. Sparse updates are rescaled to the actual L2 size of their own donor c=0.5 update, so a shared density does not imply shared retained energy.
