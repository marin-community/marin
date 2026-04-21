# Probe Comparison — JSR/BJS Framing (48 tension points)

BCG is retained only as a deprecated diagnostic. The primary metrics here are joint satisfaction rate (JSR), balanced joint score (BJS), and oracle feasibility slices.

## Aggregate metrics

| model | n_points | JSR | BJS | weakest marginal | mean A | mean B |
|---|---:|---:|---:|---:|---:|---:|
| gpt-5.1 (oracle) | 48 | 0.432 | 0.536 | 4.771 | 6.505 | 5.833 |
| M0 SFT (marin-8b-instruct) | 50 | 0.140 | 0.405 | 3.385 | 5.380 | 4.495 |
| M1 DPO LoRA lr=1e-5 seed=0 | 50 | 0.295 | 0.527 | 4.570 | 5.880 | 5.830 |

## Oracle feasibility decomposition

| slice | n_points | share |
|---|---:|---:|
| Feasible | 18 | 37.5% |
| Marginal | 11 | 22.9% |
| Infeasible | 19 | 39.6% |

## Feasible-slice metrics

| model | n_points | JSR | BJS | weakest marginal |
|---|---:|---:|---:|---:|
| gpt-5.1 (oracle) | 18 | 0.903 | 0.873 | 8.319 |
| M0 SFT (marin-8b-instruct) | 18 | 0.361 | 0.559 | 4.958 |
| M1 DPO LoRA lr=1e-5 seed=0 | 18 | 0.500 | 0.661 | 6.069 |

## DPO effect on shared probe points

| metric | slice | n_shared | improved | regressed | ties | win/loss | mean delta |
|---|---|---:|---:|---:|---:|---:|---:|
| JSR | All | 50 | 14 | 4 | 32 | 3.50x | 0.155 |
| JSR | Feasible | 18 | 7 | 4 | 7 | 1.75x | 0.139 |
| BJS | All | 50 | 33 | 14 | 3 | 2.36x | 0.122 |
| BJS | Feasible | 18 | 11 | 6 | 1 | 1.83x | 0.103 |
