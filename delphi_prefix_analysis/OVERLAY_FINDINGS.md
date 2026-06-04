# Delphi prefix vs canonical — POST-FIX runs only

One post-resume-fix gen-2 run per FLOP bucket, overlaid on its canonical compute-optimal base.
Gen-1 pre-fix single-step runs excluded (deleted for 3e18/3e19; 3e20 has no gen-2 run yet).

| base | prefix run | 70% step | 80% step | window | Δloss@start | Δloss@end | settled mean\|Δ\| | val mean/max \|Δ\| |
|---|---|---|---|---|---:|---:|---:|---:|
| 3e18 | `delphi-3e18-prefixes-qwen3` | 26134 | 29868 | 20001-29868 | -0.0001 | +0.0059 | 0.00329 | 0.0016/0.0047 |
| 9e18 | `delphi-9e18-prefixes-qwen3` | 31021 | 35453 | 30001-35453 | +0.0002 | +0.0056 | 0.00279 | 0.00264/0.0043 |
| 2e19 | `delphi-2e19-prefixes-qwen3` | 38587 | 44100 | 30001-44100 | -0.0000 | +0.0002 | 0.00394 | 0.00306/0.0088 |
| 3e19 | `delphi-3e19-prefixes-qwen3` | 26609 | 30411 | 20001-30411 | +0.0001 | +0.0042 | 0.00427 | 0.00488/0.0126 |
| 9e19 | `delphi-9e19-prefixes-qwen3` | 28198 | 32226 | 20001-32226 | +0.0002 | +0.0023 | 0.00412 | 0.0035/0.0096 |
| 2e20 | `delphi-2e20-prefixes-qwen3-from40k` | — | 45113 | 40001-45113 | +0.0002 | +0.0059 | 0.00349 | 0.00334/0.0063 |

## Verdict
- All 6 post-fix runs start exactly on the canonical curve (Δ@start ≈ 0), settled train Δ ≈ 0.003 (minibatch RNG only), val Δ ≤ ~0.013 — bit-faithful continuations of the compute-optimal base.
- Saved checkpoints (dashed verticals): 70% via `--also-save-step`, 80% as target; 2e20 = 80% only from step-40000.
- **3e20 excluded** — its only prefix is the pre-fix gen-1 `delphi-3e20-step24857` (resume transient); needs a gen-2 re-run to produce a clean 70% and the missing 80%.

