# Fixed plan: cross-switch analogue (A) and Stage 2 (B)

Written 2026-11-05 about 23:50Z, before any A or B result exists. Windows and rules below do not change after seeing data. Prior exposure: Larry's 6k-spaced raw 4K curves (quoted in the proposal) and Stage 1's 4K results. No 65K or 262K score of any merge has been seen, and no merge across the switch has been scored at any length.

## Facts checked
| Step | Tree | Trained at (W&B config) | LR |
|---|---|---|---|
| 360000, 366000 | hero-gradclip-step356k | 16K, qk 1.48, cf 1.15 | 4.09e-4, 3.61e-4 |
| 372000 | hero-gradclip-step356k | 16K, qk 1.48: its last update was at 16K; the 65K run forks from it | 3.12e-4 |
| 378000, 384000, 390000 | hero-ctx65k-step372k | 65K, qk 1.66, cf 1.5 | 2.64e-4, 2.15e-4, 1.67e-4 |

The dense tail (385500-390251) has LR 1.65e-4 to about 1.9e-4.

## Harness for both
Same as keep-1.66 and Stage 1: 16 nodes, context axis 4, zigzag layout, eval tokens 4194304, dropless. Scoring set is the fixed clean 19 (Paloma + Uncheatable minus ptb, wikitext_103, gab, twitterAAE). Paloma-clean and Uncheatable are reported separately. No weights written; object storage read-only.

## A. Cross-switch analogue (one rack)
Qk 1.66 all layers. Lengths 4K and 65K.

Reference T = raw step-390000: on the 6000 grid, so every window is a nested prefix and the job holds no extra accumulators. All raw points are scored at both lengths: 390000, 384000, 378000, 372000, 366000, 360000.

| Window | Steps | Phases |
|---|---|---|
| X2 | 390000, 384000 | 65K only |
| X3 | + 378000 | 65K only: largest 65K-only window at this spacing |
| X4 | + 372000 | crosses: adds the switch-point checkpoint, 16K-trained |
| X5 | + 366000 | crosses |
| X6 | + 360000 | crosses: 3 + 3 points |

Stops at 360000: before it, 356531 is the gradient-clip change.

U11 reference: U11's sources sit on another grid under T = 390251, and adding them here needs more host copies than the budget allows. So U11 comes from B's qk 1.66 arm (same harness, against raw 390251) and from Stage 1 at 4K. Both are reported beside A, and if B has not finished, A's report says so.

Quantities, each for clean 19, Paloma-clean and Uncheatable, at length L:
- D_L(W) = score(W, L) - score(T, L).
- Crossing effect: C_L(Xk) = D_L(Xk) - D_L(X3), k = 4, 5, 6.
- Length interaction: I(Xk) = C_65K(Xk) - C_4K(Xk).
- Lag split: G_L(W) = score(W, L) - mean of raw score(s, L) over s in W, the averaging gain; lag is mean raw(s, L) - score(T, L). D = G + lag.

Directional verdict, fixed now, primary on X6 with X4 reported alongside:
- "Crossing costs more at 65K": I(X6) >= +0.0005 on clean and the same sign on Paloma-clean and Uncheatable.
- "Crossing costs less at 65K": I(X6) <= -0.0005 on clean, same sign on both.
- Otherwise "no difference between lengths at this bar".
- Separately, "crossing helps at 65K" only if C_65K(Xk) <= -0.001 on clean, with Paloma-clean and Uncheatable both negative, and "crossing hurts at 65K" if C_65K(Xk) >= +0.001 likewise.

0.0005 is the proposal's bar for qk orthogonality, kept as the same scale. These are not statistical tests.

Screen: the proposal's dip test on each subset's raw series at each length. The scoring set stays at 19; any flagged subset gets a sensitivity row. A dip at 378000 at 65K is expected from the switch and would not on its own indicate contamination.

Guard: raw 390000 at 4K within 0.0001 clean of the same-harness value (smoke run). Raw 384000 and 378000 at 4K are also compared with Larry's run, at the qk it used, for information only. On failure, stop and report.

What this cannot show, and must be stated in the report: spacing is 6000 against the planned 500; LR is up to 2.5x the tail's and 1.5-2.5x at the 16K points; the wider window adds lag; the planned merge crosses from 65K to 262K rather than from 16K to 65K, and from a low and flat LR; the 16K points trained at qk 1.48 but are scored at 1.66. Directional evidence only. It does not decide a merge recipe or the launch.

## B. Stage 2 as written (U11 and B11a50)
T = raw step-390251, windows and recipes from Stage 1 unchanged.

| Arm | qk_mult | long scale | Baseline run for guard | Lengths | Scored |
|---|---|---|---|---|---|
| 1.66 | 1.66 | 1.0 | hero-mrcr-keep166-step390251 | 4K, 65K, 262K, MRCR | raw, U11, B11a50, B11a50+experts |
| global 1.75 | 1.66 | 1.054 | hero-mrcr-global175-step390251 | 65K, 262K, MRCR | raw, U11, B11a50 |
| global 1.88 | 1.66 | 1.133 | hero-mrcr-global188-step390251 | 65K, 262K, MRCR | raw, U11, B11a50 |

I read "1.75" as global 1.75: the proposal's steep-curve reason is the global 1.75 result, and no all-layer 1.75 baseline exists. 1.88 is provisional.

B11a50+experts is the proposal's "blend restricted to the expert banks": B11a50 on stacked_blocks.stacked.mlp.expert_mlp.{w_up, w_gate, w_down}, T everywhere else including the router bias. The 4K in the 1.66 arm exists to give it a 4K condition, and repeats U11 and B11a50 as a check.

Rule, from the proposal, unchanged. A candidate is better than T only if all hold:
1. 4K: clean Delta <= -0.001; >= 14/19 improve; Paloma-clean and Uncheatable both negative; no subset worse by more than 0.003; survives dropping the top contributor. U11 and B11a50 passed in Stage 1.
2. 65K and 262K at qk 1.66: same sign and at least half the 4K size, with the 4K size from Stage 1 (U11 -0.00299, so <= -0.00150; B11a50 -0.00260, so <= -0.00130). B11a50+experts uses its own 4K from this arm.
3. MRCR recovery at qk 1.66 does not fall by more than 0.02 against raw in the same arm. Recovery is the tracker's metric; if micro and macro disagree on the outcome, both are reported.

The rule is applied at qk 1.66. The other arms answer orthogonality:
- I = Delta_merge(arm) - Delta_merge(1.66), clean. Orthogonal if |I| <= 0.0005 at 65K and 262K, and recovery change agrees within 0.02 with the 1.66 arm's at both global 1.75 and global 1.88.
- Also reported: does each candidate meet condition 2 and 3 at global 1.88, since it may be the finalist?

The rule gives pass or fail per candidate; choosing between two passes is the parent's. The 262K base condition from the proposal (gain at least -0.005 and passes) is reported, not changed. Launch stays on raw step-390251.

Guards: each arm's raw within 0.0001 of its baseline run on clean and official-23 at each length; raw MRCR recovery within 0.001 of baseline. On failure, stop and report; no resubmit.

## Scheduling
One rack until the parent says the 11-rack test is done; then two. A first, then the 1.66 arm. Other arms follow as racks allow. IRIS_USER=mwittmann, interactive, --cluster=marin --target-cluster cw-us-east-08a. Never displace /marin or /larry jobs: check before each submit.
