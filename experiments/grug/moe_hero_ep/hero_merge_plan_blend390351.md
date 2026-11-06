# Blend of step-390251 and keep166 r3 step-390351: fixed plan
Written 2026-11-06 about 02:05Z, before any blend score exists. Windows and rule do not change after this.

Question: does keep166's MRCR recovery gain survive being diluted with pre-switch weights?

Candidates (linear blends, qk 1.66, router bias mixed linearly like every other parameter):
w = weight on 390351: 0, 0.25, 0.5, 0.75, 0.9, 1 (named w000 ... w100).
T = raw step-390251 (s3://marin-us-east-02a/marin/grug/hero-ctx65k-step372k/2026.08.19.2/checkpoints).
N = s3://marin-us-east-08a/tmp/ttl=1d/hero-semifinal-20261105-r3/keep166/checkpoints/step-390351.

Scoring: 262K held-out, MRCR two_shot cap_262144, 4K. Same 19-subset clean macro, Paloma-clean and Uncheatable as before.
Job: IRIS_USER=mwittmann, interactive, cw-us-east-08a, 16 nodes, CP4. Read-only, no weights written.

Primary metric: R(w) = eval_merge/w*/mrcr/two_shot/cap_262144/micro_needle_recovery_ratio.
This key is the one that reads 0.24339 at T and 0.98306 at N. Macro is reported beside it.
Gain kept: f(w) = (R(w) - R(0)) / (R(1) - R(0)). f = w is the straight line.

Verdict, on micro R, f from the same job:
- "fragile": f(0.75) < 0.5. A quarter of pre-switch weight loses over half the gain.
- "robust": f(0.5) >= 0.75. Half pre-switch weight keeps at least three quarters.
- "near linear": neither of those.
- "non-monotonic, no verdict": f falls by more than 0.1 between neighbouring w on the grid, or fragile and robust both hold.
Also report the smallest grid w with f >= 0.9, and whether macro would change the verdict.

Held-out: D_L(w) = score(w, L) - score(0, L), L in 262K, 4K, on clean, Paloma-clean, Uncheatable.
"Interior beats both ends" at L only if clean is at least 0.001 below the lower endpoint and Paloma-clean and Uncheatable are also below it.

Guards, from this job's endpoints, against the existing evals:
- w000 vs W&B hero-mrcr-keep166-step390251; w100 vs hero-mrcr-keep166-step390351.
- Clean and official-23 within 0.0001 at 262K and 4K; micro R within 0.001 (0.24339; 0.98306).
- On failure: stop, report, no resubmit, and no verdict.

The report must say: this blends one 100-step run at one-rack batch, so it says nothing final about extension checkpoints.
A blend of two points 100 steps apart is not a merge of many points. N's gain may be specific to that run.
It tests only the line between T and N, one seed, one recipe (keep166).

Order: before B's global arms. Rack: one, displacing nobody; A and B keep the two-rack cap unless the parent lifts it.

## Amendment, 2026-11-06 03:05Z, before any blend score has been read
Decided by the parent after the 1.66 arm missed the official-23 guard because of PTB alone. Disclosed as a deviation from the plan as first written.
Guards at 262K and 4K now read: clean and official-23 without PTB, each within 0.0001. Micro R within 0.001 is unchanged.
The report still gives the full official-23 and PTB's own number. Nothing else changes.
