# TPP10 calibration optimizer review — 9 September 2026

The eight calibration runs support retaining proxy batch 32 and releasing the three frozen p=1 canaries. The user authorized this review and canary submission with “Do those.” Pilot and dense-grid stages remain unreleased.

The archived endpoint check passed for all eight runs. Mean unmatched-proxy BPB is 1.126364 at batch 32 and 1.121061 at batch 128: the 0.005304 gap passes the prespecified 0.010 threshold. Matched-arm outcomes remain diagnostic and did not select the recipe.

W&B history scans cover 12,644 training-loss/LR records and 1,264 paired gradient/parameter-norm records across the eight runs. All scanned values are finite. Training/LR records cover steps 2–2531 for batch 32 and 2–632 for batch 128; norms are logged approximately every ten updates. The GCS `tracker_metrics.jsonl` files contain configuration and final summaries, so the longitudinal review uses the separately archived W&B histories.

Training loss falls from about 10.3 to 2.66–2.81, with no sustained divergence. Batch 32 has visibly greater minibatch noise. Its largest late gradient spike, in unmatched seed 20260911 at step 2480, rises to 0.908 and returns to 0.164 at the next logged norm (step 2490). The final logged gradient norms range from 0.101 to 0.189. These are observed total norms; clipping is configured per optimizer group.

Both learning rates follow the frozen warmup, plateau and final-20% cosine decay, with peaks 0.020 (Muon) and 0.008 (Adam). The last logged Muon rates are 1.92e-7 at batch 32 and 3.06e-6 at batch 128. Parameter norms settle near the end of training. The rendered diagnostic was visually checked for loss/gradient instability and legibility.

This is a feasibility review at p=0.5 on the proxy architecture. The target architecture and the three p=1 endpoints are the next canary checks; this review establishes neither target stability nor a simulated-epoching benefit.

Artifacts:

- [Trace plot](calibration_optimizer_traces.png)
- [Numerical trace summary and history hashes](calibration_trace_summary.json)
- [Collector and plot script](review_calibration_traces.py)
- [Verified calibration endpoints](calibration_metrics.csv)

The rebuilt canary plan exactly matches the frozen plan `4362a8b1b2e37d7ae73096f7658c5b3a79774288bf268f240fa4e49edd826b60`; all 27 source and asset pins match. No optimizer, data, mixture-grid or runtime changes were made.
