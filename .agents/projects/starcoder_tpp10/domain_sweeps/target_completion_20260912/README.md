# Complete the FineMath target sweep

The user authorized completing the four paused FineMath targets and evaluating the finished checkpoints on the same math metrics. This release resumes only the 30%, 50%, 70%, and 100% target runs. All fourteen Wikipedia/FineMath matched proxies and the three lower-fraction FineMath targets are already complete.

The original target model, optimizer state, random state, data ordering, seed, token horizon, runtime pins, output paths, and four training fingerprints are retained. The native artifact runner uses the original materialized configurations; only its dispatch function changes to require restoration of a full trainer state. A separate resume receipt records the recovery specification and chosen checkpoint. `allow_partial_checkpoint` must remain false, and `load_checkpoint=True` prevents a missing checkpoint from silently restarting training.

| FineMath fraction | Preserved completed step | Final step | Original fingerprint |
|---|---:|---:|---|
| 30% | 7965 | 11490 | c1a0bf9e |
| 50% | 6426 | 11490 | ccaa1599 |
| 70% | 7219 | 11490 | 066c48ff |
| 100% | 6432 | 11490 | 9c2ae241 |

Checkpoint search includes the original permanent root, the original temporary root, and the verified preserved copy. Levanter chooses the latest complete checkpoint across them, so retries use new progress. The preserved copies remain available if the original temporary objects expire. The wrapper checks all preserved object generations, sizes, CRC32C and MD5 values through metadata requests; it does not copy or download their tensors in preflight.

The rebuilt training plan must equal the archived plan exactly. Preflight verifies final permanent checkpoints, runtime identities and required metrics for all fourteen proxies, checks the four output leases have expired, and requires existing data artifacts to be successful. The coordinator also verifies that the entire original Iris job tree is terminal. It releases four native steps with concurrency four, preserving successful outputs on resubmission.

The parent requests one on-demand CPU and 4 GiB in us-central1-a. Each target retains the original v5p-8, 8 CPU and 64 GiB child request, with interactive priority inherited from the parent. All storage stays in marin-us-central1.

`submit.sh` is the exact reviewed command. The recovery plan embeds the original training plan and checkpoint-preservation receipts; `preflight.json` records the checks.

Submitted as `/calvinxu/tpp10-finemath-target-completion` at 08:13 UTC on 12 September. All four children restored their full trainer states from the listed checkpoints and resumed at the following step. `observed_resume_receipts.json` and `training_progress.json` record the live evidence. By 08:24 UTC all four had advanced beyond restoration, to approximately steps 8,390, 6,840, 7,650 and 6,870. Completion requires successful Iris states and verified permanent step-11490 checkpoints.

After completion, `evaluate_tpp10_target_math_complete.py --build` freezes the four new final endpoints alongside the eleven target endpoints already evaluated. The subsequent evaluation reuses those eleven receipts by verified payload hash and scores only the four new checkpoints. The shared MATH-500/GSM8K scoring protocol and PALOMA control threshold remain unchanged.
