# CC review and release checks

CC reviewed the concrete ten-run package using subscription-authenticated `claude -p` with `env -u ANTHROPIC_API_KEY`, Opus 5 at maximum effort and read-only tools. The review is complete in `CC_REVIEW.md`: no correctness blockers; proceed after the final checks.

The requested correction is applied: `region_validation.txt` now records the same passing check as `region_validation.json`, explicitly allowing the historical child zone us-east5-b while keeping the CPU parent in us-east5-a. The superseded default-zone failure is retained under `diagnostics/`.

The final hash audit verifies 35 source/input files, both sets of reconstructed predictor checkpoints and the frozen mixture/prediction CSVs. All 1,220 original figure predictions match within 2.22e-16 BPB. The final bundle was rebuilt after review: 25,405,024 bytes, with the frozen candidate and all 454 imported repository files present. Its SHA256 is fb16e62d3bfe4986871cd387cf14a1c01bcf82d1b77be44abdb5f501166b5ec9.

For later analysis, score the actual rounded mixtures against the runtime predictions. Keep all ten paths and distinguish the small Uncheatable quadratic/spline prediction gaps from the more discriminating paths; one seed gives descriptive calibration errors. No extra runs are required by this review.

The active prediction receipt is `prediction_freeze/summary.json`, fingerprint 78b225d045b6d42ed175dd91d5b06cebecca7e7b04ad07102597fd447809337c. Earlier fingerprint directories and `midpoint_plan.pre_format.json` are superseded provenance, not additional experiments. The formatting change altered only the materializer's source hash; both mixture CSVs remained byte-identical. `figure10_path_reference.csv` now also retains the figure's parity reference beside the experiment artifacts.

Ten planned Fieldbook runs and the passing prediction/recipe validation are recorded. The submitting parent and actual Iris acknowledgment are recorded separately at launch.
