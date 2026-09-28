CC coordinator review dispositions, 9 September 2026

The subscription-safe read-only Opus review confirmed the authorization, source/data identity, failed-preparation, and regional placement gates. Its three operational findings were addressed before calibration submission:

1. Preparation wait reduced from 160,000 to 43,200 seconds; the coordinator timeout remains 172,800 seconds. This leaves at least 36 hours after the maximum preparation wait, before accounting for setup and scheduler time.
2. The submitted calibration parent explicitly sets failure and preemption retries to zero. Recovery is manual after inspecting descendants; completed matching training artifacts are reused. No claim is made that in-flight children are reattached.
3. The coordinator persists the complete endpoint rows and the batch-screen summary to the plan's immutable central1 `calibration_results.json` after all eight endpoints verify.

The frozen 27-file data/training identity, the scientific design, and the calibration plan hash are unchanged. The added coordinator passed the repository lint entry point and pyrefly. Its module imports and unchanged eight-run plan were checked locally. These checks do not establish live TPU training; admission and final endpoints remain live checks.
