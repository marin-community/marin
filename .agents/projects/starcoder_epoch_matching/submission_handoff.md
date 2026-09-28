Pilot submitted after CC review on 2026-09-08 at 17:29:55 PDT.

Iris parent: /calvinxu/starcoder-epoch-matching-pilot
Dashboard: https://iris.oa.dev/#/job/%2Fcalvinxu%2Fstarcoder-epoch-matching-pilot
Fieldbook parent: job_01m21s421dw5vb3p9c2d67q9ts
Frozen v2 design: 3191f3d005ebc3e1c653f1de664bfb0c3a81c291665a7a06b872903a0b4f9e0b

Ten new runs: nine 277.873M-token proxies on p={0,0.1,0.3,0.7,1}, with the identical p=0 arms deduplicated, and one corrected 7.408B-token target endpoint. Estimated training compute is 8.705e18 FLOPs. Later full-grid and replication releases remain separate decisions.

The intended new child environment is the bundled uv.lock (JAX/JAXlib 0.11.1, NumPy 2.3.5). Full parent and matched index hashes reproduce the reviewed historical mapping and are checked at submission. This establishes corpus identity, not numerical equivalence to the historical 0.10.1 training. All ten run names, paths and configuration fingerprints remain unchanged from the reviewed v1 protocol. The v1 baseline is archived under pre_cc_runtime_fix/.

100 targeted tests passed. Scoped lint and Pyrefly passed. Live cache/runtime/index audit passed under 0.11.1; exact command passed central1 placement validation. Iris accepted the parent, and its initial task was building dependencies. Child training has not yet been verified. The ledger recorded all ten datapoints and a submitting parent before the external call, then attached the acknowledged Iris ID.

All valid pilot outcomes, including ties and an unfavorable ordering, are retained. No one-grid-step promotion threshold is used on this irregular grid. Independent subset replication remains a separate follow-up; changing a trainer/data seed does not by itself change the fixed named-key support. Analysis now reports selected-weight displacement, matched-minus-unmatched target regret and optional excess-over-own-grid-minimum plots.

Specification and exact collection/analysis commands: .agents/projects/starcoder_epoch_matching/review_packet.md. Do not resubmit while this parent is queued or running. If it fails, inspect its failure and durable per-run artifacts before preparing a linked retry.

Startup verified at 17:32:18 PDT: all ten training children were acknowledged; one was running and had reached Qwen3/W&B initialization, while nine were queued for central1 resource capacity. No failures or preemptions were reported. The persisted GCS plan exactly matches the local audited plan. Child IDs and timestamps are in pilot_initial_child_snapshot.json. The nine queued logical-run mappings remain unassigned rather than inferred from asynchronous submission order.
