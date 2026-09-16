# Claude Code review: East5 bridge repair

Verdict: **GO** for isolated East5 reference training. Production remains gated.

Claude Code reviewed the exact command, dry-run artifacts, frozen acceptance contract, launcher, and launch-safety checker. It found that training all four preregistered East5 reference rows together is scientifically preferable to mixing one historical row with three fresh rows: the row set and thresholds remain frozen, while the code-bundle heterogeneity is removed.

The review verified that scientific inputs remain fixed across the East5 and Europe halves: data and trainer seeds, model and optimizer resolution, 80/20 schedule, mixture coordinates, global batch size, and evaluation definitions. The deployment changes are the accelerator and region-local paths. Table-9 runs on v6e-8 in both halves, avoiding an evaluation-hardware confound for that metric.

The review also verified namespace isolation from the legacy East5 production graph, the Europe bridge, and future production assignment state; strict region-local paths; the corrected working-directory includes; idempotent executor behavior; and retention of the phase-0 checkpoint at step 21,855 plus the endpoint at step 27,335.

Three requested pre-submission confirmations were completed after the review:

- The exact staged workspace has no files newer than the Europe v3 submission, so the scientific code bundle is unchanged.
- The isolated East5 bridge namespace is empty.
- Launch safety passes only with the explicit `us-east5-b` Table-9 zone override, as intended.

The East5 runtime and validation cache path hashes are pinned in `east5_bridge_preflight_v1.json`. Phase-0 Uncheatable evaluation remains a post-training detached sidecar because the ordinary 5,000-step evaluation cadence does not land on step 21,855.

Review provenance: session `ceff3620-7295-40b6-8fa1-8b4ea7d544d3`; all 42 assistant records identify `claude-opus-5`. The review ran through `claude -p` with `ANTHROPIC_API_KEY` removed, OAuth account `plambdafour@proton.me`, billing type `stripe_subscription`, and read-only tools.
