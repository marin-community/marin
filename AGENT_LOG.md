# AGENT_LOG — ep25-d1 (custom scatter-add adjoint for gather-dispatch backward)

Worktree: `/home/marin/projects/marin/.worktrees/ep25-d1-adjoint`, branch `agent/ep25-d1-adjoint`, base `rav/ep-2` @ fe21ea495.

## Design (established from comments 5073017396, 5074952738, EVIDENCE-slack)

The fixed-a2a MoE has two structured gathers whose XLA-autodiff backward is a
generic scatter-add — the two costs the XProf trace flags (gather-dispatch backward
+ combine/all-gather backward, the latter called out by David Hall as
"pathologically bad ... worth custom vjp work"):

- **Dispatch** (gather mode): `send_x = padded_x[token_sources]`. Fan-out topk, so its
  transpose is a genuine segment-sum: `d_x[t] = sum_k cotangent[linear_indices[t*topk+k]]`
  over kept assignments. Replaces scatter-add with gather+reduce.
- **Combine**: `gathered = send_output[linear_indices]`. Injective on kept assignments,
  so its transpose is a pure gather along the slot->assignment inverse
  (`assignment_sources`), not a scatter-add: `d_send_output[j] = cotangent[assignment_sources[j]]`.

Both reuse the forward's int32 index composition. Exact transposes (no approximation).

Flags: `SCALE_A2A_GATHER_DISPATCH=1` selects gather dispatch (the 20.558% control);
`SCALE_A2A_CUSTOM_ADJOINT=1` (requires gather dispatch) wraps both gathers in custom_vjp.

## Check-in (start)
Findings so far:
- Reconstructed gather-dispatch patch matches comment 5073017396 snippet exactly; commit 1.
- custom_vjp math derived and verified on paper for both dispatch (segment-sum) and combine
  (injective gather); both are exact transposes reusing linear_indices / assignment_sources.
- Combine adjoint added to scope per David Hall's Slack note (prime target).
Confidence: 6/10 that this direction is a significant step toward 25% MFU (backward is 1 of 2
top costs; combine-backward being pathological is the upside; A/B not yet measured).
Next: CPU numerics test (gradient parity rtol=atol=1e-5, identical drop counts), then 1-replica smoke, then rack A/B.

## Check-in 23:28 UTC
Findings so far (numbers first):
- IMPLEMENTATION DONE + VALIDATED. Two commits: gather-dispatch reconstruction
  (SCALE_A2A_GATHER_DISPATCH) + structured custom_vjp for BOTH gathers
  (SCALE_A2A_CUSTOM_ADJOINT). Dispatch backward = segment-sum over topk; combine
  backward = injective gather along assignment_sources. Exact transposes.
- NUMERICS: new CPU tests pass. Gradient parity vs autodiff at rtol=atol=1e-5 for
  x/combine_weights/w13/w2; identical nonzero drop counts. (test_custom_adjoint_matches_autodiff_gradients,
  test_gather_dispatch_matches_scatter_forward_and_drops)
- HLO: backward drops from scatter=544 -> scatter=0 (gathers only) with custom adjoint
  at d5120/8-of-256. The pathological scatter-add transpose is fully eliminated.
- CLUSTER OVERLAP (critical): rav is LIVE on this exact direction right now.
  /rav/ep64-dispatch-grad-only-30-v1 (running) shows **p50 MFU 25.43%** (p10 25.20 / p90 25.63,
  415.8K tok/s, ~10.09s/step) at the EP64 operating point — past the 25% goal vs 20.558% baseline.
  His custom-combine XProf (profile-ep64-custom-combine-nocg): backward scatter gone, comm now
  29.5%, top op ncclDevKernel_SendRecv 22.4% — consistent with the adjoint removing the backward cost.
  Caveat: "grad-only" may be a partial-step bench; treat 25.43% as directional not a locked matched A/B.
  rav's 120-step ep64-custom-combine-stability was KILLED (no completed matched MFU A/B yet).
Confidence: 8/10 that this direction is a significant step toward 25% MFU (rav's live 25.4% + my
  scatter->0 HLO + 1e-5 parity all agree; the only gap is a clean completed 120-step matched A/B).
Next: coordinate before burning a shared rack (etiquette: avoid duplicating rav's in-flight work,
  cluster heavily contended). Messaging coordinator with numbers + recommendation.
