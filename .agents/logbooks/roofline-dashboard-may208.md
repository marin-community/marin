# May208 Roofline Dashboard

Experiment issue: https://github.com/marin-community/marin/issues/6573
Branch: `codex/experiment-roofline-dashboard-may208`
Parent issue: https://github.com/marin-community/marin/issues/4302
Implementation issue: https://github.com/marin-community/marin/issues/6557

## Scope

Build an auditable roofline dashboard for the May208 H100 x8 MoE profile. The
dashboard should answer, per device, how much time each semantic operation
should spend, how much time it actually spent in the profile, and which profile
regions are still observed-only or unattributed.

## 2026-06-23

- Added the roofline dashboard package and CLI entry point.
- Imported the May208 W&B run and profile into `scratch/rooflines/may208.json`.
- Added hardware selection, node count, sortable columns, hover math, and profile
  step normalization.
- Normalized reported profile timing to per-device work by dividing
  track-summed XProf kernel time by profile devices and train-step count.
- Added attribution rows for MoE expert exchange, expert activation
  reduce-scatter, expert backward psums, MuonH Newton-Schulz gram/polynomial/apply
  phases, Hyperball, optimizer vector ops, uncategorized kernels, and unaccounted
  empty train-step time.
- Marked `expert_all_to_all` observed time as a proxy row rather than comparable
  to the modeled all-to-all estimate, because its current profile attribution
  includes nearby gather/scatter kernels rather than an isolated exposed exchange.
- Captured `.agents/artifacts/roofline-dashboard-may208/actual-device-desc.png`
  with the dashboard sorted by actual/device time descending.

## Current Readout

The current May208 dashboard top rows by actual/device time are:

- `expert_backward_psum`: 371.073 ms/device observed-only.
- `grouped_muon_restore`: 276.597 ms/device.
- `muon_ns_apply`: 115.610 ms/device.
- `muon_ns_polynomial`: 111.075 ms/device.
- `moe_expert`: 105.864 ms/device.
- `muon_ns_gram`: 91.611 ms/device.
- `unaccounted_for`: 64.929 ms/device of empty train-step time.
- `expert_all_to_all`: 57.935 ms/device observed proxy; modeled at
  697.932 ms/device under the current inter-host fabric assumption.

## Open Questions

- Confirm whether `expert_backward_psum` is expert-weight gradient accumulation
  and whether it should stay separate from forward MoE exchange.
- Keep reducing `uncategorized` by assigning concrete framework paths to semantic
  rows when the XProf names are unambiguous.
- Revisit collective efficiency only when the observed profile row is comparable
  to the modeled transfer, not just nearby kernels in the MoE block.
