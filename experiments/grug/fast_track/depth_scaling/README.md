# d512 depth-scaling analysis

Artifacts for the depth-scaling experiment issue (fast_track MoE, 16k BPE, H100).

- `depth_vs_width_scaling.png` — Paloma macro loss vs total training compute: the
  d512 depth ladder (6/12/18/24 layers, 1-4x data) overlaid on the width scaling
  law fit from the main ladder (d512->d1280).
- `grad_norm_by_depth.png` — per-layer gradient norm at the training midpoint
  (100-step average) for the 4 width baselines, split into attention / routed-expert
  / shared-expert components.
- `grad_norm_by_depth_normalized.png` — the same, each curve normalized to its
  per-run mean vs relative depth.
- `ladder_points.json` — the (compute, loss) points for both ladders.
