# Domain-Mix Paper Plots

This directory is the central home for presentation-quality plots from the
two-phase domain-mixture scaling work.

Conventions:

- Plot scripts live directly in this directory.
- Rendered artifacts live under `img/`.
- One-off debugging plots should stay near their experiment output directory,
  not here.
- Corrected scale labels must show both non-embedding parameter count `N` and
  realized training tokens `D`; do not present these as N-only scaling curves.
- Prefer Plotly for refreshed paper plots and use `RdYlGn_r` when a continuous
  color scale is needed.

Current plots:

- `baseline_scaling_trajectories.py`: 1x Chinchilla scaling trajectories for
  Proportional, Olmix, Uniform, UniMax, and GRP no-L2.
