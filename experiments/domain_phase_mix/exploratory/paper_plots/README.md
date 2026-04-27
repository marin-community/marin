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

- `starcoder_two_phase_nonmonotonicity.py`: background figure for the
  low-dimensional two-phase StarCoder diagnostic. It renders separate
  `starcoder_two_phase_landscape.{png,pdf}` and
  `starcoder_two_phase_slice.{png,pdf}` artifacts so LaTeX controls the final
  side-by-side layout. The plots show an interpolated phase-0/phase-1
  StarCoder loss landscape and the Nemotron-first U-shaped slice where more
  StarCoder eventually worsens code BPB through repetition. It reads the
  143-row W&B export vendored under `data/` so the legacy global-minimum point
  is available.
- `baseline_scaling_trajectories.py`: 1x Chinchilla scaling trajectories for
  Proportional, Olmix, Uniform, UniMax, and GRP no-L2. The HTML output is
  interactive and can switch among available perplexity eval metrics; the PNG
  and PDF outputs render the default `eval/uncheatable_eval/bpb` view without
  interactive controls for paper use.
- `downstream_eval_scaling_trajectories.py`: GSM8K, HumanEval, and MMLU
  scaling trajectories for the same baseline-scaling cells. This script merges
  collected downstream-eval result CSVs with checkpoint-attached lm-eval
  artifacts so historical `skip_existing` MMLU rows are plotted. The HTML
  outputs are interactive and include per-eval metric dropdowns, while the PNG
  and PDF outputs render each eval's default headline metric without
  interactive controls for paper use.
