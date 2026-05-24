# Delphi Midtraining Visualization Logbook

This logbook is the compact handoff for the Delphi midtraining visualization
thread. Use it instead of searching the very large
`.agents/logbooks/midtraining_delphi.md` unless you need older training-run
details.

## 2026-05-23T01:15Z — current state and next-agent instructions

### User Goal

Ahmed wants an interactive report for Delphi midtraining validation prediction:

- choose mix and learning rate filters;
- choose a prefix percentage of a run and a functional form;
- see predicted final `math_val_loss` from partial trajectories;
- compare per-cell fits against joint fits over flop scale, mix, and LR;
- find methods/prefixes that hit a target max absolute final-loss error.

The report is published at:

<https://ahmeda14960.github.io/delphi-midtraining/?v=ee726ed8b>

The local generated HTML is:

`midtrain_analysis_outputs/small_final_loss_scaling/delphi_midtraining_interactive.html`

The staged GitHub Pages copy is:

`/tmp/ahmed-pages-delphi/delphi-midtraining/index.html`

### What Changed In The Latest Pass

Warmup was 10% for all runs, so prefixes below 10% are invalid. The analysis
and UI now treat `10%` as the minimum prefix.

Updated files:

- `scripts/analysis/delphi_within_run_prediction.py`
  - `PREFIX_FRACS` is now `10%-90%` in `5%` steps.
  - This affects both per-cell prediction and the joint script, because the
    joint script imports this constant.
- `scripts/analysis/build_delphi_midtraining_interactive_report.py`
  - per-cell prefix slider has `min="10"`;
  - joint prefix slider has `min="10"`;
  - report text says final model comparison uses endpoint MAE.
- `.agents/projects/delphi_midtraining.md`
  - notes now say the current train/tune split uses the clean small ladder
    through `3e20`, with `1e21` and `1e22` held out;
  - notes now say the prefix grid starts at 10% because of warmup.
- `.agents/logbooks/midtraining_delphi.md`
  - appended a short entry recording the 10% warmup correction.
- Personal GitHub Pages repo:
  - `ahmeda14960/ahmeda14960.github.io`
  - commit `ee726ed84515e65f64a7905cd68fd6cd24831ba8`
  - message `Drop warmup-only Delphi prefixes`

Generated outputs refreshed:

- `trajectory_prefix_predictions.csv`
- `trajectory_prefix_summary.csv`
- `trajectory_method_selection.csv`
- `trajectory_prediction_summary.md`
- `trajectory_joint_prefix_predictions.csv`
- `trajectory_joint_prefix_summary.csv`
- `trajectory_joint_prefix_models.csv`
- `delphi_midtraining_interactive.html`

Verification already done:

- per-cell prediction CSV minimum prefix is `0.10`;
- joint prediction CSV minimum prefix is `0.10`;
- joint model CSV minimum prefix is `0.10`;
- local report sliders have `min="10"`;
- hosted GitHub Pages HTML has `min="10"` for both sliders.

### Current Modeling Setup

The report has two main sections.

#### Curve Prediction Within A LR / Mix / Flop Cell

This section fits/predicts separately for each completed run/cell, i.e. each
fixed `(flop scale, mix, learning rate)` trajectory.

Data split:

- train/tune: clean small-ladder runs through `3e20`;
- held out for large-scale generalization checks: `1e21` and `1e22`.

Important: `3e20` is not held out anymore. Treat it as part of the small ladder.

Prediction methods:

- `last_value`
- `linear_tau`
- `template_global`
- `template_by_mix`
- `template_by_recipe`
- `curve_log_mae`, `curve_log_huber`
- `curve_exp_mae`, `curve_exp_huber`
- `curve_power_mae`, `curve_power_huber`
- `curve_rational_mae`, `curve_rational_huber`

Parametric per-cell curves are fitted with SciPy:

- MAE variants use `scipy.optimize.minimize`;
- Huber variants use `scipy.optimize.least_squares(loss="huber")`;
- evaluation/comparison is final endpoint MAE, regardless of fit objective.

Current selected small-scale recipes from the regenerated summary:

| metric | method | prefix | small MAE | held-out MAE |
|---|---|---:|---:|---:|
| `eval_loss` | `template_by_recipe` | `0.70` | `0.00245` | `0.01519` |
| `math_val_loss` | `template_by_recipe` | `0.65` | `0.00235` | `0.01857` |
| `paloma_c4_loss` | `template_by_recipe` | `0.10` | `0.00557` | `0.02048` |
| `paloma_macro_loss` | `template_by_recipe` | `0.10` | `0.00600` | `0.02139` |

#### Joint Trajectory Fits Across LR, Mix, And Flop

This section is inspired by:

- paper: <https://arxiv.org/abs/2507.21184>
- project page: <https://linhaowei1.github.io/scaling_law_discovery/>

It fits shared trajectory regressions over:

- normalized progress `tau`;
- flop scale;
- mix;
- LR.

Scopes:

- `global`: one model across all flops, mixes, and LRs;
- `by_flop`: one model per flop scale, sharing only across mix and LR.

Joint forms currently implemented:

- `joint_*_exp_drift`
- `joint_*_power_drift`
- `joint_*_gompertz_shoulder`

Current headline: joint fits are useful to visualize but are worse than the
per-cell SciPy fits for the strict all-run target. Best joint all-run config is
`joint_by_flop_power_drift @ 90%` with MAE about `0.009892` and max error about
`0.028735`; it does not satisfy max absolute error `<= 0.02` across all
completed math runs.

### Commands To Rebuild

If new W&B data arrived, run the per-cell script without `--use-cache`; that
updates the cached trajectory points by fetching completed run specs missing
from `trajectory_points.csv`.

```bash
uv run python scripts/analysis/delphi_within_run_prediction.py
uv run python scripts/analysis/delphi_joint_trajectory_prediction.py
uv run python scripts/analysis/build_delphi_midtraining_interactive_report.py \
  --output midtrain_analysis_outputs/small_final_loss_scaling/delphi_midtraining_interactive.html
uv run python scripts/analysis/build_delphi_midtraining_interactive_report.py \
  --output /tmp/ahmed-pages-delphi/delphi-midtraining/index.html
```

If no new W&B data arrived and only code/report logic changed, this is faster:

```bash
uv run python scripts/analysis/delphi_within_run_prediction.py --use-cache
uv run python scripts/analysis/delphi_joint_trajectory_prediction.py
uv run python scripts/analysis/build_delphi_midtraining_interactive_report.py \
  --output midtrain_analysis_outputs/small_final_loss_scaling/delphi_midtraining_interactive.html
uv run python scripts/analysis/build_delphi_midtraining_interactive_report.py \
  --output /tmp/ahmed-pages-delphi/delphi-midtraining/index.html
```

Validation snippet:

```bash
uv run python - <<'PY'
import json
import re
from pathlib import Path

import pandas as pd

out = Path("midtrain_analysis_outputs/small_final_loss_scaling")
for name in [
    "trajectory_prefix_predictions.csv",
    "trajectory_joint_prefix_predictions.csv",
    "trajectory_joint_prefix_models.csv",
]:
    df = pd.read_csv(out / name, low_memory=False)
    print(name, df["prefix"].min())

html = (out / "delphi_midtraining_interactive.html").read_text()
payload = json.loads(re.search(
    r'<script id="payload" type="application/json">(.*?)</script>',
    html,
    re.S,
).group(1))
print("prefix slider min=10", 'id="prefix" type="range" min="10"' in html)
print("joint slider min=10", 'id="jointPrefix" type="range" min="10"' in html)
print(
    "payload min prefixes",
    min(row["prefix"] for row in payload["predictions"]),
    min(row["prefix"] for row in payload["jointPredictions"]),
)
PY
```

Pre-commit command:

```bash
./infra/pre-commit.py --fix \
  scripts/analysis/delphi_within_run_prediction.py \
  scripts/analysis/delphi_joint_trajectory_prediction.py \
  scripts/analysis/build_delphi_midtraining_interactive_report.py \
  .agents/projects/delphi_midtraining.md \
  .agents/logbooks/delphi_midtraining_visualization.md
```

Avoid running pre-commit over `.agents/logbooks/midtraining_delphi.md` unless
you specifically need to; it currently trips the repo large-file guard because
that logbook is already about 560 KB.

### Publishing To GitHub Pages

Local `git` is currently blocked by the Xcode license prompt on this machine.
Use the GitHub API via `gh` to update the personal Pages repo.

```bash
set -euo pipefail
sha=$(gh api \
  'repos/ahmeda14960/ahmeda14960.github.io/contents/delphi-midtraining/index.html?ref=master' \
  --jq .sha)
base64 -i /tmp/ahmed-pages-delphi/delphi-midtraining/index.html | tr -d '\n' \
  > /tmp/delphi-midtraining-index.b64
jq -n \
  --rawfile content /tmp/delphi-midtraining-index.b64 \
  --arg message 'Update Delphi midtraining visualization' \
  --arg branch master \
  --arg sha "$sha" \
  '{message: $message, branch: $branch, sha: $sha, content: $content}' \
  > /tmp/delphi-midtraining-index-update.json
gh api -X PUT \
  repos/ahmeda14960/ahmeda14960.github.io/contents/delphi-midtraining/index.html \
  --input /tmp/delphi-midtraining-index-update.json \
  --jq '{commit: .commit.sha, content: .content.path}'
```

Then open with a cache-busting query string:

```bash
open -a 'Google Chrome' 'https://ahmeda14960.github.io/delphi-midtraining/?v=<commit-prefix>'
```

Check the served HTML if GitHub Pages seems stale:

```bash
curl -L -sS -D /tmp/pages-headers.txt \
  -o /tmp/pages-delphi-index.html \
  'https://ahmeda14960.github.io/delphi-midtraining/?v=<commit-prefix>'
rg -n 'id="prefix"|id="jointPrefix"|min="5"|min="10"' /tmp/pages-delphi-index.html
```

### Gotchas

- Do not reintroduce `5%` prefixes. The first 10% was warmup for these runs.
- `3e20` is part of the small ladder/training side now, not a held-out target.
- `1e21` and `1e22` remain held out in this visualization framing.
- The GitHub Pages report is a standalone Plotly/JavaScript HTML page, not a
  live marimo server.
- Keep the report payload compact. Do not embed full joint curve grids in the
  HTML; embed joint model coefficients and compute curves in JavaScript. The
  current standalone HTML is about 17 MB.
- For new 3e20/large-run completions, first rerun
  `delphi_within_run_prediction.py` without `--use-cache`, then rerun the
  joint script and report builder.
- Final metric in target tables is max absolute final-loss error, with MAE
  also shown as the average error column. The user cares that a config works
  for every completed run, not just on average.

### Current Uncommitted State To Be Aware Of

At the time this logbook was created, these relevant files were modified or
untracked in the Marin worktree:

- `.agents/logbooks/midtraining_delphi.md`
- `.agents/projects/delphi_midtraining.md`
- `scripts/analysis/__marimo__/session/delphi_small_final_loss_scaling_notebook.py.json`
- `scripts/analysis/delphi_small_final_loss_scaling.py`
- `scripts/analysis/delphi_small_final_loss_scaling_notebook.py`
- `scripts/analysis/delphi_within_run_prediction.py`
- `scripts/analysis/build_delphi_midtraining_interactive_report.py`
- `scripts/analysis/compare_scipy_within_run_fits.py`
- `scripts/analysis/delphi_joint_trajectory_prediction.py`

Do not revert unrelated changes. The generated analysis outputs under
`midtrain_analysis_outputs/` may be ignored by git but are the source for the
standalone HTML.
