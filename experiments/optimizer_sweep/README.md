## Optimizer Sweep Experiments

This directory contains the code and configs to reproduce the optimizer study described in the paper “Fantastic Pretraining Optimizers And Where to Find them.” The goal is to fairly benchmark modern optimizers on Transformer LMs across model/data scales, estimate true speedups over AdamW, and fit simple scaling rules for hyperparameters. See the study description for context and results: [marin-community/marin#1290](https://github.com/marin-community/marin/issues/1290).

### What’s here

- Baseline configs and sweep grids for each optimizer/model size/Chinchilla regime
- A launcher to materialize training jobs from those JSON payloads
- Utilities to query best runs and persist a uniform results tree
- Analysis scripts to:
  - plot loss vs. data scale (Chinchilla ratio)
  - estimate effective data speedup vs. AdamW
  - identify stable vs. sensitive hyperparameters
  - fit simple power-law scaling rules and predict configs OOD (e.g., 1.2B and higher Chinchilla)

### Directory layout

- `baseline_config/` and `sweep_grids/`: per-optimizer JSONs that define the baseline configuration and the discrete sweep grids explored in the study.
- `Analysis/Results/`: canonical results written by the collector as JSON under `{optimizer}/{model_size}/{chinchilla}/result.json`.
- `Analysis/figs/`: figures produced by analysis scripts.
- `Analysis/predicted_baseline_config/`: predicted configs emitted by the scaling script under `{optimizer}/{model_size}/{chinchilla}/config.json`.

### Python files and their purpose

- `launch.py`
  - Discovers matching baseline and sweep JSONs and launches training runs via `marin.optimizer_sweep.template`.
  - New layout expected: `baseline_config/{optimizer}/{model_size}/{chinchilla}/*.json` and `sweep_grids/{optimizer}/{model_size}/{chinchilla}/*.json`.
  - CLI (dry-run prints the resolved payloads):
    ```bash
    python experiments/optimizer_sweep/launch.py adamw 1 130m --dry-run
    python experiments/optimizer_sweep/launch.py muon 8 520m --tpu-type v5litepod-128
    ```
  - Noted here for legacy reason, Muon and Scion uses keys 'muon_to_adamw_lr' and 'scion_to_signgd_lr', this means the ratio between 'adam_lr' used for LM heads, embeddings and LayerNorm and 'muon_lr' used for other linear layers.

- `defaults.py`
  - Shared experiment utilities and “best-practice” wrappers used by training pipelines:
    - dataset download (`default_download`), tokenization (`default_tokenize`), validation set plumbing
    - training entrypoints (`default_train`, `simulated_epoching_train`, `default_sft`, `default_anneal`)
    - scaling-law projection helper (`default_scaling_law_pred`)
  - Bridges to Levanter/Marin configs (e.g., `TrainLmConfig`, `WandbConfig`) so sweeps can reuse consistent training setup.

- `Analysis/end_to_end_results_from_configs.py`
  - End-to-end collector for best runs and ablations. For each `{optimizer, model_size, chinchilla}` triple:
    - Loads the baseline config and matching sweep grid JSONs
    - Uses `marin.optimizer_sweep.utils_simp` to find near-best runs on W&B and retrieve losses/run IDs
    - Writes a canonical payload to `Analysis/Results/{optimizer}/{model_size}/{chinchilla}/result.json` including:
      - `best_config`, `min_loss`, `approximate_best_config_list`
      - `baseline` and per-parameter `ablations` (losses and W&B run IDs when available)
  - Run to populate the results tree:
    ```bash
    python experiments/optimizer_sweep/Analysis/end_to_end_results_from_configs.py
    ```

- `Analysis/loss_plotting.py`
  - Aggregates `Analysis/Results/**/result.json` into a dataframe and plots loss vs. Chinchilla ratio per model size.
  - Produces PDFs at `Analysis/figs/optimizer_loss_scaling_{model_size}.pdf` using a consistent style from `plotting_config.py`.
  - Run:
    ```bash
    python experiments/optimizer_sweep/Analysis/loss_plotting.py
    ```

- `Analysis/speedup_estimation.py`
  - Fits an AdamW baseline scaling curve per model size, L(D) = α·D^(−B) + β, using losses from `Analysis/Results`.
  - For each optimizer point, computes the effective data budget D_adamw that would achieve its observed loss on the AdamW curve.
  - Plots D_adamw vs. D (with 1.0–1.4× speedup bands) per model size and a cross-size 8×-Chinchilla speedup summary.
  - Outputs PDFs under `Analysis/figs/`.
  - Run:
    ```bash
    python experiments/optimizer_sweep/Analysis/speedup_estimation.py
    ```

- `Analysis/find_stable_hyper.py`
  - Reads `Analysis/Results` payloads and identifies hyperparameters that are stable (single value works) across model sizes and Chinchilla ratios.
  - Writes `Analysis/non_stable_keys_by_optimizer.json` with keys that need retuning and minimal covering value-sets.
  - Run:
    ```bash
    python experiments/optimizer_sweep/Analysis/find_stable_hyper.py
    ```

- `Analysis/hyper_scaling.py`
  - Fits simple power-law parameterizations over model size and data scale for each hyperparameter using the best-configs in `Analysis/Results`.
  - Emits text reports `hyperparameters_fit_{optimizer}.md` and predicted baseline configs at `Analysis/predicted_baseline_config/{optimizer}/{model_size}/{chinchilla}/config.json` (e.g., 1.2B at 1/2/4/8× Chinchilla).
  - Run:
    ```bash
    python experiments/optimizer_sweep/Analysis/hyper_scaling.py
    ```

- `Analysis/plotting_config.py`
  - Centralized plotting styles: `color_map`, `correct_name`, `line_style` to keep figures consistent across scripts.

### Typical workflow

1. Ensure `baseline_config/` and `sweep_grids/` are populated for the target optimizer(s) and regimes.
2. Launch training jobs from JSONs:
   ```bash
   # preview jobs
   python experiments/optimizer_sweep/launch.py soape 4 300m --dry-run
   # launch
   python experiments/optimizer_sweep/launch.py soape 4 300m --tpu-type v5litepod-128
   ```
3. Collect best runs and persist a uniform results tree:
   ```bash
   python experiments/optimizer_sweep/Analysis/end_to_end_results_from_configs.py
   ```
4. Generate analysis artifacts:
   ```bash
   python experiments/optimizer_sweep/Analysis/loss_plotting.py
   python experiments/optimizer_sweep/Analysis/speedup_estimation.py
   python experiments/optimizer_sweep/Analysis/find_stable_hyper.py
   python experiments/optimizer_sweep/Analysis/hyper_scaling.py
   ```

### Notes

- Model sizes follow the study (e.g., 130M/300M/520M/1.2B) and “Chinchilla ratio” refers to tokens relative to a Chinchilla-optimal budget (20x non-embedding parameters)
- The overall findings and motivation, including why matrix preconditioners (e.g., Muon, Soap, Kron) often lead at higher budgets and the observed ≤1.4× speedups, are summarized in [marin-community/marin#1290](https://github.com/marin-community/marin/issues/1290).


