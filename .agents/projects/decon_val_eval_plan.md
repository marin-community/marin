# Eval-only plan: Delphi midtrain checkpoints × paranoid decon val sets

Status: PLAN (2026-06-07). Prereq: `/ahmed/decon-val-build-all-v2` caches verified.

## Goal

For each midtrained Delphi checkpoint, load model weights and compute val loss
on the three paranoid short-doc caches as **separate tagged datasets** —
`gs://marin-us-east5/tokenized/nemotron_math_val_decon/{j050,j075,j090}` —
plus the **original 12,500-window math val carve-out as an in-harness anchor**,
so the contamination effect per scale is (anchor loss − decon loss) measured
with one harness. Decides whether the 1e22 "too good" math val loss survives
decontamination.

## Verified ground truth (don't re-derive)

- Entry point: `lib/levanter/src/levanter/main/eval_lm.py` — `EvalLmConfig`
  (`checkpoint_path` XOR `hf_checkpoint`, `data`, `model`, `trainer`,
  `max_eval_length`), evaluates `data.tagged_eval_sets(Pos)` → per-dataset
  `eval/{tag}/loss` via `TaggedEvaluator`, plus micro/macro averages.
- In-repo wrapper pattern: `lib/marin/src/marin/evaluation/log_probs.py`
  (`evaluate_lm_log_probs`) — builds `LevanterEvalLmConfig`, WandB +
  `JsonFileTrackerConfig` trackers, submits a fray TPU `JobRequest`
  (`Entrypoint.from_callable(do_eval_lm, ...)`). **Traps**: it leaves
  `max_eval_length` at the levanter default **2048** (we need **4096**), and
  hardcodes wandb project "marin". So: copy the pattern, don't call it.
- Checkpoints: midtrain run dirs `gs://marin-us-east5/checkpoints/delphi-*`
  have `checkpoints/` cleaned (only `eval_metrics.jsonl`); weights live in
  **`hf/step-N/`** exports (e.g. 1e21 K=0.20: final `hf/step-4410/`). So load
  with `hf_checkpoint` / `checkpoint_is_hf=True`.
- Architecture is **Qwen3** (`Qwen3ForCausalLM`, e.g. 1e21: hidden 2560, 26
  layers, 20 heads, head_dim 128, vocab 128256, rope llama3, max_pos 4096).
  Build the levanter model config from each run's `hf/step-N/config.json` via
  the HF converter (`config_from_hf_config`) — never hand-pick dims.
- Canonical run selection: use the registry
  (`/Users/ahmed/code/marin/.claude/worktrees/midtrain_data/experiments/delphi_models.py`
  + `experiments/midtrain_specs/delphi_small_cpt_k020.py` there), NOT GCS
  globbing — the bucket holds deprecated/ablation runs alongside canonical
  (standing rule). Known anchors: `delphi-1e21-p33m67-9p25b-lr0.5-efbc63`
  (+ p50m50 `-973c46`, p67m33 `-114e49`).

## Scope (default; expand later)

- Mix `p33m67` (highest math), K=0.20 CPT, lr0.5, all scales 3e18 → 1e22,
  **final** hf step per run. Start with 1e22 + 1e21 (the two ends of the
  question), then backfill smaller scales.

## Implementation — `scripts/analysis/eval_decon_val_sets.py`

One driver script, one TPU job per checkpoint. CLI:
`--run gs://…/delphi-1e21-p33m67-9p25b-lr0.5-efbc63 [--step N (default: max under hf/)] [--tpu v6e-8] [--skip-anchor]`

1. **Resolve checkpoint**: glob `{run}/hf/step-*/`, pick max step (or `--step`),
   read its `config.json`, build levanter model config via the Qwen3 HF
   converter (verify levanter has Qwen3 — `lib/levanter/src/levanter/models/`;
   it must, since the runs exported it).
2. **Data config** (`LmDataConfig`, tokenizer `meta-llama/Meta-Llama-3.1-8B`):
   - 3 components `decon_j050/j075/j090`: `cache_dir=<cache root>`,
     `split="validation"`, no source, `train_weights=0.0` each
     (validation-only; `tagged_eval_sets` evaluates each separately, tag =
     component name).
   - 1 anchor component `math_4plus_origval`: the original math component
     copied verbatim from
     `experiments/midtrain_specs/data_sections/p33m67.json` — same cache
     (`4plus-2c5519`), `num_validation_sequences: 12500`,
     `shuffle_before_trainval_split: true`, same feistel shuffle params →
     reproduces the byte-identical original val split in-harness.
   - `block_cross_document_attention` + any other data flags copied from the
     data-section JSON, not defaulted.
3. **Eval config**: `max_eval_length=4096` (explicit — the 2048 default is the
   known trap), `per_device_eval_parallelism` 4–8, trackers = WandB (project
   tags `["decon_val_eval", run_name]`) + `JsonFileTrackerConfig(output_path=
   gs://marin-us-east5/scratch/ahmed/midtrain_dedup/decon_val_sets/evals/{run_name}/)`.
4. **Launch**: iris TPU job in us-east5 (data + checkpoints in-region),
   **preemptible always** (standing rule): `v6e-8` for 1e22 (9.7B), `v6e-4`
   for ≤1e21 (≤3.4B). Eval volume is tiny (≤107M tokens incl. anchor → minutes
   per checkpoint), so don't upsize beyond this. Submit either via
   `iris job run --tpu …` (check CLI support) or via the fray
   `JobRequest`/`Entrypoint.from_callable` pattern from `log_probs.py`.
5. **No-overwrite**: eval outputs land under
   `…/decon_val_sets/evals/{run_name}/step-{N}/` — refuse to run if
   `metrics.jsonl` already exists there unless `--force`; never write near the
   caches. Distinct iris job names `decon-eval-{scale}-{mix}-{step}`.
6. **Babysit** each job (30s×5min then 60s×10min); preemption-safe because
   eval is stateless and re-runnable.

## Verification

- Sanity gate (run once, before the sweep): 1e21 anchor loss from this harness
  must match the run's recorded final math-val loss in
  `tracker_metrics.jsonl` / W&B to ~1e-3 — proves split + seq-len + packing
  parity. If it doesn't match, stop and debug before trusting any decon number.
- Each job's JSON tracker file has `eval/decon_j050/loss`, `…j075…`, `…j090…`,
  `eval/math_4plus_origval/loss` plus sequence counts matching the manifests
  (~2510 / 5073 / 6187 / 12500).
- Collect into one table: per scale × {anchor, j090, j075, j050} loss; the
  contamination signal is the gap widening with scale (and with stricter
  cutoff).

## Risks / open items

- Levanter Qwen3 converter coverage (rope llama3 scaling block) — verify by
  loading the smallest (3e18) checkpoint first.
- All-zero `train_weights` edge: eval path only calls
  `tagged_eval_sets`/`validation_sets`, which tolerates it; confirm no
  validation in `LmDataConfig.__post_init__` rejects all-zero weights (if it
  does, give one component weight 1.0 — training never runs).
- The exact canonical run list per scale needs the registry in the
  `midtrain_data` worktree; do not guess from bucket listings.
- Original-val anchor doubles eval tokens; `--skip-anchor` exists for re-runs.
