The purpose of this experiment ([marin#4251](https://github.com/marin-community/marin/issues/4251)) is to run a parameter scaling sweep with all other settings optimized over a single epoch.
This will use Complete(d) [[1](https://arxiv.org/abs/2512.22382)] heuristics for transfer, as adapted for Adam with Hyperball [[2](https://whenwen.github.io/wd_blog/public/hyperball-part-1.html), [3](https://psychedelic-sunstone-851.notion.site/Fantastic-Pretraining-Optimizers-and-Where-to-Find-Them-2-1-Hyperball-Optimization-2e924306e6f280e7a5ffee00eb40a0dd), [4](https://github.com/marin-community/levanter/pull/1253)] normalization in [marin#3292](https://github.com/marin-community/marin/pull/3292). Extensions for epoching will follow, but let's see how this goes first.


---

### Data

Animals / single bp tokenization: union of `region` ∈ {`upstream`, `downstream`, `CDS`}. Context: 255 bp (256−1 for BOS).

- Training: [CDS](https://huggingface.co/datasets/bolinas-dna/genomes-v5-genome_set-animals-intervals-v5_255_128) (242,334,716) | [Upstream](https://huggingface.co/datasets/bolinas-dna/genomes-v5-genome_set-animals-intervals-v1_255_128) (68,286,166) | [Downstream](https://huggingface.co/datasets/bolinas-dna/genomes-v5-genome_set-animals-intervals-v15_255_128) (20,501,856) = 331,122,738 total (~84.8B tokens) ([counts](https://gist.github.com/eric-czech/656b63dc78ac7792f5c5d824e0b5f103))
- Validation (16,384 each): [CDS](https://huggingface.co/datasets/bolinas-dna/genomes-v5-validation-intervals-v5_255_255) | [Upstream](https://huggingface.co/datasets/bolinas-dna/genomes-v5-validation-intervals-v1_255_255) | [Downstream](https://huggingface.co/datasets/bolinas-dna/genomes-v5-validation-intervals-v15_255_255)
- IMPORTANT: Lowercase = repeats in training, but = non-functional (non-conserved per phyloP) in validation. These are NOT the same.
- Mixture weights (proportional to examples, equivalent to concatenation): CDS=0.7319, upstream=0.2062, downstream=0.0619
- Tokenizer: [tokenizer-char-bos](https://huggingface.co/bolinas-dna/tokenizer-char-bos), vocab_size=7 (PAD, UNK, BOS, a, c, g, t). Usage in [`exp94_human_enhancers.py`](https://github.com/marin-community/marin/blob/human-enhancers/experiments/dna/exp94_human_enhancers.py).

<details><summary>Tokenized dataset size</summary>

```
tokenized dataset sizes in gs://marin-dna-us-central1/tokenized/:

  ┌────────────────────────────────────┬───────────┐
  │              Dataset               │   Size    │
  ├────────────────────────────────────┼───────────┤
  │ Training                           │           │
  ├────────────────────────────────────┼───────────┤
  │ bolinas-v5-cds-char-bos            │ 92.0 GB   │
  ├────────────────────────────────────┼───────────┤
  │ bolinas-v5-upstream-char-bos       │ 28.2 GB   │
  ├────────────────────────────────────┼───────────┤
  │ bolinas-v5-downstream-char-bos     │ 8.7 GB    │
  ├────────────────────────────────────┼───────────┤
  │ Training total                     │ ~128.9 GB │
  ├────────────────────────────────────┼───────────┤
  │ Validation                         │           │
  ├────────────────────────────────────┼───────────┤
  │ bolinas-v5-val_cds-char-bos        │ 14.9 MB   │
  ├────────────────────────────────────┼───────────┤
  │ bolinas-v5-val_upstream-char-bos   │ 12.2 MB   │
  ├────────────────────────────────────┼───────────┤
  │ bolinas-v5-val_downstream-char-bos │ 11.8 MB   │
  ├────────────────────────────────────┼───────────┤
  │ Validation total                   │ ~38.9 MB  │
  └────────────────────────────────────┴───────────┘
```

</details>

### Metrics

- Online: unweighted CE loss nats / BPB (cf. [marin#2310](https://github.com/marin-community/marin/pull/2310)), stratified by `region` (inferred from dataset source or added as explicit field)
- Online: VEP ([marin#3144](https://github.com/marin-community/marin/pull/3144), [marin#3333](https://github.com/marin-community/marin/pull/3333))
- Online: `LL(functional)`, `LL(non-functional)`, `LL(functional) - LL(non-functional)` ([bolinas#8](https://github.com/Open-Athena/bolinas-dna/issues/8), [marin#3483](https://github.com/marin-community/marin/pull/3483))
- Offline (final checkpoint at largest scale): VEP by variant type
- Offline: VEP vs `LL(functional) - LL(non-functional)` and validation loss

Wandb key prefixes (confirmed from smoke test `dna-bolinas-smoke-v0.1-ae4ba7`):
- `eval/loss`, `eval/bpb` — per-token average across all validation datasets (mixture weights not used in eval; equal-sized datasets → ~equal contribution per region)
- `eval/macro_{loss,bpb}` — mean of per-dataset averages (each dataset weighted equally)
- `eval/val_{cds,upstream,downstream}/{loss,bpb}` — per-region validation
- `lm_eval/traitgym_mendelian_v2_255/auprc` — overall VEP AUPRC
- `lm_eval/traitgym_mendelian_v2_255/{variant_type}/auprc` — per-variant (missense, splicing, etc.)
- `train/loss` — training loss

### Step 1: Reference sweep

Run per [marin#2432](https://github.com/marin-community/marin/pull/2432), adapted for DNA. Follow [`reference_hyperparameter_sweep.py`](https://github.com/marin-community/marin/blob/a243fe5628532f8213860b1668c0259275a37774/experiments/references/reference_hyperparameter_sweep.py) for sweep structure.

- Sizing: Text reference sweep → 130M params, C = 6ND = 2.028×10¹⁸ FLOPs (<span>@</span>20:1). Same hidden_size=512 with DNA vocab_size=7 yields **~25M params** (via `Qwen3Config.total_trainable_params`). At ~100:1 token-to-param ratio for DNA ([IsoFLOP analysis](https://github.com/marin-community/marin/issues/2343#issuecomment-3974111513)) → **N=25M, D=2.5B**, C = 6ND ≈ 3.75×10¹⁷ FLOPs (~5.4× smaller than text reference)
- Sweep `initializer_range` ∈ {.04, .02, .01, .005, .0025} via `Qwen3Config.initializer_range` (inherited from [`LlamaConfig.initializer_range`](https://github.com/marin-community/levanter/blob/main/src/levanter/models/llama.py#L64), default 0.02)
  - Set per study via `dataclasses.replace(base_model_config, initializer_range=value)`
  - Guard against overfitting given greater sequence similarity vs text (as N→∞ in single epoch)
- Architecture: `Qwen3Config` (not Grug) via [`CompletedAdamHHeuristic._build_model_config`](https://github.com/marin-community/marin/blob/a243fe5628532f8213860b1668c0259275a37774/experiments/scaling_law_sweeps/completed_adamh.py#L227) with `seq_len=256`, `vocab_size=7`
- Training: `run_levanter_train_lm` called directly inside `remote(run_vizier_train)` (not `default_train` ExecutorSteps — hparams not known at DAG construction time). Build `TrainLmOnPodConfig` from Vizier suggestion.
- Optimizer: `AdamHConfig` built from Vizier suggestion, same as reference sweep's `_build_adamh_config`
- Group: `dna-bolinas-reference-sweep-{VERSION}`
- Run name: `dna-bolinas-reference-{VERSION}-IR{initializer_range}-E{epochs}-L{loop}-T{trial}`
- Tags: `sweep`, `dna`, `bolinas`, `reference`, `version`, `epochs`, `initializer_range`, `lr`, `beta1`, `adam_lr`, `beta2`, `epsilon`, `max_grad_norm`, `z_loss_weight`, `batch_size`, `params`, `tokens`, `loop`, `trial`

Executor DAG structure (5 IR studies × 10 loops × 4 suggestions/loop):
```python
# DAG construction (at __main__ time, not runtime)
for epochs in EPOCHS:                               # 1 for now
  for init_range in INITIALIZER_RANGES:             # 5 independent Vizier studies
    model = replace(base_model, initializer_range=init_range)
    for loop in range(num_loops):                   # sequential (DB dependency)
        suggest   ← previous_update / vizier.db
        train × N ← suggest / suggestions.json     # parallel
        update    ← [train_0..N] + suggest / vizier.db
    optimal ← final_update / vizier.db
executor_main(steps=all_optimal_steps)
```

Poll progress (replace `VERSION`):
```python
runs = wandb.Api().runs('eric-czech/marin',
    filters={'group': 'dna-bolinas-reference-sweep-VERSION'})
for r in runs:  # r.state, r.created_at, r.summary['eval/macro_loss'],
    ...         # r.summary['_step'], r.config['trainer']['num_train_steps']
```

### Step 2: Transfer validation

At single-epoch scale, sweep key hypers (LR, beta1, beta2) in isolation to test loss basin alignment. Use largest model size from the parameter scaling sweep (derive from same code).

Best reference params (fetch from wandb to seed transfer/scaling sweeps):

```python
source .env && uv run python3 -c "
import wandb; api = wandb.Api()
runs = api.runs('eric-czech/marin', filters={'group': 'dna-bolinas-reference-sweep-v0.6'})
KEYS = ['initializer_range','lr','adam_lr','beta1','beta2','eps','mgn','zloss']
data = []
for r in runs:
    if r.state != 'finished' or not isinstance(r.summary.get('eval/loss'), float): continue
    tags = {k: float(v) for t in r.tags if '=' in t for k, v in [t.split('=', 1)] if k in KEYS}
    data.append({'loss': r.summary['eval/loss'], 'url': r.url, **tags})
best = min(data, key=lambda d: d['loss'])
print(f'{len(data)} finished runs. Best (eval/loss={best[\"loss\"]:.6f}):')
print(best['url'])
print({k: v for k, v in best.items() if k not in ('loss', 'url')})
"
# 69 finished runs. Best (eval/loss=1.240387):
# https://wandb.ai/eric-czech/marin/runs/dna-bolinas-ref-v0.6-IR0.01-E1-loop5-trial1-371753
# {'adam_lr': 0.01391, 'beta1': 0.6001, 'beta2': 0.7163, 'eps': 8.816e-08,
#  'initializer_range': 0.01, 'lr': 0.003345, 'mgn': 0.5227, 'zloss': 0.000169}
```

### Step 3: Parameter scaling sweep

Follow [`_build_model_configs` in `completed_adamh.py`](https://github.com/marin-community/marin/blob/a243fe5628532f8213860b1668c0259275a37774/experiments/scaling_law_sweeps/completed_adamh.py#L274) for model sizing and configs.

### Code

Marin (`~/repos/crfm/marin`, branch `eac/dna-bolinas-scaling-sweep` off `dna`). Pending [marin#4247](https://github.com/marin-community/marin/pull/4247); use `eac/dna-rebase` until merged.
- Module: `experiments/dna/exp109_bolinas_scaling_sweep.py`
- Subcommands via `if __name__ == "__main__"` switch: `run_{smoke_test,reference_tuning,transfer_validation,parameter_scaling}_sweep`
- Config generation must be shared between reference sweep and param sweep

### Logging

- `VERSION = "v1.0"` — module-level constant, manually bumped on restart
- Step tag: `reference` | `transfer` | `scaling`
- Run names: `dna-bolinas-{step}-{VERSION}-...` with step-specific suffixes. Output path = `checkpoints/{run_name}`.
- Wandb group per step+version. Run name is a strict subset of tags.
- `epochs` is hardcoded to 1 for now but must appear in run names, tags, and all analyses as a first-class dimension
- Checkpointing: Use `_final_checkpoint_only` for all sweeps by default, unless otherwise specified.
- IMPORTANT: Analysis code in Bolinas collects from wandb only, not Marin source code.

Per-step progress with timestamps (for ETA estimation):
```python
import wandb; api = wandb.Api()
run = api.run('eric-czech/marin/<RUN_ID>')
hist = list(run.scan_history(keys=['_step', '_timestamp', 'run_progress']))
# _timestamp: unix epoch when the step was logged; run_progress: fraction in [0, 1]
```

### Execution

Subcommand dispatch uses `SWEEP_COMMAND` env var.  Ask the subcommand to use is not clear from context.

### TODOs

- [ ] Compare scaling sweep results to [past results](https://gist.github.com/eric-czech/787e7ab1a0e0be87bfecc7bce1fa8e83)
- [X] Fix vizier sweep bugs
  - [X] Need to set crash_on_nan=False, crash_on_inf=False
  - [X] Need to skip completed trials on resumption: `vizier._src.service.grpc_util.LocalRpcError: Trial owners/marin/studies/dna-bolinas-ref-v0.6-IR0.0025-E1/trials/1 has state SUCCEEDED. Only trials in state ACTIVE or STOPPING can be completed.`
  - See https://github.com/marin-community/marin/pull/4563
- [ ] Fix lm-eval bugs
  - [ ] Fix `PreTrainedTokenizerFast has no attribute encode_batch`
    - See https://github.com/marin-community/marin/pull/4677 and See https://github.com/marin-community/marin/issues/4678
  - [X] Fix `alueError: No resource mapping found`
    - See https://github.com/marin-community/marin/pull/4911
  - [X] Fix `TypeError: fused_cross_entropy_loss_and_logsumexp_penalty() got an unexpected keyword argument 'implementation'`
    - See https://github.com/marin-community/marin/issues/4852
  - [ ] Fix `ModuleNotFoundError: No module named 'lm_eval'`
    - See https://github.com/marin-community/marin/commit/71939d3f47cbbbb1c4c5d59d444ca7a6c18650dc


### Misc

Old `.gitignore` lines:

```
# lib/iris/src/iris/rpc/*_pb2.py
# lib/iris/src/iris/rpc/*_pb2.pyi
# lib/iris/src/iris/rpc/*_connect.py
```

