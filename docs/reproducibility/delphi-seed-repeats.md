# Reproduce the additional Delphi seeds

[`experiments/references/train_delphi_seeds.py`](https://github.com/marin-community/marin/blob/main/experiments/references/train_delphi_seeds.py)
records the recipes for additional random seeds at seven Delphi compute budgets
and three seeds each at two additional budgets. It preserves the original
training settings, dataset component names and order, artifact identities, and
Weights & Biases IDs.

The runner defines 22 runs. The Hugging Face release contains 20 checkpoints;
the seed-0 runs at `3e18` and `9e18` are controls and are excluded. Existing
Delphi releases at `1e21`, `1e22`, and `1e23` are unchanged.

## Checkpoint names

All model repositories are under [`marin-community`](https://huggingface.co/marin-community).
For seeds 42 and 62746, append `-seed42` or `-seed62746` to the base name.
Seed 0 has no suffix. These are training seeds.

| Compute label | Published seeds | HF repository base name |
| --- | --- | --- |
| `3e18` | 42, 62746 | `delphi-3e18-447Mparams-1.2Btokens` |
| `9e18` | 42, 62746 | `delphi-9e18-550Mparams-2.9Btokens` |
| `2e19` | 42, 62746 | `delphi-2e19-837Mparams-3.6Btokens` |
| `3e19` | 42, 62746 | `delphi-3e19-998Mparams-5Btokens` |
| `5e19` | 0, 42, 62746 | `delphi-5e19-1.2Bparams-7.4Btokens` |
| `9e19` | 42, 62746 | `delphi-9e19-1.4Bparams-10.6Btokens` |
| `2e20` | 42, 62746 | `delphi-2e20-1.9Bparams-14.8Btokens` |
| `3e20` | 42, 62746 | `delphi-3e20-2.5Bparams-18.6Btokens` |
| `6e20` | 0, 42, 62746 | `delphi-6e20-2.9Bparams-29.9Btokens` |

The `5e19` and `6e20` labels refer to configured budgets of 5.3e19 and 5.5e20
FLOPs, respectively. Parameter and token counts in repository names are rounded;
the runner records the architecture, batch size, step count, and optimizer for
each budget.

## Inputs and setup

These commands require access to Marin's Iris cluster, the existing tokenized
GCS caches listed in the runner's `DATASETS`, and its
`meta-llama/Meta-Llama-3.1-8B` tokenizer. The runner adopts the existing caches;
it does not rebuild or redistribute them. Set up the checkout and credentials
using the [internal onboarding guide](../dev-guide/guidelines-internal.md).
Run the commands below from the repository root with `WANDB_API_KEY` exported.

The recorded recipes were recovered from W&B configurations and execution logs
associated with Marin commit `9e946e47cfa74efe43cca1b74360c27f64dd1226`.
That identifies the source recipes, not necessarily the revision used to train
each new checkpoint. Use the published model card and original run records for
checkpoint-specific provenance.

## Launch the runs

Define the submission helper once per shell:

```bash
submit_delphi() {
  local delphi_label="$1"
  shift
  uv run iris --cluster=marin job run --no-wait --no-cancel-on-exit \
    --priority batch --region us-central2 --cpu 1 --memory 2G --extra cpu \
    --job-name "delphi-seeds-${delphi_label}-$(date -u +%Y%m%d-%H%M%S)-$$" \
    -- python -m experiments.references.train_delphi_seeds "$@"
}
```

Submit the 16 original-recipe runs, including the two seed-0 controls:

```bash
(
  set -e
  submit_delphi single-host --slugs 3e18 9e18 2e19 3e19
  for delphi_slug in 9e19 2e20 3e20; do
    for delphi_seed in 42 62746; do
      submit_delphi "${delphi_slug}-seed${delphi_seed}" \
        --slugs "$delphi_slug" --seeds "$delphi_seed"
    done
  done
)
```

Submit the six runs at the additional compute budgets separately:

```bash
(
  set -e
  submit_delphi 5e19 --slugs 5e19
  for delphi_seed in 0 42 62746; do
    submit_delphi "6e20-seed${delphi_seed}" \
      --slugs 6e20 --seeds "$delphi_seed"
  done
)
```

Each multi-host run has its own top-level Iris job to avoid JAX coordinator
name collisions. The single-host runs share a top-level job within each group.
Use `--slugs` and `--seeds` to select runs; `--max-concurrent` controls executor
concurrency within one top-level job and defaults to 16.

Before retrying a run, stop its previous coordinator. Successful artifacts are
reused, and interrupted runs resume from their own checkpoints. The unique
submission job names do not change training identities. `VERSION = "2026.09.22"`
fixes the recorded artifact identities; use a new version for a new recipe or
an independent rerun that must not reuse those artifacts.

## Outputs and HF export

The artifact name is `delphi-seed-repeats-<slug>-seed<seed>`, and the trainer
and W&B run ID is that name followed by `-2026.09.22`. Under the configured
Marin storage prefix, the final HF export is:

```text
<prefix>/delphi-seed-repeats-<slug>-seed<seed>/2026.09.22/hf/step-<num_train_steps - 1>/
```

The model is Qwen3. Its historical HF reference checkpoint can leave
`architectures: ["LlamaForCausalLM"]` in the exported `config.json`.
Before publishing a fresh export, change only that field to
`"architectures": ["Qwen3ForCausalLM"]`. This is a metadata correction;
it does not convert or change the weights. The released checkpoints already
contain the corrected field.

Publish the config, tokenizer files, safetensors files, and shard index when
present. Executor files such as `.artifact.json`, `.executor_info`, and
`.executor_status` are not inference inputs. The models use Apache-2.0, matching
the existing [Delphi release](https://huggingface.co/collections/marin-community/delphi).
