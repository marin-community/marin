# Fine-tune Snowball with Levanter on H100s

The [Snowball SFT presets](https://github.com/marin-community/marin/blob/main/experiments/post_training/snowball_sft.py) train Levanter's `SnowballLMHeadModel` from pinned Hugging Face weights. The `openthoughts-32k-4node` preset transforms OpenThoughts Agent conversations, supervises assistant tokens, and saves a native Levanter checkpoint and a Hugging Face export.

The preset is a 32,768-token, ten-step recipe on 32 H100s. A four-node smoke overrode the step count to five and saved a native checkpoint and Hugging Face export at step 4.

The mesh splits each sequence across four nodes and shards experts across eight H100s within a node. Attention keeps local queries and gathers the full keys and values. Large weight and optimizer leaves are sharded over both axes. The OpenThoughts preset puts one conversation in each sequence and keeps its first 32,768 tokens. Its scan layers are checkpointed during reverse mode to bound activation memory.

## Prepare

Install the workspace and CUDA-enabled JAX packages as described in [Setting up a Local GPU Environment](local-gpu.md):

```bash
uv sync --all-packages --extra=gpu
```

Provide a shared `MARIN_PREFIX` in the same region as the H100 workers, a Hugging Face token that can read the [Snowball base model](https://huggingface.co/open-athena/snowball-67b-a2b-base-262k-qk175-skew8) and [OpenThoughts Agent dataset](https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-SFT-100K), and the required storage credentials. The recipe pins the model weights and dataset revisions. It loads the tokenizer from the model repository name.

```bash
export MARIN_PREFIX=s3://your-bucket/snowball-sft
export HF_TOKEN=hf_example
```

## Launch

Print the plan by running `python -m experiments.post_training.snowball_sft --preset openthoughts-32k-4node`. To train, submit a CPU coordinator through Iris with `--run`.

```bash
uv run iris --cluster=marin job run --job-name snowball-sft-coord \
  --region us-east5 --cpu 1 --memory 2G --extra cpu \
  --priority interactive --no-wait \
  -e MARIN_PREFIX "$MARIN_PREFIX" -e HF_TOKEN "$HF_TOKEN" \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -- python -m experiments.post_training.snowball_sft --preset openthoughts-32k-4node --run
```

The coordinator and workers need the same storage and Hugging Face credentials. The completed training artifact is recorded under `checkpoints/snowball-openthoughts-agent-sft` in `MARIN_PREFIX`. Change the `SFTSpec` fields in the recipe to choose a different step count, batch size, or context length; increasing context length requires a separate memory and attention validation.

## Choose an SFT preset

Preset names use `<data>-<seq-len>-<nodes>`.

### Snowball

| Preset | What it reproduces | Training layout | Measured run | Model returned |
| --- | --- | --- | --- | --- |
| `openthoughts-32k-4node` | OpenThoughts Agent SFT with one conversation per sequence. | 4 nodes × 8 H100s | The five-update smoke used 17.0 H100-hours. | Final Levanter HF export. |
| `mixture-262k-32node` | Snowball SFT with the published [supervised mixture](https://github.com/marin-community/marin/pull/11120) and its packed chat format. | 32 nodes × 8 H100s | The H100 smoke used a five-update override. | Final Levanter HF export with FP32 weights. |

### Mochi

| Preset | What it reproduces | Training layout | Measured run | Model returned |
| --- | --- | --- | --- | --- |
| `mixture-262k-16node` | Three updates using the production recipe. | 1 rack, 16 nodes × 4 GB200s | Three default updates used 13.27 GB200-hours and saved native step 393494. A CPU job exported the checkpoint. | With `--export`, a schema-3 BF16 HF model. |
| `mixture-262k-48node` | Three updates using the production recipe. | 3 racks, 48 nodes × 4 GB200s | CPU plan only. | With `--export`, a schema-3 BF16 HF model. |
| `mixture-262k-176node` | Full production continuation `hero-sft-fullab-step393k-v2` from its pinned native checkpoint. | 11 racks, 176 nodes × 4 GB200s | 58.0 h of recorded training runtime and 1 h 9 min 53 s of CPU export worker time. | With `--export`, a schema-3 BF16 HF export of step `400407`. |

Run the matching module with `--preset` to print its plan. Add `--run` in the coordinator to execute
the plan. The preset supplies the training allocation.

```bash
uv run python -m experiments.post_training.snowball_sft --preset mixture-262k-32node
uv run python -m experiments.post_training.mochi_sft --preset mixture-262k-16node --steps 3 --export
```

Use `SFTPreset.model()` to request an export in Python workflows. Snowball's `--run` writes its HF model
and tokenizer at `<training-output>/hf/step-N`. Mochi's `--run` needs `--export` to write an HF model.
Run Mochi jobs in a coordinator placed on `cw-us-east-08a`.
The short Mochi presets train without checkpoints or exports by default. A retry starts from the pinned
checkpoint. `--steps` sets the number of updates from that point. `--export` saves the final native
checkpoint and runs a CPU export that writes the weights, pinned tokenizer, and generation config at
`<export-output>/hf`. The short presets request a 448 GiB CPU worker for export, sized to read a full
expert bank from the native checkpoint. Short runs use the cluster's `tmp/ttl=3d/` prefix. Set
`--prefix` to choose another storage root. Set `GIT_COMMIT` to the source commit when exporting from an
Iris bundle without Git metadata.

The 176-node preset uses the original production training artifact and checkpoint schedule. Pass
`--export` to request its associated export. Use `--steps` to run a bounded variant in scratch storage.

## Checkpoints and recovery

The shared `SFTSpec` defaults to one rolling temporary native checkpoint containing the complete training state, saved at the first training step after each 30-minute interval. A replacement is committed before the previous temporary checkpoint is deleted. Marin places these recovery checkpoints under its [TTL scratch prefix](storage-bucket.md#step-4-ttl-scratch-prefixes-tmpttlnd), where managed cloud stores expire objects 14 days after they are written. `marin.training.training.temporary_checkpoint_base_path(output_path)` returns the resolved recovery root. Local paths use local scratch without lifecycle deletion. Keep the output path and scratch routing stable across retries: automatic resume searches both the permanent and temporary roots and restores the newest complete training state, including optimizer state and the step counter. An interrupted run remains resumable only while its temporary checkpoint exists.

Only the final native checkpoint is permanent by default, under `<output_path>/checkpoints/step-N`. The final save removes the remaining temporary checkpoints after it commits. The Levanter backend also exports the final model under `<output_path>/hf`; permanent native checkpoints do not expire with the temporary prefix. Evaluation does not retain intermediate permanent checkpoints.

To retain intermediate native checkpoints deliberately, set `checkpointer` in the recipe's `SFTSpec`:

```python
from dataclasses import replace
from datetime import timedelta

from levanter.checkpoint import CheckpointerConfig

from experiments.post_training.snowball_sft import OPENTHOUGHTS_32K_4NODE, OPENTHOUGHTS_SPEC

preset = replace(
    OPENTHOUGHTS_32K_4NODE,
    recipe=replace(
        OPENTHOUGHTS_SPEC,
        checkpointer=CheckpointerConfig(
            save_interval=timedelta(minutes=30),
            keep_last_temporary_checkpoints=2,
            keep=[{"every": 1000}],
        ),
    ),
)
```

Here `keep` retains a permanent checkpoint every 1,000 steps in addition to the final save; use `keep=None` for final-only retention. `keep_last_temporary_checkpoints` bounds complete rolling recovery checkpoints after a successful save. Native checkpoints contain full training state and can be much larger than HF exports, so size the permanent cadence explicitly. The generic `marin.experiment.train.train_lm` builder also defaults to rolling recovery checkpoints and a final-only permanent save.

## Match the served vLLM forward

Select `--preset vllm-parity-32k-4node` in the commands above to use dropless routing and FP32 router scoring and activations. For Python workflows, import `VLLM_PARITY_SPEC` from `experiments.post_training.snowball_sft`.

See [Serving Numerical Parity](../references/serving-parity.md) for the measured scope and checkpoint conversion settings.
