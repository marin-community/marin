# Agent guide: Grug SFT experiments

Write an ordinary experiment module, following
[the Grug launch pattern](../grug/moe/launch.py). The experiment chooses its data,
checkpoint, model, optimizer, training budget, and evaluations. Use the shared
`marin.experiment.cli`; keep model choices out of Datakit's data helpers.

## Define the comparison

State the proposed change and the metric that will decide whether to keep it.
Use the same starting checkpoint, training budget, and evaluation settings for
baseline and variant unless one of those is the variable being tested. Record
the tokenizer revision, chat template, decoding settings, and token limits.
There is no established numerical baseline for this Datakit SFT pipeline yet.

## Prepare the data

The [262k preparation recipe](prepare_data.py) builds all registered sources
with a pinned Marin tokenizer and records packed sequence counts. Use the
[Grug Iris submission pattern](../grug/moe/agent.md#job-submission) with CPU resources
near the data, running this command inside the job:

```bash
python -m experiments.grug_sft.prepare_data --version 2026.09.12 --max-concurrent 2 --run
```

It prepares data only. Completed source stores are reused on restart.

Select sources from `marin.datakit.sft_sources.all_sft_sources()` and build their
`.normalized` steps. For a new source, follow
[the chat dataset tutorial](../../docs/tutorials/add-chat-dataset.md).

Use `build_sft_store` from
[marin.datakit.sft](../../lib/marin/src/marin/datakit/sft.py) in the experiment's
data preparation. Build one store per source, passing `[SftInput(name, path)]`
for its normalized
Parquet directory (`<source.normalized.output_path>/outputs/main`), the frozen
tokenizer path, context length, shuffle seed, shard count, and worker limit.
Keep the results in a `stores` mapping keyed by source name and reuse them
when comparing training changes.

Each store retains conversation rows and records retained, overlength, and
packed-sequence counts. `sft_data_config` weights sources by their packed counts
and concatenates sources below `minimum_weight` into a shared component. If that pool is still
too small, it includes the smallest remaining source. Empty sources are omitted.
Packing uses whole conversations, all-token loss, and attention and loss masks
across conversation boundaries. Existing per-source exact deduplication remains;
cross-source deduplication and benchmark decontamination are not part of this builder.

Use the checkpoint's tokenizer vocabulary and export it with
`marin.datakit.chat_template.MARIN_CHAT_TEMPLATE`. Pin the source revision and
use that same tokenizer artifact for preprocessing, training, and evaluation.
Do not change token IDs to accommodate a different template. Given your chosen
checkpoint tokenizer, pinned revision, and output directory:

```python
from transformers import AutoTokenizer

from marin.datakit.chat_template import MARIN_CHAT_TEMPLATE

tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, revision=tokenizer_revision)
tokenizer.chat_template = MARIN_CHAT_TEMPLATE
tokenizer.save_pretrained(tokenizer_output)
```

## Build the training experiment

Return an `ArtifactStep[LevanterCheckpoint]` from the module's `build()` function,
using the existing trainer for the model. Declare its data dependencies and
choose accelerator resources through runtime arguments, as in the Grug example.
Finish the module with:

```python
from marin.experiment.cli import experiment_main

if __name__ == "__main__":
    experiment_main(build)()
```

For the June Grug model, reuse
[GrugMoeSFTConfig and run_grug_moe_sft_trial](../june_tpu_67b_a2b/moe/sft_launch.py).
Build one `LmDataConfig` for 80% SFT and 20% pretraining replay. Given the
pretraining experiment's data config with fixed source weights:

```python
from dataclasses import replace

from marin.datakit.sft import sft_data_config

block_size = 5 * pretraining_data.mixture_block_size
sft = sft_data_config(stores, minimum_weight=1 / (0.8 * block_size))
pretraining_weights = pretraining_data.train_weights
weight_sum = sum(pretraining_weights.values())
data = replace(
    sft,
    components={
        **sft.components,
        **{f"pretrain/{name}": component for name, component in pretraining_data.components.items()},
    },
    train_weights={
        **{name: 0.8 * weight for name, weight in sft.train_weights.items()},
        **{f"pretrain/{name}": 0.2 * weight / weight_sum for name, weight in pretraining_weights.items()},
    },
    mixture_block_size=block_size,
)
```

Pass `data` to `GrugMoeSFTConfig`. Pretraining caches must use the same token IDs
as the SFT store. Keep their cache grouping and continuous packing; SFT keeps
whole-conversation packing. Increasing the sampling block fivefold compensates
for the 20% share so rare sources do not round to zero.

The [reference LCR configuration](https://github.com/marin-community/marin/blob/09989c43010e9fef0a5520cdc4af8bae90252a06/experiments/june_tpu_67b_a2b/moe/sft_datakit_chat_mix.py)
used a 49,152-sequence pretraining block and four copies of long-document caches.
The combined block is therefore 245,760 sequences. The 80/20 split counts
fixed-length training sequences, not loss-bearing tokens: SFT sequences can
contain padding. For SFT-only experiments, use `minimum_weight=1 / block_size`
and set the returned config's `mixture_block_size` to that block size.

Choose the remaining settings in the experiment. The launcher initializes
checkpoint weights with a fresh optimizer and step counter; restarts resume the experiment's own full state.

On TPU, select `attention_implementation="tpu_splash"` and
`moe_implementation="ring"`. Set `context_parallel` explicitly, make the model's
context length match the store's, and use a multiple of 128 tokens per context
shard. The context-parallel port has CPU model and Splash simulator parity
coverage; a full-context TPU training run remains unvalidated.

## Run and evaluate

For a module saved as `experiments/grug_sft/my_experiment.py`, preview the plan:

```bash
uv run python -m experiments.grug_sft.my_experiment --version dev
```

Use the [Grug Iris submission pattern](../grug/moe/agent.md#job-submission) with
your module and TPU allocation; add `--run` to execute. Use immutable versions
for prepared data so training iterations can reuse it. `--version dev` is for
the experiment; it rebuilds any dependency whose version also resolves to `dev`.

First run a short pilot. Check source counts, loss, throughput, checkpoint
recovery, and a few generated conversations, including tool calls and stopping
behavior where relevant. Then run the matched baseline and variant and evaluate
both with the same harness, using the shared
[evaluation steps](../../lib/marin/src/marin/experiment/evaluation.py).
Report final metrics only after the runs finish;
include failures and truncation rates alongside task scores. A lower training
loss alone does not establish better SFT behavior.
