# Fine-tune Snowball at 262K Context on Your H100 Cluster

This tutorial runs supervised fine-tuning from Hugging Face inputs on one
8xH100 node in your cluster. The experiment downloads the Snowball base model
and OpenThoughts Agent dataset, prepares the chat data, converts the model
checkpoint, and runs a short training job. All generated artifacts go to your
`MARIN_PREFIX`; the run does not require access to Marin's storage.

The default command samples 256 source records and trains for 10 steps. Human
setup takes about 15 minutes. The full wall time has not been measured end to
end. The first run downloads and converts a 67B MoE checkpoint, processes the
source data, compiles the model, and trains. Later runs reuse completed
artifacts under the same prefix.

## Prerequisites

- Complete the [installation tutorial](installation.md) in the image or
  environment used by your cluster workers.
- Make one node with eight H100 GPUs, 64 CPU cores, 768 GiB of RAM, and 384 GiB
  of local disk available to the job.
- Create a persistent artifact prefix that the job can read and write. A shared
  filesystem path or any [fsspec](https://filesystem-spec.readthedocs.io/)
  object-store backend supported by your environment can be used.
- Export an `HF_TOKEN` that can read
  [`open-athena/snowball-67b-a2b-base-262k-qk175-skew8`](https://huggingface.co/open-athena/snowball-67b-a2b-base-262k-qk175-skew8)
  and
  [`open-thoughts/OpenThoughts-Agent-SFT-100K`](https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-SFT-100K).
- Make the Hugging Face token and storage credentials available inside the
  scheduled job.

The experiment pins the Hugging Face revisions for both inputs. A rerun uses
the same model and data even if either repository changes later.

The H100 resource request does not select a GPU memory size or interconnect.
The 8xH100 path has not completed an end-to-end run, so other H100 layouts are
unverified.

## Prepare the worker environment

From the Marin checkout, install the workspace and CUDA-enabled JAX packages:

```bash
uv sync --all-packages --extra=gpu
```

Verify on a scheduled GPU node that the worker environment exposes all eight
GPUs:

```bash
nvidia-smi --list-gpus
```

The command should print eight H100 devices. See [Setting up a Local GPU
Environment](local-gpu.md) for the required NVIDIA driver and JAX runtime.

## Set the artifact prefix

Choose a persistent prefix for this run and export the Hugging Face token in
the scheduled job. This example uses a shared filesystem:

```bash
export MARIN_PREFIX=/shared/training/snowball-demo
export HF_TOKEN=hf_example
```

The prefix stores the downloaded dataset, normalized chat records, tokenized
data, converted checkpoint, and training checkpoints. An `s3://` prefix also
works when the job has the required credentials. Keep object storage in the
same region as the compute nodes.

## Inspect the execution graph

Print the graph before allocating the H100 node:

```bash
uv run python -m experiments.grug_sft.snowball_262k_h100 \
  --version 2026.09.22.1 \
  --steps 10 \
  --sample-count 256
```

The graph contains these stages:

1. Download and normalize OpenThoughts Agent through the registered Datakit
   source.
2. Download and convert the pinned Snowball Hugging Face checkpoint to Grug's
   stacked checkpoint layout.
3. Render, tokenize, and pack the sampled conversations into a Levanter store.
4. Train at a sequence length of 262,144 tokens with context parallelism across
   all eight H100s.

## Run on an allocated H100 node

Request one complete 8xH100 node from your cluster, then run this command inside
that allocation:

```bash
uv run python -m experiments.grug_sft.snowball_262k_h100 \
  --version 2026.09.22.1 \
  --steps 10 \
  --sample-count 256 \
  --run
```

In this mode, Marin's local execution backend runs preprocessing and training
on the allocated node. No separate coordinator service is required.

## Run through Iris

If your cluster runs Iris, submit the same command as a CPU-only Iris job
instead. Iris acts as the coordinator and dispatches the preprocessing and
8xH100 child tasks from the resource requests in the experiment. This
coordinator pattern is specific to Iris; do not allocate H100s to the
coordinator itself. Configure the Iris task environment so the coordinator and
child tasks receive the same `MARIN_PREFIX`, `HF_TOKEN`, and storage
credentials.

Use a new `--version` when you want a separate output. Reusing the same version
resumes incomplete work and skips stages whose success records already exist.

The final training artifacts are written below:

```text
<MARIN_PREFIX>/grug-sft/snowball-67b-262k-h100-demo/<version>/
```

Permanent checkpoints are in the `checkpoints/` child directory. Temporary
checkpoints are retained separately while the run is active.

The first training step includes JAX compilation, so it takes longer than later
optimizer steps. Training checks for a permanent checkpoint every 30 minutes;
it does not write one after every optimizer step. Each completed graph stage
writes a success record under the artifact prefix.
