# Converted science-forward Snowball SFT

## Dr Doom two-epoch run

`dr_doom.py`, `snapshot.py`, `prepare.py`, `train.py`, and
`export_dr_doom.py` define the October 2 run from the pinned Dr Doom export.
The conversion producer is in `experiments/datakit/science_sft_conversion`;
its Ortet batch launch is documented in `ORTET.md` there. The exact producer
and handoff commands, run configuration, evaluation traces, and audits are in
the [public experiment archive](https://huggingface.co/datasets/open-athena/marin-science-expert-sft-results-2026-10).
The frozen 3,477,625-row training snapshot is in a [private dataset](https://huggingface.co/datasets/open-athena/marin-science-expert-sft-2026-10) pending review of the transformed sources' original licenses and derivative-output rights.
The snapshot manifest fixes the input file set. The training code checks the
model revision and tokenizer template before consuming the packed store.

## Step38 conversion run

This run starts from the pinned Step38 Snowball Hugging Face export and trains
one pass over the MiniMax-converted science-forward chat corpus. The trainer
packs conversations at 32,768 tokens, applies loss only to assistant reasoning
and final answers, and overwrites the frozen router bias with the previous
step's QB thresholds. It uses eight H100x8 nodes, batch size 64, a 5e-6 peak
learning rate, 5% warmup, and W&B online logging.

The converted source must first pass the completeness audit documented in
`experiments/datakit/science_sft_conversion/README.md`. Build the token store
from the science SFT worktree on RNO2A with CoreWeave storage access. The
tokenizer uses the model's pinned Hugging Face revision; training weights come
from the CoreWeave mirror:

```bash
uv run iris --cluster=cw-rno2a job run --priority interactive --enable-extra-resources \
  --job-name science-sft-converted-store-20260927 --cpu 8 --memory 32GB \
  --disk 20GB --extra cpu --no-wait \
  -- python -m experiments.grug.science_sft.prepare \
    --input-path s3://marin-us-east-02a/marin/users/benfeuer/science-sft-converted/2026.09.27-v3/outputs/main \
    --output-path s3://marin-us-east-02a/marin/users/benfeuer/science-sft-converted-store/2026.09.27-v3 \
    --tokenizer open-athena/Snowball-67B-A2B-5.7T-Mixed-RLVR-Step38@cfc1d845dae89b067cdc7250d0164abefa5a69cf \
    --num-shards 1024 --max-workers 32
```

Inspect the resulting artifact's source counts. Preparation fails if any
conversation exceeds the 32K context. Once the store is complete, submit the
training coordinator from the same worktree. Its task environment must contain
CoreWeave credentials and `WANDB_API_KEY`:

```bash
uv run iris --cluster=cw-rno2a job run --priority interactive --enable-extra-resources \
  --job-name science-sft-converted-step38-20260927 --cpu 8 --memory 32GB \
  --disk 20GB --extra cpu --no-wait -e WANDB_API_KEY "$WANDB_API_KEY" \
  -- python -m experiments.grug.science_sft.launch \
    --store-path s3://marin-us-east-02a/marin/users/benfeuer/science-sft-converted-store/2026.09.27-v3 \
    --version 2026.09.27-v3
```

The launcher checks the model mirror revision and refuses a store with a
different tokenizer, context, or source name. It uses only full 64-sequence
batches; fewer than 64 packed sequences may remain unused at the end.
