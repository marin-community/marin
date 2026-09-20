# Grug 67B Datakit SFT ablations

`head_only_scaled.py` reproduces the September 17 head-initialized, scaled-LR,
frozen-router recipe and supports leave-groups-out data ablations. Its source was
recovered from Iris bundle
`85b3d97a5fed66084b5633e32e910aec59219657761e179fca9f968e436b147f`.

## Mixture behavior

The baseline uses the recorded 23-group SFT allocation at 80% and the bundled
long-context replay manifest at 20%. It restores the complete step-157000 train
state, trains through step 158000, and enforces the original one-pass SFT limit.

Each `--exclude-group` removes one complete allocation group. The launcher
renormalizes the other SFT weights to 80% and keeps replay at 20%. The update
count, checkpoint, optimizer, and hardware stay fixed. Surviving SFT stores can
restart because the renormalized mixture cannot retain the baseline's one-pass
limit.

The baseline and ablations need distinct immutable versions. The version and a
stable hash of the sorted exclusion set form the W&B run ID and GCS output path.
Changing the order of repeated `--exclude-group` arguments does not change the
identity.

## Launch

These commands submit a non-preemptible `v4-2048` training job. They do not have
a dry-run mode.

Reproduce the baseline under a new identity:

```bash
uv run python -m experiments.grug_sft.head_only_scaled \
  --version baseline-v1
```

Run a leave-one-group-out ablation:

```bash
uv run python -m experiments.grug_sft.head_only_scaled \
  --version math-drop-v1 \
  --exclude-group nemotron_sft/sft_math
```

Repeat the flag for a combined ablation:

```bash
uv run python -m experiments.grug_sft.head_only_scaled \
  --version agent-data-drop-v1 \
  --exclude-group agenttrove \
  --exclude-group agenttrove-glm53-compactions \
  --exclude-group swe-zero-12m
```

Group names must match the keys in the recorded `allocation.json`. Unknown,
duplicate, or all-group exclusions fail before training starts. Versions may
contain letters, digits, `.`, `_`, and `-`.
