# Hero inference export

Export one permanent native `moe_hero_ep` checkpoint to BF16 Hugging Face
weights for the Marin vLLM fork.

Create an export YAML with these fields:

```yaml
checkpoint: s3://marin-us-east-02a/marin/<run>/checkpoints/step-144000
metadata_digest: <SHA-256 of canonical checkpoint metadata JSON>
model: <complete GrugModelConfig mapping from the training run>
destination: s3://marin-us-east-02a/marin/<new-export-root>
source_revision: <40-character lowercase hexadecimal Marin producer commit>
expert_axis_size: 32
replica_axis_size: 1
```

`metadata_digest` uses `weights.metadata_hash`: SHA-256 of `metadata.json`
decoded and re-encoded with sorted keys and separators `(',', ':')`. Use the
complete model configuration from the training run. Launch the exact checkout
or bundle recorded in `source_revision`. Keep the source checkpoint immutable
throughout all attempts.

Run in the existing Levanter GPU environment:

```bash
uv run --package marin-levanter --extra gpu python -m \
  experiments.grug.moe_hero_ep.ops.export_vllm --config_path export.yaml
```

Launch every process with the same YAML using the usual Iris/JAX distributed
initialization. The global device count must be divisible by
`expert_axis_size * replica_axis_size`; the data axis uses the remaining
devices. Both configured axes default to one for local fixtures. Use
region-local storage. See [Iris operations](https://github.com/marin-community/marin/blob/main/lib/iris/OPS.md)
for launch procedures.

## Weight and file contract

Restore supports current manifests, older OCDBT and directory-backed layouts,
including the legacy `train_state` wrapper. It selects `master_params` when
present, otherwise `params`. Pending QB betas set the router bias to
`-beta + mean(beta)` exactly once before BF16 conversion.

The export splits routed expert banks into
`experts.<global_id>.<gate_proj|up_proj|down_proj>.weight`, each a two-dimensional
`[output, input]` matrix. Loading requires the split loader from
[Marin vLLM #77](https://github.com/marin-community/vllm/pull/77). Use the training
run's pinned tokenizer separately; tokenizer files are not copied.

Gathering replicates the largest expert bank on every device. Process zero
stages one layer in host RAM plus serialization buffers, then writes files;
global tensors form a separate shard. There is no host-byte cap.

## Interruption and completion

Use a fresh destination and run only one exporting gang against it at a time.
`export-request.json` records the inputs before any shard is written.
Re-running the same YAML resumes partial output. Changed inputs require a
fresh destination.

Resume re-reads completed shards to verify their identity, tensor names, size
and SHA-256. Verified shards are preserved. Corruption stops the export and
preserves the object for inspection. An upload without its progress record may
be rewritten.

`config.json`, `model.safetensors.index.json` and `export-manifest.json` are
published after all shard uploads succeed. The manifest is the completion
marker; a completed destination is refused on subsequent runs.
