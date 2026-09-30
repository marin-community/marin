# Hero inference export

Export one permanent native `moe_hero_ep` checkpoint to BF16 Hugging Face
weights for the Marin vLLM fork. The producer accepts an explicit checkpoint
and destination; it does not discover checkpoints or allocate hardware.

Create an export YAML with these fields:

```yaml
checkpoint: s3://marin-us-east-02a/marin/<run>/checkpoints/step-144000
metadata_digest: <SHA-256 of canonical checkpoint metadata JSON>
model: <complete GrugModelConfig mapping from the training run>
destination: s3://marin-us-east-02a/marin/<new-export-root>
source_revision: <40-character Marin producer commit>
expert_axis_size: 32
replica_axis_size: 1
```

`metadata_digest` uses `weights.metadata_hash`: SHA-256 of `metadata.json`
decoded and re-encoded with sorted keys and separators `(',', ':')`. Pin the
full model configuration from the source run, including RoPE, ShortConv,
local/global KV heads and shared experts. `source_revision` records the
producer revision supplied by the caller; launch that exact checkout or
bundle. The source checkpoint must remain immutable throughout all attempts.

Run in the existing Levanter GPU environment:

```bash
uv run --package marin-levanter --extra gpu python -m \
  experiments.grug.moe_hero_ep.ops.export_vllm --config_path export.yaml
```

Launch every process with the same YAML and use the usual Iris/JAX distributed
initialization. `expert_axis_size` and `replica_axis_size` select the existing
compact Grug mesh; they default to one for local fixtures. Full Hero previously
used 32 GB200s, four GPUs per task and 400 GB host RAM per task. These are
execution choices, not constraints imposed by the producer. Use region-local
CW storage and interactive priority for validation. Iris launch and dev-GPU
procedures are in [Iris operations](../../lib/iris/OPS.md).

## Weight and file contract

The shared weights-only restore handles current manifests, older OCDBT
checkpoints and directory-backed layouts, including the legacy `train_state`
wrapper. It selects `master_params` when present, otherwise `params`, and
requires the pinned permanent metadata. Pending QB betas set the effective
router bias to `-beta + mean(beta)` exactly once before BF16 conversion.

The model's existing inference mapping defines every weight and HF config.
Only this producer splits routed expert banks into
`experts.<global_id>.<gate_proj|up_proj|down_proj>.weight`, each a two-dimensional
`[output, input]` matrix. Snowball's stacked mapping stays unchanged. vLLM
requires the split loader from [Marin vLLM #77](https://github.com/marin-community/vllm/pull/77).

The writer stages one layer at a time, plus safetensors serialization buffers;
global tensors form one separate shard. Each rank replicates parameters in
the same order, while process zero alone writes files. The largest individual
expert bank is still replicated on each rank during its gather. This preserves
the historical producer's memory pattern and does not promise an arbitrary
host-byte cap. The generic `HFCheckpointConverter.save_pretrained` has
concurrent budgeted writers, but does not provide durable progress or skip
completed gathers; it is not used to add another generic export interface.

## Interruption and completion

Use a fresh destination and run only one exporting gang against it at a time.
`export-request.json` pins the checkpoint metadata digest, complete model
config, source revision, mesh settings, destination and HF config before any
shard is written. Re-running the same YAML resumes partial output.

Before reusing a shard, the producer checks its identity and exact expected
tensor names, byte size and freshly computed SHA-256 of its stored contents.
Corruption stops the export and preserves the object for inspection. An upload
without its progress record may be rewritten on resume. Shards with verified
progress are preserved. Changed inputs require a fresh destination.

`config.json`, `model.safetensors.index.json` and `export-manifest.json` are
published after all shard uploads succeed. The manifest is the completion
marker; a completed destination is refused on subsequent runs. The HF index
reports tensor payload bytes, while the manifest reports safetensors file
bytes and hashes. Tokenizer files are not copied; use the source run's pinned
tokenizer separately.

This producer derives from the executed
[experimental writer at da624140](https://github.com/marin-community/marin/blob/da624140b92912c2bfef705320a477b91b8df4de/experiments/grug/moe_hero_ep/ops/export_vllm.py),
whose split layout began in
[d626958619](https://github.com/marin-community/marin/commit/d626958619c1666c9759276178e0f5fdbafa59ae).
Small-checkpoint comparison and loading evidence accompany the producer PR;
they do not constitute a fresh full Hero export or a vLLM wheel release.
