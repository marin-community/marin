# Hero inference export

Choose a permanent checkpoint and fresh destination in the same region.
Copy `model` from the training run's recorded `config.yaml` artifact or resolved
launch configuration into `export.yaml`:

```yaml
checkpoint: s3://marin-us-east-02a/marin/<run>/checkpoints/step-144000
metadata_digest: <digest from the command below>
model: <complete GrugModelConfig mapping>
destination: s3://marin-us-east-02a/marin/<new-export-root>
source_revision: <40-character lowercase hexadecimal Marin exporter commit>
expert_axis_size: 32
replica_axis_size: 1
```

Compute `metadata_digest`; replace `<checkpoint>` with the YAML's checkpoint URL:

```bash
uv run --package marin-levanter --extra gpu python - <<'PYTHON'
import json
from experiments.grug.moe_hero_ep.weights import metadata_hash
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
configure_coreweave_s3()
print(metadata_hash(json.loads(StoragePath("<checkpoint>/metadata.json").read_text())))
PYTHON
```

Launch the checkout or bundle pinned by `source_revision` on every JAX process
with the same YAML. Use [Iris launch procedures](https://github.com/marin-community/marin/blob/main/lib/iris/OPS.md):

```bash
uv run --package marin-levanter --extra gpu python -m \
  experiments.grug.moe_hero_ep.ops.export_vllm --config_path export.yaml
```

Keep the checkpoint immutable. Run one exporting gang per destination. The
global device count must be divisible by `expert_axis_size * replica_axis_size`;
both axes default to one. Every device must fit the largest expert bank.
Process zero needs host RAM for one layer plus serialization buffers.

Load the BF16 output with [Marin vLLM's split-expert loader](https://github.com/marin-community/vllm/pull/77).
Tokenizer files are not copied. Set vLLM's `--tokenizer` and
`--tokenizer-revision` to the training run's tokenizer ID and pinned revision.

## Resume and completion

After interruption, rerun the same YAML. Resume preserves shards only after
checking their identity, names, size and freshly computed SHA-256. Uncommitted
uploads may be rewritten. If corruption is reported, inspect the object;
restore its original bytes or choose a fresh destination. Changed inputs also
require a fresh destination.

Use the output only when `export-manifest.json` exists. It is written after
all shards, `config.json` and `model.safetensors.index.json`. Completed
destinations are refused on subsequent runs.
