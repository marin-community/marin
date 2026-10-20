# Harrier inference with Levanter

`HarrierJaxEmbedder` loads the pinned `microsoft/harrier-oss-v1-0.6b` base
Qwen3 checkpoint and returns 1,024-dimensional `float32` embeddings. It uses
Levanter's Qwen3 forward pass on JAX CPU, GPU, or TPU devices. The loader adds
the causal-LM `model.` key prefix while reading the base checkpoint. The
prefix rewrite does not copy tensor data; this checkpoint has a tied output
head, so inference only uses the base model.

```python
from marin.processing.embeddings.harrier import HARRIER_DIM, HARRIER_REPO, HARRIER_REVISION
from marin.processing.embeddings.harrier_jax import HarrierJaxConfig, load_harrier_jax

config = HarrierJaxConfig(
    model=HARRIER_REPO,
    revision=HARRIER_REVISION,
    tokenizer_revision=HARRIER_REVISION,
    dimensions=HARRIER_DIM,
    max_tokens=8_192,
    compute_dtype="bfloat16",
    platform="tpu",  # Use "gpu" on a JAX GPU worker.
)
vectors = load_harrier_jax(config).embed(["First document", "Second document"])
```

The model uses right padding, the last non-padding token, and L2
normalization. Input order is preserved. Documents need no instruction prefix;
retrieval queries should include a task instruction as described in the
[Harrier model card](https://huggingface.co/microsoft/harrier-oss-v1-0.6b).
The `embed` method rejects inputs above `max_tokens` rather than truncating them.

Each worker loads the model once. Texts are grouped into power-of-two sequence
buckets, starting at 128 tokens. Every bucket uses a fixed batch shape, so
repeated calls reuse compiled executables. `batch_size_per_device` limits rows
per accelerator and `batch_token_limit_per_device` limits padded tokens per
accelerator; both apply before multiplication by the local device count. The
model is replicated on local devices and the batch is sharded across them.
The default limits are 32 rows and 16,384 padded tokens per device. A single
input can exceed the token limit when its sequence bucket is larger.

The loader mirrors a pinned Hugging Face snapshot once to a region-local
30-day cache for cluster workers. Set `cache_ttl_days=0` for a local model path
or direct Hub loading. The existing task-curation driver still selects
the Torch CPU embedder; callers of `curation_embedding_pipeline` can
pass `partial(load_harrier_jax, config)` as its `embedder_factory` on GPU or
TPU workers.

## Iris and Zephyr demo

Run the following from the Marin checkout. The shell variable holds the Python
entrypoint passed to Iris. With no input paths it embeds three sample texts.
Additional arguments after the output prefix are Parquet files with string
`id` and `text` columns.

```bash
HARRIER_ZEPHYR_DEMO=$(cat <<'PY'
import sys
from functools import partial

import pyarrow as pa
from fray.types import ResourceConfig
from marin.processing.embeddings.harrier import HARRIER_DIM, HARRIER_MAX_TOKENS, HARRIER_REPO, HARRIER_REVISION
from marin.processing.embeddings.harrier_jax import HarrierJaxConfig, load_harrier_jax
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.input_file import InputFileSpec
from zephyr.readers import load_file
from zephyr.runners import InlineRunner


def embed_batches(batches, shard, *, config):
    embedder = load_harrier_jax(config)
    for batch in batches:
        vectors = embedder.embed([record["text"] for record in batch])
        for record, vector in zip(batch, vectors, strict=True):
            yield {"id": record["id"], "embedding": vector.tolist()}


platform, output_prefix, *input_paths = sys.argv[1:]
config = HarrierJaxConfig(
    model=HARRIER_REPO,
    revision=HARRIER_REVISION,
    tokenizer_revision=HARRIER_REVISION,
    dimensions=HARRIER_DIM,
    max_tokens=HARRIER_MAX_TOKENS,
    compute_dtype="bfloat16",
    platform=platform,
)
if input_paths:
    files = [InputFileSpec(path=path, columns=["id", "text"]) for path in input_paths]
    documents = Dataset.from_list(files).flat_map(load_file)
else:
    documents = Dataset.from_list([
        {"id": "0", "text": "Harrier embeds documents with a Qwen3 encoder."},
        {"id": "1", "text": "Zephyr schedules batches on an accelerator worker."},
        {"id": "2", "text": "The output contains normalized float32 vectors."},
    ])

worker_resources = (
    ResourceConfig.with_gpu("GB200", count=1, cpu=8, ram="32g", disk="64g")
    if platform == "gpu"
    else ResourceConfig.with_tpu("v6e-4", cpu=8, ram="32g", disk="64g")
)
schema = pa.schema([("id", pa.string()), ("embedding", pa.list_(pa.float32(), HARRIER_DIM))])
pipeline = (
    documents.window(128)
    .map_shard(partial(embed_batches, config=config))
    .write_parquet(f"{output_prefix.rstrip('/')}/part-{{shard:05d}}.parquet", schema=schema)
)
context = ZephyrContext(
    resources=worker_resources,
    coordinator_resources=ResourceConfig(cpu=1, ram="8g", preemptible=False),
    max_workers=1,
    name="harrier-jax-demo",
    stage_runner_factory=InlineRunner,
)
context.execute(pipeline, verbose=True)
PY
)
RUN_ID=$(date -u +%Y%m%d-%H%M%S)
```

Launch it with one GB200 GPU worker in the CoreWeave region:

```bash
uv run iris --cluster=cw-us-east-08a job run \
  --cpu 2 --memory 8GB --disk 32GB --extra gpu --sync-package marin-core \
  --enable-extra-resources --priority batch --no-wait -- \
  python -c "$HARRIER_ZEPHYR_DEMO" gpu \
  "s3://marin-us-east-08a/tmp/ttl=7d/harrier-jax-zephyr-demo/gpu-$RUN_ID"
```

Or launch it with a four-chip v6e-4 TPU worker in `us-east5`:

```bash
uv run iris --cluster=marin job run \
  --region us-east5 --cpu 2 --memory 8GB --disk 32GB --extra tpu \
  --sync-package marin-core --enable-extra-resources --priority batch --no-wait -- \
  python -c "$HARRIER_ZEPHYR_DEMO" tpu \
  "gs://marin-us-east5/tmp/ttl=7d/harrier-jax-zephyr-demo/tpu-$RUN_ID"
```

Iris starts the command as a CPU driver. Inside that job, `ZephyrContext` uses
Iris to submit a CPU coordinator and one accelerator worker. The worker's
`ResourceConfig` selects GB200 or v6e-4. `--sync-package marin-core` installs
the workspace package, while the driver's `--extra gpu` or `--extra tpu`
selects the JAX dependencies inherited by its child jobs.

The worker loads the pinned checkpoint once, groups texts into fixed-shape
batches, and writes normalized vectors as 1,024-element `float32` lists.
CoreWeave GPU jobs use regional S3 storage; Marin TPU jobs use regional GCS
storage. Zephyr also writes intermediate chunks to regional temporary storage.
`RUN_ID` gives each run a fresh output prefix.

For larger inputs, pass Parquet paths after the output prefix and increase
`max_workers` to run more accelerator workers. Keep the input and output in the
same region as the chosen cluster.
