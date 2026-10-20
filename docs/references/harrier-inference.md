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
