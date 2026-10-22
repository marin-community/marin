# Hugging Face inference releases

Levanter and the vendored MarinSkyRL checkpoint exporter use policy version 1
for Hugging Face inference releases. Native training checkpoints keep their
backend format and are not governed by this policy. Adapter-only PEFT artifacts
also keep their own format.

## Precision

`release_precision` accepts `bf16`, `fp32`, or `preserve`. The default is
`preserve` until a representative Snowball export passes the supported Marin
vLLM runtime gate described below. `bf16` stores ordinary floating tensors as
BF16; `fp32` stores them as FP32; `preserve` retains their source dtype. Grug
router biases remain FP32 in every mode. A missing or lower-precision router
bias fails export. The release `config.json` contains a scalar `dtype` only
when ordinary floating weights have a common dtype.

Levanter's standalone export accepts `release_precision` in its export config.
Training hooks use the existing `hf_save_dtype` setting: `bfloat16` selects
`bf16`, `float32` selects `fp32`, and `null` selects `preserve`. The vendored
SkyRL checkpoint export request stores `release_precision` and
`release_policy_version` with the checkpoint path and step. It also stores a
40-character source Hugging Face commit when Hub metadata contributes to the
release, explicit EOS IDs, generation controls, and a chat template when
supplied. If an optional reference config is unavailable and the exporter uses
its local architecture config, it records no Hub source. A retry must use the
same persisted request.

## Metadata and validation

Explicit export values take precedence over checkpoint metadata, then pinned
source Hugging Face metadata. Required vocabulary and EOS IDs must resolve.
Grug releases also require a chat template. The generation EOS list must
contain the tokenizer EOS ID. Export rejects conflicting Grug architecture
aliases and writes one canonical name per field, retaining its architecture
schema version. The exporter does not add stop or suppression tokens while
serializing metadata.

SkyRL validates other supported Transformers causal language models against
the state-dict names and shapes declared by their HF model class. It reads
generation metadata from a local, pinned Hub, or cloud source directory when
available. For these non-Grug models, the tokenizer EOS ID supplies the
generation EOS list if the request and source metadata omit it. Grug requires
an explicit or source generation EOS list so an additional end-of-turn token
cannot silently disappear.

The release validator checks safetensors headers against expected tensor names,
shapes, and dtypes, plus index coverage and byte totals. It checks tokenizer,
config, and generation metadata before writing `hf_release_manifest.json`.
The manifest records hashes of the release files and source provenance.
`hf_release_request.json` records the resolved policy for retries. The
completion manifest is the signal that a directory is ready for publication.

## Enable BF16 by default

Before changing the default to `bf16`, load a representative Snowball release
in the supported Marin vLLM runtime. Compare fixed-prompt logits with the
source checkpoint using a declared precision tolerance. Verify generation
terminates on each configured EOS token. Keep existing published Hugging Face
revisions immutable; evaluate a replacement release separately.
