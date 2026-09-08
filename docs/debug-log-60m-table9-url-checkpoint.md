# Debugging log for 60M Table-9 URL checkpoint recovery

Recover the missing native Table-9 evaluations without rerunning training.

## Initial status

The historical and phase-order recovery parents materialized exactly 19 and 2
evaluation children. Local fsspec tokenizer staging succeeded, but every child
failed before scoring when Transformers 5 passed the URL-like GCS repository
through `AutoConfig.from_pretrained`. The checkpoint `config.json` exists and
declares `model_type=llama`.

## Hypothesis 1

Transformers 5 no longer consumes the return value from Levanter's temporary
`hf_hub_download` monkeypatch while resolving a configuration. URL-like
repositories therefore need their small Hugging Face metadata staged locally,
while model weights should continue through Levanter's existing fsspec
streaming path.

## Changes to make

- Add one shared Hugging Face config loader that stages configuration metadata
  for URL-like repositories.
- Use the known Hugging Face config class when the converter already has one.
- Exercise both inferred and explicit config classes through an in-memory
  fsspec repository.
- Verify the exact failed GCS checkpoint config locally before resubmission.

## Results

The focused HF utility suite passes all ten tests. Pyrefly reports no errors
in the modified loader. Against the exact failed GCS checkpoint, the loader now
resolves `LlamaConfig`, matches tokenizer and model vocabulary size at 128,256,
and locates the remote `model.safetensors` shard through the existing fsspec
weight path.

An independent Opus 5 review identified two pre-submit edge cases. The loader
now requires a root `config.json` rather than allowing Transformers to silently
construct a default config from an empty dictionary, and URL revisions fail
early to match the existing weight-loader contract. Configuration staging also
excludes weights and custom Python when the converter already knows the config
class.

## Future work

- [ ] Confirm retry children progress through config and model loading.
- [ ] Verify all 21 missing Table-9 result files before closing coverage.
