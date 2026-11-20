# Native Llama

`model.type: llama_native` selects the array-backed Llama implementation in
`levanter.main.train_lm` and `levanter.main.eval_lm`. The default `llama` model
continues to use named arrays. Native Llama keeps the `LmHeadModel` input and
loss interface and stores parameters in output-first Hugging Face layouts.

See [Native model sharding](Native-Model-Sharding.md) for the mesh and parameter layouts.
Native Llama requires scanned layers, a replicated sequence dimension and
compatible head, MLP and vocabulary dimensions. Fresh runs pad the model
vocabulary to a multiple of the tensor-parallel size and preserve tokenizer
IDs. The default attention path uses Grug reference attention. Set
`model.attn_backend: splash` to use TPU Splash.

Native Llama supports unquantized training, HF weight initialization, full-state
Levanter checkpoints, and tagged held-out evaluation. Set `hf_save_path` to a
local directory and `hf_save_steps` to export full-model HF snapshots. Export
requires standard weighted RMSNorm and bias-free projections. Training from
scratch exports using the data tokenizer and the model configuration. The
training entry point rejects quantization, adapters, Hub uploads, and optional
entropy and harness callbacks.

Native checkpoints include model architecture and
vocabulary metadata; restore validates those fields before constructing the
explicitly sharded template. Metadata-free weight loading relies on the
caller's complete vocabulary and architecture configuration.
