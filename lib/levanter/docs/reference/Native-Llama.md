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

The initial route supports unquantized training, HF weight initialization,
full-state Levanter checkpoints and tagged held-out evaluation. Quantization,
adapters, HF export and optional entropy/harness callbacks are rejected by
the training entrypoint. Native checkpoints include model architecture and
vocabulary metadata; restore validates those fields before constructing the
explicitly sharded template. Metadata-free weight loading relies on the
caller's complete vocabulary and architecture configuration.
