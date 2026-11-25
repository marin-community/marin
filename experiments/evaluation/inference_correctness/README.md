# Inference correctness bindings

These JSON files bind the shared endpoint checker to Snowball v0.1.1 and
Hero/Mochi SFT step400407. They retain the checkpoint and tokenizer identities,
saved reference manifests and provisional acceptance policies from
[#11320](https://github.com/marin-community/marin/pull/11320), source
`3d04a90d36e393683a25ea01c1b3528f12ec3204`.

See [the checker reference](../../../docs/references/inference-correctness.md)
for Python, CLI and endpoint-only pytest examples, required inputs and reports.
Callers own deployment and cleanup. Saved scores remain in object storage;
routine checks do not generate goldens.
