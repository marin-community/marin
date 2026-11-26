# Inference correctness specifications

The `specs/` files select shared immutable goldens, Iris resources, the serving
engine and provisional comparison limits for BF16 Snowball v0.1.1 and Hero/Mochi
SFT step400407. Reference manifests, checkpoint/tokenizer identities and bounds
come from [#11320](https://github.com/marin-community/marin/pull/11320), source
`3d04a90d36e393683a25ea01c1b3528f12ec3204`.

See [the checker reference](../../../docs/references/inference-correctness.md)
for Python, CLI and pytest usage, report fields, units and versioning. Callers
own Iris allocation, `vllm serve`, cleanup and explicit report saving. Checks
keep observations in memory and never generate goldens.
