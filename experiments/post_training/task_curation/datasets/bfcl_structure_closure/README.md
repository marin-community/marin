# BFCL structural completion tasks

The task curation catalog exposes `bfcl-structure-closure-train` and
`bfcl-structure-closure-validation`. The reader creates JSON, tool, reasoning,
and code-fence cases during `process_rows`. Cases that include a reference
in the prompt measure conditional completion. Each BFCL parent belongs to
one cohort, and BFCL-Parity parents stay out of training and validation.

Shared download steps stage the published split audit, oracle inventory,
and initializer tokenizer from a pinned Hugging Face release. The split audit
stores the exact task IDs from the original diagnostic manifest. The inventory
and audit must name the same original split checksum and agree on the
training and validation task IDs. [Campaign release][release]

The download step fetches the original `solution/solve.sh` files at a pinned
Harbor datasets commit and checks each file against its SHA-256 in the
inventory. Case generation reads the answer heredoc as JSON. [Upstream BFCL
source][upstream]

Tool cases use the staged tokenizer's native chat template and advertise the
`write_result` tool schema. The reader serializes tool calls before advancing
training continuation prefixes to a boundary where immediate closure would
drop required content. Select RolloutEngine's `AssistantPrefixHarness` to
preserve the raw continuation for the shared `StructureClosureSpec` grader.

[release]: https://huggingface.co/datasets/open-athena/Complex-Closure-RLVR-Snowball-67B-A2B-Artifacts/tree/d62ce3939fa900b2dad9344428bb7ac16917127a
[upstream]: https://github.com/laude-institute/harbor-datasets/tree/6bedd7878dc5d6f3456b4d80b781eb3c2d84f262/datasets/bfcl
