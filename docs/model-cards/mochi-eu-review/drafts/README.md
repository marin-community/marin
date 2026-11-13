---
license: openmdw-1.1
pipeline_tag: text-generation
tags:
  - marin
  - grug_moe
  - mixture-of-experts
  - base-model
---

# Marin Mochi 535B-A23B

**DRAFT CARD — disclosure additions await the confirmations in [the publication review](REVIEW.md). Do not publish with unresolved fields.**

Research artifact. This model has not been properly tested or evaluated and is not necessarily secure. Use at your own risk in production settings.

BF16 weights-only exports of retained Hero pre-training checkpoints. The model
has 535,477,106,688 stored parameters and approximately 22.76B active
non-embedding parameters per token; `A23B` rounds that active count to the
nearest billion.

## Access status

Weight downloads are gated and require provider approval. As of 4 November
2026, no access requests have been approved.

**Planned release date:** 5 November 2026. Actual first availability will be
recorded when access opens. **[CONFIRM C02: refresh the access status and
record the actual date at release.]**

## Checkpoint branches

Each checkpoint uses a branch named `step-N`. The retained inventory contains
48 checkpoints from step 6000 through step 378000. Uploads are in progress.
A branch is complete only when `hf-upload-manifest.json` exists and pins a
verified weights commit. The `main` branch contains repository metadata.

Read the checkpoint branch's model card and `config.json` for its training
phase, context settings and source revision. Earlier phases use 4K context,
later phases use 16K, and step 378000 uses 65K. Each completed checkpoint contains
193 safetensors shards with a BF16 tensor payload of 1,070,954,213,376 bytes.
Optimizer and data-loader state are excluded.

## Provider and training documentation

Provider: **Open Athena AI Foundation Inc., United States.**
Contact: **[privacy@openathena.ai](mailto:privacy@openathena.ai).**
Copyright questions and complaints: **[privacy@openathena.ai](mailto:privacy@openathena.ai).**
**[CONFIRM C01: proposed contact; confirm routing and monitoring before publication.]**

The [training-content summary](TRAINING_CONTENT.md) identifies the model
versions it covers, their training sources and relevant data-processing
measures. Read the summary applicable to the selected checkpoint.
The [copyright policy](COPYRIGHT_POLICY.md) describes source review,
rights-reservation handling and the complaint process.
**[CONFIRM C04/C07/C11: approve version coverage, policy implementation and
permanent website/HF links before publication.]**

The [Datakit overview](https://marin.community/blog/2026/09/11/marin-data-pipeline-overview/)
and [source reconstruction guide](https://github.com/marin-community/marin/blob/67657d4b9347c5341bbd9ba0fa4fdeeeb8f510e3/docs/reproducibility/pretraining-source-pool.md)
describe the source catalog and processing recipes. The training summary
supplies the version-specific account; the available source pool alone does
not identify the inputs to a checkpoint.

The context-corrected architecture estimate for pre-training through the
latest retained checkpoint, step 378000, is approximately **2.7 × 10²⁴ FLOP**.
It counts active MoE forward/backward computation across the 4K, 16K and
65K phases. It excludes activation recomputation. **[CONFIRM C08: complete
any additional applicable compute accounting before presenting this as a
regulatory cumulative total.]** Earlier checkpoint totals are lower; exporting
48 checkpoints does not multiply the training compute by 48.

## Inference and validation

Serving requires the [Marin vLLM fork](https://github.com/marin-community/vllm)
with its [split-expert loader](https://github.com/marin-community/vllm/pull/77).
Artifact integrity checks verify dtype, names, sizes and hashes. Full inference
numerical parity and serving recipes have not been validated for these exports.
Generic Transformers loading has not been validated. These are base models;
a bundled tokenizer chat template does not indicate instruction tuning.
**[CONFIRM C09: provide sufficient versioned integration/use information
before relying on the open-source documentation exemption.]**

The bundled tokenizer is `marin-community/marin-tokenizer` at revision
`a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2`, the pin used by Hero's native
sampling configuration. The training runs record the tokenizer repository but
do not record an exact tokenizer revision.

## Copyright and use

Copyright-infringing uses of this model are prohibited by applicable law.
The model license does not grant rights in third-party works or guarantee
that generated material can be reproduced or distributed lawfully.
Users must assess the rights and restrictions applicable to their use.

## License

OpenMDW 1.1. See [LICENSE](https://github.com/OpenMDW/OpenMDW/blob/main/1.1/LICENSE.OpenMDW-1.1) for the complete text.

We release our tools to the public with the intent that they are used safely, ethically, and responsibly. They are meant to contribute to the public's understanding of science, and aid in its advancement. By using this Marin model, you agree to comply with all applicable law.
