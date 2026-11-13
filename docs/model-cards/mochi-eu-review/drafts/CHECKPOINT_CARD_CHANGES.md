# Proposed checkpoint card additions

Draft editing instructions, 4 November 2026. Do not publish this file as a model card.

Keep each checkpoint's architecture, context settings, training/source provenance, tokenizer caveat, export validation limits, required warning, responsible-use paragraph and license metadata. Add the following section immediately before `## Tokenizer and inference`, after resolving its confirmation fields.

```markdown
## Provider and training documentation

Provider: Open Athena AI Foundation Inc., United States.
Contact: [privacy@openathena.ai](mailto:privacy@openathena.ai).
Copyright questions and complaints: [privacy@openathena.ai](mailto:privacy@openathena.ai).
[CONFIRM C01: proposed contact; confirm routing and monitoring before publication.]

The [training-content summary](CONFIRM_SUMMARY_URL) identifies the versions
it covers and describes their training content and relevant data-processing
measures. This checkpoint is covered by [CONFIRM C04: summary version and
explicit checkpoint identifier]. The [copyright policy](CONFIRM_POLICY_URL)
describes source review, rights-reservation handling and the complaint process.

Copyright-infringing uses of this model are prohibited by applicable law.
The model license does not grant rights in third-party works or guarantee
that generated material can be reproduced or distributed lawfully.
```

Use immutable, approved public summary/policy URLs, or bundle the documents in each checkpoint branch and use relative links. Do not use a relative link that resolves only on HF `main`. Link the official website copy from the HF summary. Remove the `CONFIRM` markers only after the referenced version actually covers the checkpoint.

Do not copy the step-378000 compute estimate into an earlier checkpoint card as that checkpoint's own total. The optional compute paragraph in the proposed main card is explicitly scoped to the latest retained checkpoint. No compute claim is needed in each card to complete the training-content template.

The existing inference section remains candid about parity and serving limitations. Under C09, the release owner must add sufficient architecture/use instructions for the proposed open-source route, with honest validation limits. Do not replace those limits with an untested serving command or a claim of generic Transformers support.

The live uploader expects frozen card bytes. Apply approved metadata changes only after coordinating with its verification workflow; preserve weight commits and record the subsequent metadata revision. No frozen plan or live uploader card has been edited for this draft.
