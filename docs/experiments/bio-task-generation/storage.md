# Storage and public publication

[Planning overview](index.md) · [Validation](validation.md)

The project is public by default. Publish documentation, prompts, generation code, eligible inputs, references, graders and validation evidence. The proposed storage arrangement is below; the release account, repository name and exact artifact layout are not yet selected.

## Storage roles

| Artifact | Proposed home |
| --- | --- |
| Planning docs, prompts, schemas, generation code and small manifests | The public Marin GitHub repository |
| Released Harbor task packages and biological input assets | A public Hugging Face dataset repository, pinned by commit for each release |
| Reference solutions, grading assets and release validation summaries | Public versioned artifacts alongside the release, separated from solver input assembly |
| Temporary downloads, build intermediates and detailed validation outputs | Object storage near the authoring compute, with a retention policy; regional GCS for TRC work |
| Public example and coverage pages | Version-controlled static HTML, previewable from GitHub; links to the exact release and evidence |

Hugging Face is the proposed distribution surface because it provides versioned dataset repositories for large artifacts. Its [storage documentation](https://huggingface.co/docs/hub/en/storage-limits) describes repository structure and account quotas; public storage should not be assumed unlimited. Confirm capacity before large uploads. [Storage Buckets](https://huggingface.co/docs/hub/en/storage-buckets) are another option for working assets, but this plan does not require both buckets and dataset repositories.

Transient object storage is not the sole location of release evidence. Promote accepted task artifacts and the evidence needed to reproduce them into the public release. Keep large biological data out of the GitHub code repository. Do not repeatedly copy raw archives when a task needs only a scientifically valid subset or upstream-produced matrix.

## Release manifest and loading

For each release, record the generation-code commit, task IDs and versions, source accessions, transformations, environment references, asset locations, byte sizes, content hashes, licenses and validation records. Pin Hugging Face downloads by immutable repository revision rather than a moving branch. Preserve revisions used by released manifests.

The loader should select a task, stage only its required assets, verify their hashes and assemble the Harbor package before solving starts. Shared assets may be cached by content hash; task loading must not require downloading the whole corpus. The exact layout must be tested with the supported Harbor loader before publication.

## Public artifacts and solver isolation

Public availability and solver visibility are different properties. Oracles, expected results and grader code may be public for audit and reuse while remaining absent from solver-visible inputs. Assemble the sandbox from an explicit allowlist; do not mount the entire authoring repository or release archive. Execute grading in a trusted environment that solver writes cannot change.

The default offline task environment prevents live retrieval of public reference files during a solve. Before releasing a network-enabled task, demonstrate controls that preserve its input boundary and prevent retrieval of its public oracle or expected answers; otherwise retain offline staging or defer it. Data splitting and downstream evaluation design remain outside this project.

## Release eligibility

Record source terms and establish permission to redistribute each included input and supporting asset. Check applicable licenses, access conditions and any stated consent or use restrictions. A public download alone is insufficient evidence. Prefer sources compatible with a public corpus; hold or reject candidates with unresolved eligibility.

Public-by-default does not include credentials or third-party private communications. Publish the reusable design and evidence without copying internal service details. Do not silently move existing restricted archives into public storage; construct and validate the intended public release explicitly.
