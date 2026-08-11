# Twenty-second H100 contract-map evidence launch failure

This directory seals the first and only terminal result from the reviewed v22
launch. It is negative evidence only; no 24-record evidence bundle was
accepted.

- Source: `75921c162d31642c14cfa6101421295c9030ec3a`
- Source tree: `84f66573aa39bf161448d8283cecaffaaf28d864`
- Job: `/dlwh/shuttle-h100-contract-map-evidence-75921c1-v22`
- Task attempt: `0`, UID `ea8274661f7db9b3`
- Image: `ghcr.io/marin-community/iris-task-h100-evidence:5238c39e7a506b919fc803046de4a9dc2c29d02f@sha256:2efa83fbf8f2073a4175eef919a9f0c6c2db435d7c1ec9ed79017e1ea0d10cef`
- Task duration: 287.032 seconds
- Result: one failed task, exit 1, zero preemptions, zero failure retries

The launcher and manifest authenticated. The task reached the NCU SASS parser,
which rejected the export before section parsing because line 1 was not the
exact reviewed 107-byte top-level separator. The exception contains no line-1
length, hash, structure, or text, so this artifact does not infer any of those
facts. In particular, v22 did not reach the new fixed-column diagnostic.

The locally audited pre-submit Iris bundle contained exactly the capsule,
manifest, and launcher. Its hash is recorded as a pre-submit identity only;
the controller did not expose an independently retained bundle identifier in
the terminal evidence collected here.

This was one submission with `max_retries=0`. No retry, relaunch, source edit,
or post-failure GPU action was performed.
