# Twenty-eighth H100 contract-map evidence launch failure

This directory seals the first and only terminal result from the reviewed v28
diagnostic launch. It is negative evidence only; no 24-record evidence bundle
was accepted.

- Source: `65ac4bc1bfe724f1a3ce3f779a42c387d5ba57a4`
- Source tree: `4a79b9e8fa8f3cea6ce4a4cdee7b72e2e43af143`
- Job: `/dlwh/shuttle-h100-contract-map-evidence-65ac4bc-v28`
- Task attempt: `0`, UID `04089394fe2ee37b`
- Image: `ghcr.io/marin-community/iris-task-h100-evidence:5238c39e7a506b919fc803046de4a9dc2c29d02f@sha256:2efa83fbf8f2073a4175eef919a9f0c6c2db435d7c1ec9ed79017e1ea0d10cef`
- Task duration: 222.930 seconds
- Result: one failed task, exit 1, zero preemptions, zero failure retries

The authenticated source reached the closed public SASS parser after the
generated candidates, numerical gates, Nsight Systems schedule,
persistent-cache protocol, NCU profile, and wide CSV units parser. Control flow
proves that the selected 108-byte separator, matching kernel identity row,
identity-table close, and selected-width fixed-column header at lines 1 through
4 were accepted. The first rejection was the exact selected-width address-table
close requirement at line 5.

The bounded exception records line 5 as 108 UTF-8 bytes with SHA-256
`4ea99cffab873479edd9e63f9506961348c16a64bfa058dc263843795432b928`.
The selected widths are `(18, 61, 6, 6, 6, 6)` with five exact ASCII-space
gaps. Columns 0 and 1 trim to empty. Columns 2 through 5 each contain one
six-byte ASCII token with four lowercase letters and two punctuation bytes.
The complete row has 81 leading spaces, 84 spaces total, four tokens, sixteen
lowercase letters, and eight punctuation bytes. It has no tabs, controls,
digits, uppercase letters, non-ASCII code points, colon, comma, hyphen, or pipe.
No public vocabulary or reviewed public row pattern matched.

The observation does not disclose the punctuation values, private token text
or order, raw row, adjacent records, paths, or environment. The remote temporary
NCU report and SASS file were not durably exported.

The locally audited pre-submit Iris bundle contained exactly the capsule,
manifest, and launcher. Its hash is recorded as a pre-submit identity and also
matches the controller's bundle identifier.

This was one submission with `max_retries=0`. No retry, relaunch, source edit,
or post-failure GPU action was performed.
