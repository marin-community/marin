# Twenty-ninth H100 contract-map evidence launch failure

This directory seals the first and only terminal result from the reviewed v29
diagnostic launch. It is negative evidence only; no 24-record evidence bundle
was accepted.

- Source: `fd0fe901919e975f1e07b3b8036fd9d1d41e9046`
- Source tree: `20da7cfa6ab7f0a684809954dc08b3d6be18cc52`
- Job: `/dlwh/shuttle-h100-contract-map-evidence-fd0fe90-v29`
- Task attempt: `0`, UID `450a1bd0fa82aaf9`
- Image: `ghcr.io/marin-community/iris-task-h100-evidence:5238c39e7a506b919fc803046de4a9dc2c29d02f@sha256:2efa83fbf8f2073a4175eef919a9f0c6c2db435d7c1ec9ed79017e1ea0d10cef`
- Task duration: 229.449 seconds
- Result: one failed task, exit 1, zero preemptions, zero failure retries

The authenticated source reached the closed public SASS parser after the
generated candidates, numerical gates, Nsight Systems schedule,
persistent-cache protocol, NCU profile, and wide CSV units parser. Control flow
proves that the selected 107-byte separator, matching kernel identity row,
identity-table close, and selected-width fixed-column header at lines 1 through
4 were accepted. The first rejection was the exact selected-width address-table
close requirement at line 5.

The bounded exception records line 5 as 107 UTF-8 bytes with SHA-256
`f4216a396b9aabb3ac8416facab0449967742e56a45aff7f7218a0b2513d50f7`.
The selected widths are `(18, 60, 6, 6, 6, 6)` with five exact ASCII-space
gaps. Columns 0 and 1 trim to empty. Columns 2 through 5 each contain one
six-byte ASCII token with four lowercase letters and exactly two underscore
characters; every other punctuation count is zero. The complete row has 80
leading spaces, 83 spaces total, four tokens, sixteen lowercase letters, and
eight underscores. It has no trailing spaces, tabs, controls, digits, uppercase
letters, non-ASCII code points, colon, comma, hyphen, or pipe. No public
vocabulary or reviewed public row pattern matched.

The observation does not disclose the four private letter sequences or the
character order inside their six-byte tokens. It contains no raw row, adjacent
records, paths, or environment. The remote temporary NCU report and SASS file
were not durably exported.

The locally audited pre-submit Iris bundle contained exactly the capsule,
manifest, and launcher. Its content hash is recorded as a pre-submit identity
and independently matches the controller's downloaded bundle.

This was one submission with `max_retries=0`. No retry, relaunch, source edit,
or post-failure GPU action was performed.
