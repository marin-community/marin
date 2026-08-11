# Thirty-first H100 contract-map evidence launch failure

This directory seals the first and only terminal result from the reviewed v31
launch. It is negative evidence only; no 24-record evidence bundle was
accepted.

- Source: `f6a0a9143c3238e1883875ffbbb944d641874738`
- Source tree: `bacdc71fe006caf303c6948350733ef43ef95740`
- Job: `/dlwh/shuttle-h100-contract-map-evidence-f6a0a91-v31`
- Task attempt: `0`, UID `92701bb116a7249d`
- Image: `ghcr.io/marin-community/iris-task-h100-evidence:5238c39e7a506b919fc803046de4a9dc2c29d02f@sha256:2efa83fbf8f2073a4175eef919a9f0c6c2db435d7c1ec9ed79017e1ea0d10cef`
- Task-attempt duration: 252.313 seconds
- Result: one failed task, exit 1, zero preemptions, zero failure retries

The authenticated source reached the closed public SASS parser after the
generated candidates, numerical gates, Nsight Systems schedule,
persistent-cache protocol, NCU profile, and wide CSV units parser. Control flow
proves that the selected 107-byte separator, matching kernel identity row,
identity-table close, selected-width fixed-column header, and two opaque
metric-label rows at lines 1 through 6 were accepted. The first rejection was a
third fixed-column row at line 7 because the reviewed v31 grammar required all
four metric columns to contain a six-byte fragment.

The bounded exception records line 7 as 107 UTF-8 bytes with SHA-256
`0acf65e2299cf8564ded138ad6b949bfaf8a332b0b54f432de0c9ba6510be12b`.
The selected widths are `(18, 60, 6, 6, 6, 6)` with five exact ASCII-space
gaps. Columns 0 through 2 trim to empty. Column 3 is a six-byte ASCII token with
five lowercase letters and one underscore. Columns 4 and 5 are each six-byte
ASCII tokens with four lowercase letters and two underscores. Every other
punctuation count is zero. The complete row has 87 leading spaces, 89 spaces
total, three tokens, thirteen lowercase letters, and five underscores. It has
no trailing spaces, tabs, controls, digits, uppercase letters, non-ASCII code
points, colon, comma, hyphen, or pipe. No public vocabulary or reviewed public
row pattern matched.

The observation does not disclose the three private letter sequences or the
character order inside their six-byte tokens. It contains no raw row, adjacent
records, paths, or environment. The remote temporary NCU report and SASS file
were not durably exported. The retained evidence proves only that line 7 has
this aggregate form; it does not establish how many opaque rows follow or the
syntax of any later record.

The locally audited pre-submit Iris bundle contained exactly the capsule,
manifest, and launcher. Its content hash independently matches the controller's
bundle ID.

This was one submission with `max_retries=0`. No retry, relaunch, source edit,
or post-failure GPU action was performed.
