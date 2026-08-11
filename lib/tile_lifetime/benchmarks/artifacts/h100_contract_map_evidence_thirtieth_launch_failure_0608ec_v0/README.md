# Thirtieth H100 contract-map evidence launch failure

This directory seals the first and only terminal result from the reviewed v30
launch. It is negative evidence only; no 24-record evidence bundle was
accepted.

- Source: `0608ecc5b7b3949b5938aabb26c06c75bcdb75d4`
- Source tree: `13775536f62ac32f32d4c9559694a90a6d44c877`
- Job: `/dlwh/shuttle-h100-contract-map-evidence-0608ecc-v30`
- Task attempt: `0`, UID `4943120b8681d6a2`
- Image: `ghcr.io/marin-community/iris-task-h100-evidence:5238c39e7a506b919fc803046de4a9dc2c29d02f@sha256:2efa83fbf8f2073a4175eef919a9f0c6c2db435d7c1ec9ed79017e1ea0d10cef`
- Task-attempt duration: 222.731 seconds
- Result: one failed task, exit 1, zero preemptions, zero failure retries

The authenticated source reached the closed public SASS parser after the
generated candidates, numerical gates, Nsight Systems schedule,
persistent-cache protocol, NCU profile, and wide CSV units parser. Control flow
proves that the selected 108-byte separator, matching kernel identity row,
identity-table close, selected-width fixed-column header, and first opaque
metric-label row at lines 1 through 5 were accepted. The first rejection was a
second fixed-column row at line 6, where the source required the selected
separator immediately after the first metric-label row.

The bounded exception records line 6 as 108 UTF-8 bytes with SHA-256
`bd5467cdab75c1bcff12fec58aaf5ecf91886263faee8e0ac46f21e9f21565b5`.
The selected widths are `(18, 61, 6, 6, 6, 6)` with five exact ASCII-space
gaps. Columns 0 and 1 trim to empty. Columns 2 through 5 each contain one
six-byte ASCII token with five lowercase letters and exactly one underscore;
every other punctuation count is zero. The complete row has 81 leading spaces,
84 spaces total, four tokens, twenty lowercase letters, and four underscores.
It has no trailing spaces, tabs, controls, digits, uppercase letters, non-ASCII
code points, colon, comma, hyphen, or pipe. No public vocabulary or reviewed
public row pattern matched.

The observation does not disclose the four private letter sequences or the
character order inside their six-byte tokens. It contains no raw row, adjacent
records, paths, or environment. The remote temporary NCU report and SASS file
were not durably exported. The retained evidence proves only that line 6 is a
second row of this aggregate form; it does not establish how many opaque rows
follow or the syntax of any later record.

The locally audited pre-submit Iris bundle contained exactly the capsule,
manifest, and launcher. Its content hash independently matches the controller's
bundle ID.

This was one submission with `max_retries=0`. No retry, relaunch, source edit,
or post-failure GPU action was performed.
