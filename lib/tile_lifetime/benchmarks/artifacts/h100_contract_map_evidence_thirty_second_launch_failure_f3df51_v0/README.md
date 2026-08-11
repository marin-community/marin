# Thirty-second H100 contract-map evidence launch failure

This directory seals the first and only terminal result from the reviewed v32
launch. It is negative evidence only; no 24-record evidence bundle was
accepted.

- Source: `f3df518bece548d75227365e9b502415c666fb49`
- Source tree: `ee913b7e4c0529973df512a2fc46efc85d3c1139`
- Job: `/dlwh/shuttle-h100-contract-map-evidence-f3df518-v32`
- Task attempt: `0`, UID `1a16cb8e38ecc713`
- Image: `ghcr.io/marin-community/iris-task-h100-evidence:5238c39e7a506b919fc803046de4a9dc2c29d02f@sha256:2efa83fbf8f2073a4175eef919a9f0c6c2db435d7c1ec9ed79017e1ea0d10cef`
- Task-attempt duration: 174.242 seconds
- Result: one failed task, exit 1, zero preemptions, zero failure retries

The authenticated source reached the closed public SASS parser after the
generated candidates, numerical gates, Nsight Systems schedule,
persistent-cache protocol, NCU profile, and wide CSV units parser. Control flow
proves that the selected 107-byte separator, matching kernel identity row,
identity-table close, selected-width fixed-column header, and three opaque
metric-label rows at lines 1 through 7 were accepted. The first rejection was a
fourth fixed-column row at line 8 because the reviewed v32 grammar required
every nonempty metric fragment to occupy all six bytes in its cell.

The bounded exception records line 8 as 107 UTF-8 bytes with SHA-256
`2900a58a2512cf49de9cf7041b4ffbeda07220f4f37527d506530804d7eac808`.
The selected widths are `(18, 60, 6, 6, 6, 6)` with five exact ASCII-space
gaps. Columns 0 through 2 trim to empty. Column 3 trims to one four-byte ASCII
token containing four lowercase letters; its six-byte cell therefore has two
additional ASCII spaces. Columns 4 and 5 are each six-byte ASCII tokens with
five lowercase letters and one underscore. Every other punctuation count is
zero. The complete row has 87 leading spaces, 91 spaces total, three tokens,
fourteen lowercase letters, and two underscores. It has no trailing spaces,
tabs, controls, digits, uppercase letters, non-ASCII code points, colon, comma,
hyphen, or pipe. No public vocabulary or reviewed public row pattern matched.

The observation does not disclose the three private letter sequences or the
character order inside the two underscored tokens. It contains no raw row,
adjacent records, paths, or environment. The remote temporary NCU report and
SASS file were not durably exported. The retained evidence proves only that
line 8 has this aggregate form; it does not establish how many opaque rows
follow or the syntax of any later record.

The locally audited pre-submit Iris bundle contained exactly the capsule,
manifest, and launcher. Its content hash independently matches the controller's
bundle ID.

This was one submission with `max_retries=0`. No retry, relaunch, source edit,
or post-failure GPU action was performed.
