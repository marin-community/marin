# Twenty-sixth H100 contract-map evidence launch failure

This directory seals the first and only terminal result from the reviewed v26
diagnostic launch. It is negative evidence only; no 24-record evidence bundle
was accepted.

- Source: `5a4518700a232c302f0db6029e2a045a04bea174`
- Source tree: `d9e4cca003d00198d110dacfae040930de118c89`
- Job: `/dlwh/shuttle-h100-contract-map-evidence-5a45187-v26`
- Task attempt: `0`, UID `30ef3ddfb095adc8`
- Image: `ghcr.io/marin-community/iris-task-h100-evidence:5238c39e7a506b919fc803046de4a9dc2c29d02f@sha256:2efa83fbf8f2073a4175eef919a9f0c6c2db435d7c1ec9ed79017e1ea0d10cef`
- Task duration: 169.980 seconds
- Result: one failed task, exit 1, zero preemptions, zero failure retries

The launcher and manifest authenticated. The generated candidates, numerical
gates, Nsight Systems schedule, persistent-cache protocol, NCU profile, wide
CSV units parsing, and public SASS export completed far enough to enter the
closed SASS parser. Control flow proves that the selected 107-byte line-1
separator, selected-width kernel identity row at line 2, and matching
identity-table close at line 3 were accepted. The first rejection was line 4.

The bounded exception records the line number, UTF-8 byte count, SHA-256,
aggregate structural fields, and the selected-width fixed-column aggregate. It
proves that line 4 uses the reviewed `(18, 60, 6, 6, 6, 6)` widths and five
single-ASCII-space gaps. Column 0 has one seven-byte token and the exact public
word `Address`; column 1 has one six-byte token and the exact public word
`Source`; columns 2 through 5 each contain one six-byte lowercase-ASCII token.
It does not disclose the four private token values, raw row, adjacent records,
paths, or environment. The remote temporary NCU report and SASS file were not
durably exported.

The locally audited pre-submit Iris bundle contained exactly the capsule,
manifest, and launcher. Its hash is recorded as a pre-submit identity only;
the controller did not expose an independently retained bundle identifier in
the terminal evidence collected here.

This was one submission with `max_retries=0`. No retry, relaunch, source edit,
or post-failure GPU action was performed.
