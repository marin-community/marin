# Thirty-third H100 contract-map evidence launch failure

This directory seals the first and only terminal result from the reviewed v33
launch. It is negative evidence only; no 24-record evidence bundle was
accepted.

- Source: `5370e6cc72bbef392234444214bc471452b7af31`
- Source tree: `cc825d677ae234bbe2ef6ab5b8afbb328f00a47a`
- Job: `/dlwh/shuttle-h100-contract-map-evidence-5370e6c-v33`
- Task attempt: `0`, UID `5408d06ad61c1ee3`
- Image: `ghcr.io/marin-community/iris-task-h100-evidence:5238c39e7a506b919fc803046de4a9dc2c29d02f@sha256:2efa83fbf8f2073a4175eef919a9f0c6c2db435d7c1ec9ed79017e1ea0d10cef`
- Task-attempt duration: 196.363 seconds
- Result: one failed task, exit 1, zero preemptions, zero failure retries

The authenticated source reached the closed public SASS parser after generated
candidate compilation, numerical gates, Nsight Systems scheduling, persistent
cache validation, NCU profiling, and the wide CSV units parser. Control flow
proves that lines 1 through 9 passed in order: the selected 107-byte separator,
matching kernel identity row, identity-table close, selected-width header, and
five opaque metric-label rows. The first rejection was line 10 while the parser
still required the exact selected separator before admitting instructions.

The bounded exception records line 10 as 107 UTF-8 bytes with SHA-256
`ee1d0a272e963007f2624140f480735ffdbcdaf53ab557260250b0db921ba0f9`.
The selected widths are `(18, 60, 6, 6, 6, 6)` with five exact ASCII-space
gaps. The existing public-pattern classifier reports `instruction=true`, but
that classifier is diagnostic only and does not establish that the record is a
valid instruction in the current parser state. The parser rejected the record
before opcode allowlist or kernel-coverage validation.

The observation retains only bounded aggregate and fixed-column counts. It
contains no raw or redacted line, adjacent records, paths, or environment. The
remote temporary NCU report and SASS file were not durably exported. The
retained evidence does not establish the raw line syntax, whether a table-close
separator was omitted, or the grammar of any later record.

The locally audited pre-submit Iris bundle contained exactly the capsule,
manifest, and launcher. Its SHA-256 independently matches the controller bundle
ID.

This was one submission with `max_retries=0`. No retry, relaunch, source edit,
or post-failure GPU action was performed.
