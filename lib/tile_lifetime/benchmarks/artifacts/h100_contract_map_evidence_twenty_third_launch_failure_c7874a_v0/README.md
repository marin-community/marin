# Twenty-third H100 contract-map evidence launch failure

This directory seals the first and only terminal result from the reviewed v23
diagnostic launch. It is negative evidence only; no 24-record evidence bundle
was accepted.

- Source: `c7874a8fa8194772dfbe55827d9c2e8be0a14154`
- Source tree: `175cd6b8910d917f20d8678e7e06809f519f5f38`
- Job: `/dlwh/shuttle-h100-contract-map-evidence-c7874a8-v23`
- Task attempt: `0`, UID `3ccbb33dfe5201b1`
- Image: `ghcr.io/marin-community/iris-task-h100-evidence:5238c39e7a506b919fc803046de4a9dc2c29d02f@sha256:2efa83fbf8f2073a4175eef919a9f0c6c2db435d7c1ec9ed79017e1ea0d10cef`
- Task duration: 233.084 seconds
- Result: one failed task, exit 1, zero preemptions, zero failure retries

The launcher and manifest authenticated. The generated candidates, numerical
gates, Nsight Systems schedule, persistent-cache protocol, NCU profile, wide
CSV units parsing, and public SASS export completed far enough to enter the
closed SASS parser. The first rejection was the line-1 separator gate.

The bounded exception records only the line number, UTF-8 byte count, SHA-256,
and reviewed aggregate structural fields. It proves that line 1 is 108 UTF-8
bytes containing 103 hyphens and five ASCII spaces across six non-whitespace
tokens, with a maximum token width of 61 bytes. It does not prove the token
order, every individual token width, raw text, or any adjacent record. The
remote temporary NCU report and SASS file were not durably exported.

The locally audited pre-submit Iris bundle contained exactly the capsule,
manifest, and launcher. Its hash is recorded as a pre-submit identity only;
the controller did not expose an independently retained bundle identifier in
the terminal evidence collected here.

This was one submission with `max_retries=0`. No retry, relaunch, source edit,
or post-failure GPU action was performed.
