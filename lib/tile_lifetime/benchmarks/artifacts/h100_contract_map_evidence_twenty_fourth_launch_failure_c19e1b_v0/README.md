# Twenty-fourth H100 contract-map evidence launch failure

This directory seals the first and only terminal result from the reviewed v24
diagnostic launch. It is negative evidence only; no 24-record evidence bundle
was accepted.

- Source: `c19e1bb8582da83519f746da67de88e0ca55f494`
- Source tree: `0b577367dafc23a2e351c6472d60a692c1d901c8`
- Job: `/dlwh/shuttle-h100-contract-map-evidence-c19e1bb-v24`
- Task attempt: `0`, UID `6bcc3d7736184207`
- Image: `ghcr.io/marin-community/iris-task-h100-evidence:5238c39e7a506b919fc803046de4a9dc2c29d02f@sha256:2efa83fbf8f2073a4175eef919a9f0c6c2db435d7c1ec9ed79017e1ea0d10cef`
- Task duration: 237.651 seconds
- Result: one failed task, exit 1, zero preemptions, zero failure retries

The launcher and manifest authenticated. The generated candidates, numerical
gates, Nsight Systems schedule, persistent-cache protocol, NCU profile, wide
CSV units parsing, and public SASS export completed far enough to enter the
closed SASS parser. The reviewed 108-byte line-1 separator was accepted. The
first rejection was line 2, where the fixed-width kernel identity row remained
restricted to the reviewed 107-byte form.

The bounded exception records only the line number, UTF-8 byte count, SHA-256,
and reviewed aggregate structural fields. It proves that line 2 is 108 UTF-8
bytes containing three whitespace-delimited tokens, 72 ASCII spaces, 63
trailing spaces, and the exact public vocabulary tokens `Kernel` and `Name`.
It does not prove the raw identifier, raw row text, spacing before the trailing
padding, or any adjacent record. The remote temporary NCU report and SASS file
were not durably exported.

The locally audited pre-submit Iris bundle contained exactly the capsule,
manifest, and launcher. Its hash is recorded as a pre-submit identity only;
the controller did not expose an independently retained bundle identifier in
the terminal evidence collected here.

This was one submission with `max_retries=0`. No retry, relaunch, source edit,
or post-failure GPU action was performed.
