# Seventeenth H100 contract/map evidence launch failure

This directory seals the first terminal failure from the reviewed v17 launch.
It is negative evidence only; no 24-record evidence bundle was accepted.

- Source: `87183412f719f78287568eef9d9feb0cd1bcaf0c`
- Source tree: `743e06ad8a62cb050df643e25b229bd8c80fd6f7`
- Job: `/dlwh/shuttle-h100-contract-map-evidence-8718341-v17`
- Image: `ghcr.io/marin-community/iris-task-h100-evidence:5238c39e7a506b919fc803046de4a9dc2c29d02f@sha256:2efa83fbf8f2073a4175eef919a9f0c6c2db435d7c1ec9ed79017e1ea0d10cef`
- Task duration: 3 minutes 35.09 seconds
- Result: one failed task, exit 1, zero preemptions, zero failure retries

The launcher and manifest authenticated. The generated candidates, numerical
gates, Nsight Systems schedule, persistent-cache protocol, NCU profile, wide
CSV units parsing, and public SASS export completed far enough to enter the
closed SASS parser. The first rejection identifies the exact 107-byte public
table separator at line 1, before any accepted kernel section.

The remote temporary NCU report and SASS file were not durably exported. This
artifact therefore proves only the exact line-1 record and bounded exception.
It does not prove the surrounding or later SASS structure, including whether
later per-kernel separators are present. No GPU relaunch was performed.
