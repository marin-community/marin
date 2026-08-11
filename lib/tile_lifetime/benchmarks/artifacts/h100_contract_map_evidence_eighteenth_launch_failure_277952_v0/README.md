# Eighteenth H100 contract/map evidence launch failure

This directory seals the first terminal failure from the reviewed v18 launch.
It is negative evidence only; no 24-record evidence bundle was accepted.

- Source: `2779523b2c42218810bffc90c047a7f90d21aa81`
- Source tree: `53d5687887f872fa05c345998ce9809761b8ec4c`
- Job: `/dlwh/shuttle-h100-contract-map-evidence-2779523-v18`
- Image: `ghcr.io/marin-community/iris-task-h100-evidence:5238c39e7a506b919fc803046de4a9dc2c29d02f@sha256:2efa83fbf8f2073a4175eef919a9f0c6c2db435d7c1ec9ed79017e1ea0d10cef`
- Task duration: 3 minutes 31.19 seconds
- Result: one failed task, exit 1, zero preemptions, zero failure retries

The launcher and manifest authenticated. The generated candidates, numerical
gates, Nsight Systems schedule, persistent-cache protocol, NCU profile, wide
CSV units parsing, and public SASS export completed far enough to enter the
closed SASS parser. The reviewed line-1 top-level separator passed. The first
rejection was an unrecognized record at line 2.

The remote temporary NCU report and SASS file were not durably exported. The
bounded exception contains no line-2 text, so this artifact does not infer its
content, byte length, identity, or surrounding structure. No GPU relaunch was
performed.
