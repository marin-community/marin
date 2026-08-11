# Twentieth H100 contract/map evidence launch failure

This directory seals the first terminal failure from the reviewed v20 launch.
It is negative evidence only; no 24-record evidence bundle was accepted.

- Source: `d45ea8fefb7ad73677136b7e6d53d711a46e34da`
- Source tree: `c868e635fd8f571f639077b9d78f200fc33df0d1`
- Job: `/dlwh/shuttle-h100-contract-map-evidence-d45ea8f-v20`
- Image: `ghcr.io/marin-community/iris-task-h100-evidence:5238c39e7a506b919fc803046de4a9dc2c29d02f@sha256:2efa83fbf8f2073a4175eef919a9f0c6c2db435d7c1ec9ed79017e1ea0d10cef`
- Task duration: 3 minutes 34.61 seconds
- Result: one failed task, exit 1, zero preemptions, zero failure retries

The launcher and manifest authenticated. The generated candidates, numerical
gates, Nsight Systems schedule, persistent-cache protocol, NCU profile, wide
CSV units parsing, and public SASS export completed far enough to enter the
closed SASS parser. The reviewed line-1 top-level separator and line-2
fixed-width kernel identity row passed. The next exact 107-byte separator was
rejected at line 3 because the parser had no reviewed identity-table close
state.

The traceback proves only the exact separator recognized at line 3 after the
accepted line-2 identity row. It does not expose or establish the syntax of
line 4 or any later record, including the later address-source table. The
remote temporary NCU report and SASS file were not durably exported. No GPU
relaunch was performed.
