# Sixteenth H100 contract/map evidence launch failure

This directory seals the first terminal failure from the reviewed v16 launch.
It is negative evidence only; no 24-record evidence bundle was accepted.

- Source: `69221a950e251e5a3db5d6a46c670e0c9498323c`
- Source tree: `bf8c1c944fd601f78504a6da8227ea423d199c56`
- Job: `/dlwh/shuttle-h100-contract-map-evidence-69221a9-v16`
- Image: `ghcr.io/marin-community/iris-task-h100-evidence:5238c39e7a506b919fc803046de4a9dc2c29d02f@sha256:2efa83fbf8f2073a4175eef919a9f0c6c2db435d7c1ec9ed79017e1ea0d10cef`
- Task duration: 3 minutes 31.15 seconds
- Result: one failed task, exit 1, zero preemptions, zero failure retries

The launcher and manifest authenticated. The generated candidates, numerical
gates, Nsight Systems schedule, persistent-cache protocol, NCU profile, wide
CSV units parsing, and public SASS export completed far enough to enter the
closed SASS parser. The first rejection was the exact public table separator
recorded in `sass-observation.json`.

The remote temporary NCU report and SASS file were not durably exported. This
artifact therefore retains only the bounded exception and exact observed line;
it does not infer unavailable surrounding rows. No GPU relaunch was performed.
