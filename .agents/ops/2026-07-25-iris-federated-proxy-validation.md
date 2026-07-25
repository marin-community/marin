---
date: 2026-07-25
system: iris
severity: diagnostic-only
resolution: investigating
pr: none
issue: https://github.com/marin-community/marin/issues/7607
---

## TL;DR

- The validation targeted a root CPU job federated from `marin-dev` to
  `cw-us-west-04a`; that CPU coordinator then launched a one-H100 inference
  child and minted a capability URL.
- Both controllers were healthy before the rollout. `marin-dev` ran
  `e23bdaf7fb`; `cw-us-west-04a` reported `15e6016`.
- The CoreWeave TLS ingress already existed and its network policy admitted
  `marin-dev`'s fixed egress address, `35.254.13.19`.
- Commit `d7a848d7d7` declared native proxy `0.1.2`, but it neither published
  the stable wheel nor raised `marin-iris`'s `>=0.1.1` dependency floor.
- The merged design required the inference orchestrator to consume
  `MintEndpointTokenResponse.capability_url`, but
  `lib/marin/src/marin/inference/iris.py` still rebuilt the URL from the caller
  origin and child token.

## Original problem report

Validate the native federation proxy forwarding released in
`d7a848d7d72dd243acf319bd9b4b8a4959b9297e`. The regression needed to start a
CPU task on a federated cluster, launch an H100 inference child from that task,
and mint a URL that remained usable through the public parent. Federation
between `marin-dev` and `cw-us-west-04a` could be enabled for the test.

## Investigation path

1. The release workflow and PyPI state were checked first because the native
   request parser owns the `/proxy/<cluster>/t/...` route. Pull-request runs
   only built artifacts; stable publication required an `iris-native-v*` tag.
2. `lib/iris/rust/pyproject.toml` declared `0.1.2`, while
   `lib/iris/pyproject.toml` still accepted native `0.1.1`. No
   `iris-native-v0.1.2` tag or PyPI release existed.
3. Both cluster configs and live controller status were inspected before
   mutation. Neither side declared the new peer. The CoreWeave controller had
   no federation peers.
4. The `iris-federation` ingress for `iris-cw-us-west-04a.oa.dev` was already
   serving TLS. An off-allowlist workstation received HTTP 403, as expected.
   The cluster-wide allowlist source included the reserved `marin-dev` egress.
5. The accepted design in
   `.agents/projects/iris_federation/2026-07-25_federation_blind_relay_cluster_tag.md`
   was compared with the merged diff. Its requirement that the eval
   orchestrator use `resp.capability_url` was absent from the implementation.

## User course corrections

- The user explicitly authorized controller rolls, cluster configuration
  changes, and a native release tag. Those permissions made it possible to
  validate the exact merged source without waiting for a separate operator.

## Root cause

Investigation was still in progress. Two release gaps were established before
the live test: native proxy `0.1.2` was not published or required, and the Marin
inference orchestrator ignored the controller-built federated capability URL.

## Fix

The pending branch raised the native dependency floor, consumed
`MintEndpointTokenResponse.capability_url`, added the `marin-dev` /
`cw-us-west-04a` peer declarations, and added a live regression demo under
`experiments/evals/`.

## How OPS.md could have shortened this

- `lib/iris/OPS.md` "Controller Restart" could state that a merged Rust proxy
  change requires both a stable `iris-native-v*` publication and a
  `marin-iris-native` dependency-floor bump for wheel consumers, while
  controller image builds compile the local Rust source directly.
- `lib/iris/OPS.md` "Job Management" could link the federation observation
  commands in `lib/iris/docs/federation.md` and note that parent-first rollout
  is required before a child starts minting cluster-tagged capability URLs.

## Artifacts

- `.agents/projects/iris_federation/2026-07-25_federation_blind_relay_cluster_tag.md`
- https://github.com/marin-community/marin/actions/runs/30142227827
- https://iris-dev.oa.dev
- https://iris-cw-us-west-04a.oa.dev
