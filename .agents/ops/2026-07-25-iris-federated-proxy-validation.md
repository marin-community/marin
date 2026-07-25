---
date: 2026-07-25
system: iris
severity: diagnostic-only
resolution: fixed
pr: none
issue: https://github.com/marin-community/marin/issues/7607
---

## TL;DR

- The validation targeted a root CPU job federated from `marin-dev` to
  `cw-us-west-04a`; that CPU coordinator then launched a one-H100 inference
  child and minted a capability URL.
- Native proxy `0.1.2` was published from commit `d7a848d7d7`, and the Iris
  dependency floor was raised so CPU and H100 tasks both installed that version.
- Both controllers were rolled to `2117dded8f`; `marin-dev` then advertised the
  reachable `cw-us-west-04a` peer and its eight available H100s.
- The first complete request reached the H100 only after the `marin-dev` GCLB
  token route learned the cluster-tagged capability shape. The prior
  `/proxy/t/*` rule sent `/proxy/<cluster>/t/*` through IAP and returned a Google
  sign-in page to headless tasks.
- Final job `/power/federated-inference-proxy-demo-20260725-v3` succeeded in
  62.75 seconds. Its H100 child served `/v1/models` with HTTP 200 and was
  terminated by the coordinator cleanup with zero failures.

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
6. An annotated `iris-native-v0.1.2` tag was pushed at `d7a848d7d7`. Release run
   `30142227827` published the stable wheel, and a frozen workspace sync
   installed it successfully.
7. The first `marin-dev` controller image build exhausted the VM disk before
   cutover. Inactive Docker build cache and unused images were removed; running
   containers, volumes, and repository data were preserved. The retry left
   about 64 GB free and completed normally.
8. `marin-dev` was rolled before `cw-us-west-04a`. Both controllers reported
   version `2117dded8f (weaver/roll-marin-dev-marin-ci-clusters) (power)`, and the
   parent reported the west peer reachable with `cpu-erapids` and `h100-8x`.
9. Diagnostic jobs `federated-inference-proxy-demo-20260725` and
   `federated-inference-proxy-demo-20260725-v2` both launched vLLM on an H100 and
   minted the expected `https://iris-dev.oa.dev/proxy/cw-us-west-04a/t/...` URL.
   The HTTP client then followed IAP redirects to `accounts.google.com`; the
   second run captured two redirects and a 913,481-byte `text/html` response.
10. The live GCLB URL map exposed only `/proxy/t` and `/proxy/t/*` on its
    IAP-free backend. GCLB simple path rules allow `*` only as a trailing
    wildcard, so they cannot express a single child-cluster segment.
11. The token route was migrated to path-template matches `/proxy/t/**` and
    `/proxy/*/t/**`. URL-map tests assert both public capability shapes plus the
    IAP-gated `/proxy/system.log-server/` boundary before every import.
12. After propagation, fake local and federated capabilities bypassed IAP and
    reached the native controller (`401` and `502`, respectively), while the
    private log endpoint retained its IAP-generated `302`.
13. The final job launched root CPU task `0` and H100 child
    `inference-3b9371de5be64cb8ab5786ee47d393fa`. The H100 access log recorded
    `35.254.13.19 ... GET /v1/models ... 200 OK`; the root recorded a redacted
    cluster-tagged capability URL, the served model ID, and status 200.

## User course corrections

- The user explicitly authorized controller rolls, cluster configuration
  changes, and a native release tag. Those permissions made it possible to
  validate the exact merged source without waiting for a separate operator.

## Root cause

Three integration gaps prevented the merged native relay from working in the
requested path:

1. Commit `d7a848d7d7` changed the Rust package version to `0.1.2`, but no stable
   tag had invoked the native-wheel release workflow and Iris still accepted
   native `0.1.1`.
2. The inference orchestrator rebuilt the capability URL from the requested
   origin instead of consuming the cluster-tagged URL returned by the child
   controller.
3. The GCP token-proxy stage opened only local `/proxy/t/*` capabilities past
   IAP. The new `/proxy/<cluster>/t/*` URL therefore hit the default IAP backend
   before the native parent relay could validate and forward it.

## Fix

- Published `marin-iris-native==0.1.2` and required it from Iris.
- Changed the inference orchestrator to consume
  `MintEndpointTokenResponse.capability_url`.
- Added the reciprocal `marin-dev` / `cw-us-west-04a` federation trust and peer
  configuration.
- Added `experiments/evals/federated_inference_proxy_demo.py`, which asserts the
  cluster-tagged URL and consumes `/v1/models`.
- Migrated the GCLB token proxy to path-template route rules for local and
  federated capabilities, with route-boundary tests and pre-import validation.

## How OPS.md could have shortened this

- `lib/iris/OPS.md` "Controller Restart" could state that a merged Rust proxy
  change requires both a stable `iris-native-v*` publication and a
  `marin-iris-native` dependency-floor bump for wheel consumers, while
  controller image builds compile the local Rust source directly.
- `lib/iris/OPS.md` "Job Management" could link the federation observation
  commands in `lib/iris/docs/federation.md` and note that parent-first rollout
  is required before a child starts minting cluster-tagged capability URLs.
- `lib/iris/docs/iap-gclb.md` now documents the cluster-tagged capability route,
  its one-segment peer matcher, and the private-path boundary. Keeping executable
  URL-map tests in the deployment script prevents the same edge mismatch from
  recurring.

## Artifacts

- `.agents/projects/iris_federation/2026-07-25_federation_blind_relay_cluster_tag.md`
- https://github.com/marin-community/marin/actions/runs/30142227827
- https://iris-dev.oa.dev/#/job/%2Fpower%2Ffederated-inference-proxy-demo-20260725-v3
- `/power/federated-inference-proxy-demo-20260725-v3/inference-3b9371de5be64cb8ab5786ee47d393fa`
