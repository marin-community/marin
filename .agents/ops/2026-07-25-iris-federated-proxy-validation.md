---
date: 2026-07-25
system: iris
severity: diagnostic-only
resolution: fixed
pr: https://github.com/marin-community/marin/pull/7634
issue: https://github.com/marin-community/marin/issues/7607
---

## TL;DR

- The validation targeted a root CPU job federated from `marin-dev` to
  `cw-us-west-04a`; that CPU coordinator then launched a one-H100 inference
  child and minted a capability URL.
- Commit `d7a848d7d7` emitted `/proxy/<cluster>/t/...`, outside the existing
  IAP-free `/proxy/t/*` ingress. Native proxy `0.1.3` instead emits and relays
  `/proxy/t/cluster=<peer>/<token>/<endpoint>/...`.
- The live URL map was restored to its original `/proxy/t` and `/proxy/t/*`
  path rule. `/proxy/system.log-server/` still redirects to IAP.
- Both controllers were rolled to `f6812d75a7`; `marin-dev` advertised the
  reachable `cw-us-west-04a` peer, `cpu-erapids`, `h100-8x`, and eight available
  H100s.
- Final job `/power/federated-inference-proxy-demo-20260725-v4` succeeded in
  81.63 seconds. A literal curl with no headers, cookies, or authorized client
  received `/v1/models` JSON with HTTP 200 through the public parent. Cleanup
  terminated the H100 child with zero root-job failures.

## Original problem report

Validate the native federation proxy forwarding released in
`d7a848d7d72dd243acf319bd9b4b8a4959b9297e`. The regression needed to start a
CPU task on a federated cluster, launch an H100 inference child from that task,
and mint a URL that remained usable through the public parent. Federation
between `marin-dev` and `cw-us-west-04a` could be enabled for the test.

## Investigation path

1. The release workflow and PyPI state were checked first because the native
   request parser owns the public capability route. Pull-request runs
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
    IAP-free backend. The merged `/proxy/<cluster>/t/*` form therefore reached
    the default IAP backend before native Iris routing.
11. A temporary path-template rule admitted both URL forms and proved that the
    merged native relay worked: diagnostic job
    `/power/federated-inference-proxy-demo-20260725-v3` reached the H100 with
    HTTP 200.
12. Keeping the cluster discriminator beneath the existing capability prefix
    removed the load-balancer coupling. Native `0.1.3` parses
    `/proxy/t/cluster=<peer>/...`, and rigging now mints that form.
13. The URL map was restored to only `/proxy/t` and `/proxy/t/*`. Static
    validation passed before import; a fake federated route reached Iris with
    HTTP 502, while `/proxy/system.log-server/` retained its IAP-generated
    HTTP 302.
14. The final job launched root CPU task `0` and H100 child
    `inference-e876c8b2248748a0a0f9d493cdabcbfb`. Both installed
    `marin-iris-native==0.1.3`. The H100 access log recorded
    `35.254.13.19 ... GET /v1/models ... 200 OK`; the root recorded the redacted
    `/proxy/t/cluster=cw-us-west-04a/...` URL, `"client": "curl"`, and status
    200.
15. The demo URL redactor was updated for the new cluster segment before the
    final run, preventing the capability token from appearing in diagnostics.

The earlier temporary-route run launched H100 child
`inference-3b9371de5be64cb8ab5786ee47d393fa`. The H100 access log recorded
`35.254.13.19 ... GET /v1/models ... 200 OK` and established that the remaining
failure was ingress shape rather than federation or vLLM.

## User course corrections

- The user explicitly authorized controller rolls, cluster configuration
  changes, and a native release tag. Those permissions made it possible to
  validate the exact merged source without waiting for a separate operator.
- The user required the federated form to remain under `/proxy/t/*` and the
  proof to use unauthenticated curl. That moved cluster selection into the
  native path grammar and removed the temporary GCLB route expansion.

## Root cause

Three integration gaps prevented the merged native relay from working in the
requested path:

1. Commit `d7a848d7d7` changed the Rust package version to `0.1.2`, but no stable
   tag had invoked the native-wheel release workflow and Iris still accepted
   native `0.1.1`.
2. The inference orchestrator rebuilt the capability URL from the requested
   origin instead of consuming the cluster-tagged URL returned by the child
   controller.
3. The merged cluster discriminator sat before the stable capability prefix:
   `/proxy/<cluster>/t/*`. GCLB intentionally opens only `/proxy/t/*` past IAP,
   so the request reached IAP before the native parent could validate and relay
   it.

## Fix

- Published `marin-iris-native==0.1.2` for the merged relay, then published and
  required `0.1.3` for the canonical route.
- Changed the inference orchestrator to consume
  `MintEndpointTokenResponse.capability_url`.
- Added the reciprocal `marin-dev` / `cw-us-west-04a` federation trust and peer
  configuration.
- Moved the cluster discriminator under the stable capability prefix:
  `/proxy/t/cluster=<peer>/<token>/<endpoint>/...`. Native Iris removes the
  discriminator and relays the remaining local capability path to the child.
- Added `experiments/evals/federated_inference_proxy_demo.py`, which asserts the
  new route and invokes `/v1/models` with literal curl.
- Restored the original GCLB path rule. No peer-specific or broad proxy rule is
  required.

## How OPS.md could have shortened this

- `lib/iris/OPS.md` "Job Management" could link the federation observation
  commands in `lib/iris/docs/federation.md` and note that parent-first rollout
  is required before a child starts minting cluster-tagged capability URLs.
- `lib/iris/docs/iap-gclb.md` now documents why federated capabilities remain
  below `/proxy/t/*` and why every other proxy path stays IAP-gated.

## Artifacts

- `.agents/projects/iris_federation/2026-07-25_federation_blind_relay_cluster_tag.md`
- https://github.com/marin-community/marin/actions/runs/30142227827
- https://github.com/marin-community/marin/actions/runs/30144647389
- https://iris-dev.oa.dev/#/job/%2Fpower%2Ffederated-inference-proxy-demo-20260725-v4
- `/power/federated-inference-proxy-demo-20260725-v4/inference-e876c8b2248748a0a0f9d493cdabcbfb`
