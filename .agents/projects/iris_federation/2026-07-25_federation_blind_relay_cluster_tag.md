# Federated capability URLs via a parent blind relay (cluster-tagged)

Status: accepted (2026-07-25). Supersedes the 2026-07-24 absorption design.
Tracking issue #7607, PR #7627.

## Shape

A job on a child cluster mints its capability token locally. The minted URL
names the child cluster and uses the parent origin:

```
https://iris.oa.dev/proxy/<cluster>/t/<token>/<name>/<subpath>
        └─ parent strips <cluster>, forwards the rest ─┐
                                                        ▼
        https://<child-proxy>/proxy/t/<token>/<name>/<subpath>
```

The parent recognizes `/proxy/<cluster>/t/…`, looks `<cluster>` up in its
configured `peers`, and forwards the remaining `/proxy/t/<token>/<name>/…` to
that child's proxy without reading the token. The child validates its own token
exactly as it would for a direct local capability URL, and serves. The parent is
a blind relay, not an auth boundary.

Chosen over the 2026-07-24 approach (parent mirrors a child's link endpoints and
mints a parent-signed token): minting happens where the job runs, so this routes
that URL as-is instead of redirecting the mint to the parent, and it drops all
parent-side per-endpoint mirror state. Accepted cost: a native-proxy wheel bump,
and the auth boundary sitting on the child.

## Why the direct URL fails today

The eval orchestrator runs in the child's namespace and calls its local
controller's `MintEndpointToken`. The token is signed by the child key
(`aud = proxy`). The URL is assembled from the minting controller's
`config.dashboard_url` (the child origin). That URL fails externally two ways:
the child origin is not world-visible, and a child-signed token presented to
`iris.oa.dev` is rejected — the parent's `NativeVerifier` validates `aud = proxy`
tokens only against its own JWKS (`auth.federation_peers` covers `aud =
federation` bearers, a separate path, never capability tokens).

## Trust

- The child is the sole auth boundary; the parent validates nothing about the
  token. This matches the capability-URL contract already in force: the token in
  the path is the credential, validated by its issuer.
- The relay fires only when the segment after `<cluster>` is the `t/` marker, so
  it relays a capability URL and nothing else. A request without `t/` is not a
  relay; it falls through to a normal (local) lookup. There is no path by which
  the relay exposes the child's other auth modes.
- The parent relays only to a cluster in its `peers` map (404 otherwise), so it
  is not an open relay to arbitrary hosts.
- A tampered `<name>` fails the child's endpoint-scoped token check; a tampered
  `<cluster>` routes to a different configured peer, where the token does not
  verify. Neither escalates (per-cluster signing keys are already distinct).

## Changes by layer

- Rust (`lib/iris/rust/src/lib.rs`): `parse_proxy_route` recognizes a leading
  `<cluster>/t/…` (a non-`t` first segment followed by the `t/` marker) and
  carries the relay peer. `native_decision` short-circuits: no local resolve, no
  token verify; it POSTs a relay decision and forwards the remainder
  `/proxy/t/<token>/<name>/<sub>` (plus query) to the returned upstream. Add
  `FederationDirection::Relay`. Bump the crate version; publish and pin
  `marin-iris-native`.
- Python (`dashboard.py::_federation_decision`): handle `direction == "relay"` —
  reject an unconfigured peer (404), resolve the peer proxy base (reuse the
  `FederatedEndpointHandoff` peer-address path), return
  `UPSTREAM_URL = <child-proxy>/proxy/t/<token>/<name>/<sub>?<query>` with no
  Authorization header.
- Mint / URL: add `capability_url` to `MintEndpointTokenResponse`; the minting
  controller builds the fully-qualified URL — the federated form
  `<parent-origin>/proxy/<self-cluster>/t/<token>/<name>` when a parent origin is
  configured, otherwise the local form. `cli/endpoints.py::mint`, the client
  helper, and the eval orchestrator use `resp.capability_url`.
- Config: add the child's public parent origin (`federation_public_parent`, one
  string, e.g. `https://iris.oa.dev`); empty falls back to the local URL. The
  parent needs nothing new — it already has `peers`.

## Kept from #7627 (independent cleanups)

Endpoints are governed by their lease, not by a task row (renewal is refused once
the task row is gone or terminal — `EndpointsProjection.add`), so these stand on
their own and are unaffected by the routing change:

- Migration 0048 drops the `endpoints → jobs/tasks` foreign key.
- The lease is the authoritative GC (`sweep_expired`); `delete_job` removes a
  job's endpoints explicitly; the redundant `remove_by_job_ids` calls in the
  pruner and federation store are gone.

## Dropped from #7627 (absorption, superseded)

The parent no longer mirrors a child's link endpoints; blind relay needs no
parent-side endpoint state. Reverted: `reads.live_local_link_endpoints`, the
`_federation_endpoint_snapshot` link merge, `EndpointServiceImpl.advertises_link_endpoint`,
the dashboard inbound-decision relaxation, and the absorption tests. Handoff
endpoint mirroring for genuinely handed-off jobs predates #7627 and is untouched.

## Known limitation

An app that returns a root-relative redirect or absolute-path link will not
round-trip: the child rewrites `Location` against its own `/proxy/t/<token>/<name>`
prefix, which omits the `<cluster>` segment. The target use case (Daytona →
vLLM OpenAI API, `/v1/*`) issues no redirects, so this is left unhandled and
noted. A browser-facing endpoint would need parent-side inner→outer prefix
translation as a follow-up.

## Rollout

- Publish the relay-capable `marin-iris-native` wheel; bump the pin.
- Deploy the parent proxy first (so it understands the `<cluster>/t/` route),
  then configure children with `federation_public_parent` so mint begins emitting
  tagged URLs. A tagged URL never resolves before the parent can route it.
- Migration 0048 rides along, unchanged from #7627.

## Test plan

- Rust unit: relay parse (`<cluster>/t/…`), relay decision shape, no token
  validation on the relay path.
- Python: `_federation_decision` relay direction — configured peer resolves to an
  upstream; unknown peer is rejected.
- e2e with the real native proxy (as in `test_federation_proxy`): child mints,
  parent relays `/proxy/<cluster>/t/<token>/<name>/…`, child serves; a wrong
  cluster tag is rejected.
- Mint: `capability_url` federated vs local form by config.
