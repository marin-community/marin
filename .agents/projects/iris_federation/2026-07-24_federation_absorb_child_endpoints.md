# Federation: let iris.oa.dev mint a URL for a link endpoint on a child cluster

Issue: marin-community/marin#7607. RNO capability URLs return 403 from Daytona.
Revised after two peer reviews (codex + a repo-grounded agent); both redirected
the first draft away from "absorb jobs" toward an endpoint-only design.

## Problem

The Grug/OpenCode smoke starts an H100x8 vLLM serving job on `cw-rno2a` and hands
a capability URL to a Daytona sandbox. The sandbox calls `<url>/v1/models` and
gets HTTP 403, so no inference reaches vLLM. The serving job was launched
directly on `cw-rno2a` (`--cluster cw-rno2a`), so the capability URL carries the
child origin `iris-cw-rno2a.oa.dev`, which is not usable from outside CoreWeave.

The parent (`iris.oa.dev`) is the public front door, but it cannot currently be
an alternate route for that endpoint. Federation is built around *handoff*: marin
records a SENT handle, the peer a RECEIVED handle, and three checks each scope
cross-cluster visibility to that handle. For a job started directly on the child
none of them holds:

- `federation_sync` (peer, `service.py:3463`) reports only `received_jobs_for_requester`
  and `live_endpoints_for_requester` — the requester's handed-off jobs and their
  endpoints. Steady state runs off `changelog_rows_since`, and the changelog is
  written only for RECEIVED roots (`writes.py:477`), so a local job is never
  reported.
- `apply_sync_batch` (parent, `federation_store.py:272`) drops any delta whose
  root has no local SENT handle.
- the child's inbound `/proxy` owner-check `_federation_owner_check`
  (`controller.py:507`) → `has_received_job_from_peer` (`reads.py:2068`) admits a
  parent's forwarded request only for a job the parent handed here.

A child-minted token cannot be rehosted at marin either: marin's proxy validates
capability tokens against its own JWKS, and a child-signed token is not in it. So
the capability must be minted *by* marin against state marin holds.

## What already works: route the whole root through marin (Option A)

Federation moves whole root jobs only. The eval submits one orchestrator root and
the serving job is its child (`runner.py:349`, `inference/iris.py`), so the fix is
to land the whole root on `cw-rno2a` *via* marin rather than submitting it to the
child:

1. An external user (not an in-cluster job — an in-cluster submit authenticates as
   `local_admin` and is refused federation, `service.py:1200`) submits the
   orchestrator root to marin with `--target-cluster cw-rno2a`.
2. marin hands the whole tree off; the serving child registers its endpoint on the
   child under a RECEIVED-from-marin root, and it mirrors back to marin with
   `peer_id=cw-rno2a` (`replace_remote_for_peer`, `endpoints.py:385`).
3. marin mints the capability (retrying until the mirror appears — the sync
   interval is 3 s, so mint must not assume job-ready implies mirror-ready) and
   returns an `iris.oa.dev` URL. marin's proxy resolves the mirrored endpoint,
   takes the outbound federation decision (`dashboard.py:269`), and forwards to
   the child, whose inbound check passes on the RECEIVED handle.

This is mechanism-correct today with no protocol change; it is the near-term way
to make the smoke green. It does *not* answer penfever's literal question ("keep
the group controller on marin, move only the serving child") — that topology is
impossible (whole-root federation), and the correction is that the whole root
runs on the child while the endpoint is minted and proxied through marin. Option A
is not covered by any existing test (mint tests are local-only,
`test_endpoints.py:408`); an end-to-end test is part of landing it.

## The requested extension: mint through iris.oa.dev for a direct child launch (Option B)

rjpower's ask is the burden-reduction case: mint against `iris.oa.dev` even when
the job was started directly on a child, without relaunching through marin. Both
reviews established that this needs *endpoint* information, not job absorption.

Key fact (verified): `mint_endpoint_token` (`service.py:2929`) resolves the
endpoint from the in-memory projection and authorizes via
`authorize_resource_owner(row.task_id.user)`, where `user` is parsed from the
endpoint's `task_id` *string*. It never reads a `jobs`/`tasks` row. The endpoint
snapshot is already sent full-set every sync, independent of the changelog cursor,
and set-replaced on the parent (`service.py:3496`, `federation_store.py:287`), so
lease expiry, access changes, and removal self-heal within one sync interval. The
only thing that forced job/task shells in the first draft was the `endpoints` FK
to `jobs`/`tasks` (`schema.py:511`), whose sole role is CASCADE cleanup — which
the sync path already bypasses by deleting endpoints through the projection before
`delete_job` (`federation_store.py:275`).

So the minimal correct design is endpoint-only:

1. Peer side (`cw-rno2a`): widen `live_endpoints_for_requester` to also return the
   child's *locally-owned, link-access* endpoints, not only endpoints under a
   RECEIVED-from-requester root. Only a configured federation parent can call
   `federation_sync` (it authenticates as `FEDERATION_PEER_ROLE` against
   `federation_peers`), so this reports local link endpoints only to a parent hub.
   No job/task shells, no changelog change, no new direction. Private endpoints are
   never reported.
2. Parent side (`marin`): allow a peer-stamped endpoint row to exist with no
   backing `jobs`/`tasks` row — relax the `endpoints` FK (or scope it to
   `peer_id IS NULL` rows) via a migration, and drop the `_present_task_ids` skip
   for remote rows. The row's `task_id`/`job_id` are strings for owner parsing and
   display only. Lifecycle is set-replace + the child's own lease (reported as
   remaining duration, so the parent stores the child's real expiry and never
   extends it), so when the child's endpoint goes away the parent row is reaped on
   the next sync and nothing else is left behind.
3. Child inbound proxy: relax `_federation_owner_check` so a forwarded request from
   a configured parent is admitted for an endpoint the child *currently advertises
   as link-access*, resolved from the child's own registry by the decision's
   `encoded_name`, even without a RECEIVED handle. The Rust proxy already POSTs an
   inbound decision for any local endpoint hit by a federation peer
   (`lib.rs` `mapping_decision`), so this is a pure-Python change to the decision
   handler; private endpoints stay gated on the RECEIVED handle.

The whole of Option B is Python plus one SQLite migration — no native-proxy wheel
release, because the Rust proxy's outbound (marin resolves a `peer_id` endpoint via
a capability token) and inbound (child POSTs a decision for a peer-hit local
endpoint) paths already exist and are access-mode agnostic.

### The trust decision the maintainer must make

The child inbound relaxation is where the real policy choice lives, and the two
reviews disagreed:

- Implicit, link-scoped (recommended): any endpoint the child registers as
  link-access is visible to, and forwardable by, its configured parent hubs, with
  no per-job opt-in. This matches "reduce the user burden … even if you accidentally
  started the job on a child cluster directly." The escalation is bounded: a
  link-access endpoint already means "URL is the credential," and any identity the
  child authenticates can already traverse it; admitting a *pinned parent's* forward
  adds no reachability a link endpoint did not already grant. It must be stated,
  not derived from the mint gate (which is owner-or-admin, not link-access): the
  child is trusting its configured parents' admitted user population. `cw-rno2a`
  pins two parents (`marin`, `marin-dev`, `config/cw-rno2a.yaml:89`), so both gain
  sync-visibility and forward-traversal of every local link endpoint.
- Explicit peer-scoped grants (codex): the serving job explicitly grants a named
  parent access to a named endpoint (generation, owner, expiry); sync returns only
  matching grants; the inbound check requires an active grant for the exact
  `(peer_id, endpoint_id, generation)`. Strictly scoped, but needs a grant schema
  and a per-job opt-in, so it is less automatic than the ask implies.

Recommendation: ship the implicit link-scoped form, gated to link-access endpoints
and configured parents, and state the trust assumption in `federation.md`. Move to
explicit grants only if the child must withhold some link endpoints from a parent.

### Identity caveat

Mint authorizes `row.task_id.user`. Copying the child's user string to marin is
only sound where user ids are globally meaningful. This deployment is single-tenant
shared-IAP (`allowed_submitters: *@openathena.ai`), so `alice@openathena.ai` is the
same principal on both clusters; state this. A multi-tenant future needs
issuer-qualified identity or an explicit mapping before this is safe.

### What the reviews killed, and why it is gone

- `FederationDirection.ABSORBED`, absorbed job/task shells, and the
  `apply_sync_batch` guard relaxation: unnecessary once mint needs no job row. They
  also leaked command lines/env/siblings via job-delta reuse, weakened the
  anti-injection guard, and left mirrored rows un-GC'd (no requester-attributed
  tombstone reaps them). Dropped.
- Widening the job query `RECEIVED ∪ local-link roots`: fatal, because steady-state
  sync is changelog-driven and a local job has no requester attribution, so shells
  would never arrive incrementally. Moot once shells are gone — the endpoint
  snapshot is already cursor-independent.
- Collision policy, peer-vs-peer job-delta races, absorbed-row GC: all specific to
  job absorption; gone with it. Endpoint rows are keyed by `endpoint_id` (UUID) and
  peer-stamped, and set-replace is per-peer, so a mirrored row is only ever
  rewritten by its own origin peer.

## Rollout order

Land the child inbound relaxation *before* the parent mint path. The relaxation is
inert on its own (nothing forwards yet), and the reverse order would mint
`iris.oa.dev` URLs that 403 at the child — issue #7607 one hop deeper.

1. Child `_federation_owner_check` relaxation for advertised link endpoints
   (+ unit tests: peer forward to a link endpoint is admitted; to a private
   endpoint is refused; to an expired/absent endpoint is refused).
2. Peer `live_endpoints_for_requester` widening to local link endpoints, gated to
   the syncing parent identity.
3. Parent endpoints-FK relaxation migration + remove the remote-row task-present
   skip; mint against a mirrored row with no backing job.
4. End-to-end: a local serving job on a child, mint against the parent, proxy
   round-trip returns 200; privatizing or dropping the endpoint invalidates the
   parent URL within one sync interval; a second parent's forward to a private
   endpoint stays 403.

Option A (route-through-parent) can land first and independently as the immediate
smoke fix.
