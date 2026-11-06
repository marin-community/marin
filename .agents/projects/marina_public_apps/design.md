# Public Marina apps

Status: proposal. Tracking issue: [#10869](https://github.com/marin-community/marin/issues/10869).
Code references pin `main` at `70da6250be`. Checks against production ran on 2026-11-06.

## Summary

Marina can already serve an app to the internet. Public access is a per-app
setting (`mode = public`), off by default, and nine live apps use it,
including the Snowball family tree and the RL Data Atlas. This proposal keeps
that model: nothing becomes public unless someone marks that one app public.

Four gaps make the feature hard to use for maintained, checked-in apps:

1. **Merging does not release.** An app changes on the internet only when
   someone runs `marina release`. The nightly job meant to cover the family
   tree has never run. The public tree shows 130 models; `main` has 144.
2. **Named apps cannot be public.** Apps with `name` in `applet.toml` (Echo,
   EvalDash, TaskTrove, Zephyr) are rejected by two checks, and the public
   service does not serve `/<name>/` routes.
3. **Public URLs are UUIDs.** A public app lives only at
   `https://public.applets.marina.oa.dev/a/<uuid>/`. Vanity hostnames exist,
   but only on the signed-in service.
4. **"Public" is not in source.** The setting is database state. Any signed-in
   Marina user can flip any unnamed app to public with one command, with no
   review.

The proposed work, in order of value:

| Phase | Change | Size | Needs an operator |
|---|---|---|---|
| 0 | Turn on the existing nightly tree release | hours, no code | IAM apply, one repo variable |
| 1 | Release changed apps when their source merges to `main` | 1–2 days | no (reuses phase 0 grant) |
| 2 | Declare `access = "public"` in `applet.toml`; allow it for named apps; serve `/<name>/` publicly | 2–3 days | Marina rollout |
| 3 | Public vanity hostnames | 1–2 days | Pulumi apply (DNS, domain mapping) |
| 4 | Harden public backends (database role, caching) | 2–4 days | Marina rollout |

Sizes are estimates. Phases 2 and 3 were prototyped and tested for this
document (see [Evidence](#evidence)); phases 1 and 4 were not.

## How public access works today

Two Cloud Run services run the same Marina image against the same Postgres
database ([`infra/marina/__main__.py:302`](https://github.com/marin-community/marin/blob/70da6250be/infra/marina/__main__.py#L302-L375)):

| | `marina` | `marina-public-applets` |
|---|---|---|
| Hosts | `marina.oa.dev`, `applets.marina.oa.dev`, `echo.oa.dev`, `evaldash.oa.dev`, `filer.marina.oa.dev` | `public.applets.marina.oa.dev` |
| Access | Google sign-in (IAP) | anyone (`allUsers` invoker) |
| Methods | all | GET and HEAD only |
| Routes | everything | `/a/<uuid>/…` for apps whose mode is `public`; all else 404 |
| Caller identity seen by app code | the signed-in user | `None` |
| Capacity | — | 1–4 instances, 8 concurrent requests each |

The public service checks the app's mode on every request
([`server.py:782`](https://github.com/marin-community/marin/blob/70da6250be/infra/marina/src/marina/server.py#L782-L794)),
so a change to `private` takes effect at once. SQL, release, mode change, MCP
and all write methods are unreachable there.

An app's Python backend does run on the public service. Its GET handlers
execute with the app's database role and Marina's service account. PR
[#9305](https://github.com/marin-community/marin/pull/9305), which added public
mode, states the assumption: "Public mode assumes trusted applet code and
unrestricted data."

Belay ([#10829](https://github.com/marin-community/marin/pull/10829)) is the
working example of a public app with private parts. Its public page calls only
`/public/*` GET routes. Its admin and agent routes check the caller's identity,
which is `None` on the public service, and return 403. The same pattern fits
any app that needs a public view and a private view.

### Live state

`marina apps list` shows 23 active apps: 9 public, 14 private. All four named
apps are private. For the checked-in apps, `marina release-status` reports:

| App | Mode | Live vs `main` |
|---|---|---|
| Snowball family tree | public | `unrecorded` (released by hand, no commit recorded); 14 models behind |
| RL Data Atlas | public | `different`; behind by the rename commit only ([#10767](https://github.com/marin-community/marin/pull/10767)) |
| Belay | public | `current` |
| EvalDash | private, named | `current` |
| Echo | private, named | `different`; 3 commits behind |
| Filer, TaskTrove, Zephyr, Python playground | private | `unrecorded` |

Four of the nine checked-in apps have never been released from a recorded
commit, so nobody can tell from Marina what source they run.

## Gaps

### 1. Merging does not release

Before [#10137](https://github.com/marin-community/marin/pull/10137)
(2026-10-15) the four named apps were built into the Marina image, and a merge
to `main` rolled out a new image. #10137 moved every app into versioned
Postgres packages released by `marina release`. That was deliberate: it gives
each app its own version history, rollback and transactional migration, and a
Marina restart can no longer change which app version is live. Its README says
"Merging source does not release it." No automation replaced the old path.

The family tree and the Atlas were never in the old path. They have always
been released by hand.

[#10719](https://github.com/marin-community/marin/pull/10719) added a nightly
release for the family tree only. It is not running:

- The repository variable `SNOWBALL_TREE_RELEASE` does not exist, so the
  scheduled runs on 2026-11-04 and 2026-11-05 were skipped.
- If it were enabled, the job would stop at once. The live package is
  `unrecorded`, and the job refuses that state until someone runs the one-time
  `bootstrap_unrecorded` dispatch. A local dry run reproduced the refusal.
- The job signs in to Marina as
  `marin-cd-cloud-run-deploy@hai-gcp-models.iam.gserviceaccount.com`. The IAP
  grant for that account is in source
  ([`iac/gcp/marina.py:101`](https://github.com/marin-community/marin/blob/70da6250be/infra/pulumi/src/iac/gcp/marina.py#L96-L103)).
  Whether an operator has applied it was not checked.

### 2. Named apps cannot be public

`AppletStore.publish` and `AppletStore.set_mode` reject public mode for a named
app ([`applets.py:572`](https://github.com/marin-community/marin/blob/70da6250be/infra/marina/src/marina/applets.py#L572-L573),
[`applets.py:923`](https://github.com/marin-community/marin/blob/70da6250be/infra/marina/src/marina/applets.py#L923-L927)).
The public service passes only `/a/…` paths
([`server.py:245`](https://github.com/marin-community/marin/blob/70da6250be/infra/marina/src/marina/server.py#L245-L257)),
and the `/<name>/…` routes require sign-in.

Removing the two checks alone is not enough. A named app forced to public in
the database is reachable at `/a/<uuid>/v/N/`, but Echo, EvalDash and TaskTrove
build their frontends for the absolute path `/<name>/`, so their pages would
load without assets or routes there. The public service has to serve
`/<name>/…` itself.

The name carries three capabilities that unnamed apps cannot use: a data
route (`/<name>/data/*`, a proxy to an object-storage prefix), scheduled jobs,
and the Loom agent panel. Each needs a decision for anonymous readers; see
[Named apps](#2b-named-apps-on-the-public-service).

### 3. Public URLs are UUIDs

`named_applet_hosts` already maps a hostname to one app, pins requests to it
and hides every other route. `create_app` applies it only on the signed-in
service ([`server.py:1368`](https://github.com/marin-community/marin/blob/70da6250be/infra/marina/src/marina/server.py#L1368-L1373)),
and Pulumi maps every configured hostname to that service
([`__main__.py:383`](https://github.com/marin-community/marin/blob/70da6250be/infra/marina/__main__.py#L383-L389)).

### 4. "Public" is not in source

The mode lives in the `applets.mode` column. It is set by
`marina release --mode public` or `marina apps mode UUID public`, and any user
admitted by IAP may run either on any unnamed app. A later release keeps the
stored mode. Nothing in `applet.toml`, and so nothing in a reviewed PR, says
which apps are meant to be public.

This is acceptable for one-off reports. It is weak for maintained apps, and it
becomes a real risk once named apps with broad data access (EvalDash) can be
made public by the same one command.

## Goals

- A maintained app can be marked public in its checked-in source, one app at a
  time. Every app stays private unless marked.
- A merge to `main` that changes a marked app reaches the internet within
  minutes, without a person running a command.
- A public app can have a stable, readable URL.
- Anonymous readers get read-only access to that app and nothing else.

Not goals: anonymous writes, public SQL or MCP, per-page access inside one app
beyond what the app's own handlers enforce, and a public app directory.

## Design

### 1. Release on merge

Add one workflow, triggered by pushes to `main` that touch
`infra/marina/apps/**` or `infra/marina/web/**`. It replaces the nightly
family-tree workflow and `scripts/ci/snowball_release.py`.

For each app directory that opts in, the job runs the steps the nightly script
already runs: `release-status`; skip if `current`; otherwise
`marina release --base-version <observed>`; then confirm the new revision is
live. `release-status` hashes the app directory and its declared
`source_paths`, so an unrelated commit does not trigger a release.

Opt-in is a field in `applet.toml`:

```toml
release_on_merge = true
```

Reasons for opt-in over "every app with an `applet_id`": a release runs the
app's `migrate()` against production, and some apps pair a code release with a
manual data load (the Atlas returns 503 between a schema change and its
loader). Those maintainers should choose.

Details:

- One concurrency group, no cancellation, so two merges release in order.
- `marina release` already refuses a checkout that is not current `main` and a
  stale `--base-version`. A lost race fails the run; the next push releases.
- App-specific checks move into `build_command` so the generic job needs no
  per-app steps. The family tree's `verify_layout.js` is the only one today.
- Keep `workflow_dispatch` with `dry_run`, and a daily schedule as a safety
  net for a failed push run.
- Releases are fast. Local dry runs took 3–5 s for the tree, Atlas and Belay,
  and 14 s for EvalDash (frontend build included).

This is the change that removes the staleness. It does not depend on the
other phases.

Returning to image-built apps was considered and rejected. It would undo
per-app versions and rollback, and it would tie every app change to a full
Marina rollout.

### 2a. Declare access in source

Add an optional manifest field:

```toml
access = "public"   # omitted means private
```

Rules:

- When `applet.toml` declares `applet_id` (a maintained app), the manifest
  decides the mode. `marina release` sets the stored mode from it on every
  release. Removing the line and releasing makes the app private again.
- `marina apps mode` stays as it is for apps without a manifest declaration
  (ad hoc apps), and as an emergency switch to `private` for any app. Setting
  a maintained app to `public` by CLI is rejected with a message that points
  to `applet.toml`.
- A named app can become public only through the manifest, released from
  clean current `main`. That makes a reviewed PR the only way to expose a
  named app.

The stored `mode` column remains the value the public service reads, so
revocation stays immediate and needs no deploy.

### 2b. Named apps on the public service

Remove the two rejections. On the public service, serve GET and HEAD for
`/<name>/…` when the named app's stored mode is `public`, with identity
`None`. Everything else keeps returning 404. The prototype does this in about
60 changed lines of `server.py`.

Decisions for the named-only capabilities:

| Capability | Proposal for anonymous readers |
|---|---|
| Static files and client routes | Served |
| `/<name>/api/*` GET, HEAD | Served; the app's handlers must treat `None` identity as anonymous |
| `/<name>/api/*` writes | 404 (already enforced by the service) |
| `/<name>/data/*` | **Not served unless the manifest opts in** with `public_data = true`. This route proxies a whole object-storage prefix using Marina's credentials. Making an app public should not silently publish its bucket prefix. |
| Agent panel | Off. Loom needs a signed-in user. The prototype omits Loom from the page's CSP when the caller is anonymous; the app's frontend should hide the panel when `/api/marina/me` is unavailable. |
| Scheduled jobs | Unchanged. Jobs run on separate runners and are not exposed. |

App owners carry the review burden, as today: before marking an app public,
read every GET handler and ask what an anonymous caller can read or trigger.
Belay's split into `/public/*` routes plus identity checks is the model to
copy.

### 3. Public vanity hostnames

Add a Pulumi list `marin-marina:public_applet_domains`. Each host gets a Cloud
Run domain mapping to `marina-public-applets` and a Cloudflare CNAME, the same
way private hosts are provisioned now. Pass the list to the public service as
`MARINA_APPLET_DOMAINS`, and wrap the public service in the existing
`named_applet_hosts` router.

Binding stays in Postgres (`marina apps hosts bind HOST UUID`), so an operator
can rebind without a deploy. A host bound to a private app returns 404 on the
public service, because the per-request mode check still runs after the host
rewrite.

Two limits to know:

- A new hostname needs a Pulumi apply by an operator. That is a one-time cost
  per host.
- The host router mounts its app at the root of the host. That works for apps
  built with relative paths (the tree, Atlas, Belay, Zephyr). Echo, EvalDash
  and TaskTrove are built for `/<name>/`; on the public service they are
  reached at `https://public.applets.marina.oa.dev/<name>/` after phase 2, or
  they change their build base to get a host of their own.

Hostname choice is open (for example `tree.marina.oa.dev` or a name under a
public project domain).

### 4. Hardening for public backends

These are existing properties of public mode, not regressions. They matter
more as more apps with backends become public.

- **Database reach.** Every app role inherits `marina_reader`, which can read
  every app's schema ([`database_setup.py:54`](https://github.com/marin-community/marin/blob/70da6250be/infra/marina/src/marina/database_setup.py#L54-L58)).
  A SQL-injection bug in one public GET handler would expose all apps' tables,
  including private ones. Proposal: on the public service, give backends a
  role that can read only the app's own schema.
- **Credentials.** The public service runs as the same service account as the
  private one and mounts the CoreWeave keys. Proposal: a separate service
  account with no storage write access, and mount secrets only if a public app
  needs them.
- **Load.** Files and API responses are served with `public, no-cache`, from
  Postgres, by at most 4 instances that share the database with private
  Marina. A traffic spike on a public page could slow internal apps. Proposal:
  allow caching of immutable `/v/<revision>/` assets, and consider a CDN or a
  connection cap for the public service.
- **Embedding.** Pages send `frame-ancestors 'none'`, so a public app cannot
  be embedded in another site. Change only if embedding is wanted.
- **Small defect found.** On the live public service,
  `/a/<uuid>/<anything not under v/ or api/>` returns 401 where 404 is
  intended, because the path falls through to the signed-in named route. It
  leaks nothing. The prototype fixes it.

## What each app needs

| App | To be public and fresh | Notes |
|---|---|---|
| Snowball family tree | Phase 0 now; phase 1 for minutes-not-a-day; phase 3 for a readable URL | Static. Already public. |
| RL Data Atlas | Phase 1, if its maintainers opt in; phase 3 for a URL | Already public. 9 GET routes, no identity use. Data loads stay manual and separate from code release. |
| Belay | Phase 1 if wanted | Already public and current. |
| EvalDash | Phases 2a and 2b, plus an app review | 15 GET and 3 POST routes; one helper reads the caller. Exposes run logs, samples and artifact paths. Needs an explicit decision on what the public may see, likely a Belay-style public subset. Agent panel must hide for anonymous readers. |
| TaskTrove | Phases 2a and 2b, plus `public_data` | Reads `s3://marin-us-east-02a/marin/tasktrove/…` through the data route. |
| Zephyr | Phases 2a and 2b | 6 GET routes, relative paths. |
| Echo | Not a candidate | Indexes the private wiki and Discord. |

"Snowball Atlas" appears in PR titles for the family tree. This document
treats the request as covering both the family tree and the RL Data Atlas.

## Evidence

All commands ran from the `task/343` worktree. The prototype and probe file
are on branch `task/343-prototype` (commit `055cb5a078`), which is not for
merge.

Production, read-only:

| Check | Result |
|---|---|
| `marina apps list --json` | 23 apps, 9 public, named apps all private |
| `marina release-status` on 9 checked-in apps | table in [Live state](#live-state) |
| Live tree page vs `graph-data.json` on `main` | 130 vs 144 nodes; 14 on `main` only, 0 live only |
| Anonymous `curl` to the public host | tree, Atlas and Belay 307 to their revision; EvalDash by UUID, `/evaldash/` and Filer 404 |
| Scheduled runs of the nightly workflow | 2026-11-04 and 11-05 both `skipped`; variable returns 404 from the API |
| `snowball_release.py --dry-run` | fails: "Live source is unrecorded"; passes with `--bootstrap-unrecorded` in 4.6 s |
| `marina release --dry-run` on Atlas, Belay, EvalDash | pass in 3.3 s, 3.2 s, 13.5 s |

`main`, against a disposable Postgres (`tests/test_probe_public_343.py`):

| Probe | Result |
|---|---|
| Release a named app with `mode=public` | 400 "Named applications are private…" |
| `apps mode` public on a named app | 400 "Named applications remain private" |
| Named app forced public by SQL | `/a/<uuid>/v/1/` and its API 200; `/<name>/` 404 |
| Bound hostname on the public service | 404 (host ignored) |
| Public service wrapped in `named_applet_hosts` from the test, no source change | host root 307 to `/v/1/`; page, asset, API read 200; identity `None`; POST 404; `/api/marina/apps` 404; another public app's `/a/…` 404; a host bound to a private app 404 on every path |
| Same, after a release, a switch to private, an unbind | root moves to `/v/2/`; then 404; then 404 |

Prototype branch:

| Probe | Result |
|---|---|
| Named app released with `mode=public` | 201 |
| Public service: `/<public name>/`, asset, client route, API GET, `data/` | 200, `public, no-cache`, identity `None` |
| Public service: POST and DELETE on its API | 404 |
| Public service: `/<private name>/`, its API, its `data/` | 404 |
| Public service: `/`, `/api/marina/apps`, `/me`, `/mcp/`, POST `/api/marina/applets` | 404 |
| Signed-in service without a login: `/<public name>/` | 401 (unchanged) |
| Switch the named app to private | public `/<name>/` and API 404 at once |
| `/a/<uuid>/nope` on the public service | 404 (was 401) |
| `infra/marina/tests` | 148 pass, 2 fail; both failures assert the removed rejection |
| `infra/marina/apps` tests | 260 pass, 20 skipped |

Not tested: the Pulumi domain mapping for a public host, the CI identity's IAP
grant, a real release from GitHub Actions, browser behavior of EvalDash or
TaskTrove as an anonymous reader, and load on the public service.

## Rollout

0. **Unblock the tree today.** Confirm the `marin` stack grant is applied, run
   the `bootstrap_unrecorded` dispatch after comparing live with `main`, then
   set `SNOWBALL_TREE_RELEASE=enabled`. The steps are in
   [the tree's maintenance guide](../../../docs/references/snowball-family-tree.md#nightly-publication).
1. Land release-on-merge. Opt in the family tree first, then delete the
   nightly workflow, its script and the repository variable.
2. Land the manifest field and the named-app public routes with regression
   tests built from the probes above. Add `access = "public"` to the tree,
   Atlas and Belay; their stored mode does not change.
3. Add `public_applet_domains`, apply, bind the first host.
4. Hardening, sized after a look at traffic.

Each phase is independently useful and independently revertible. Reverting an
app to private is one release, or one `marina apps mode UUID private`.

## Open questions

- Which apps should be public besides the tree and the Atlas? EvalDash needs a
  product decision on what outsiders may see before any engineering.
- Hostnames for public apps, and whether they belong under `oa.dev`.
- Should release-on-merge be opt-in per app (proposed) or the default for
  every maintained app?
- Should making an unnamed, ad hoc app public keep working for any signed-in
  user, or require an operator?
- For purely static apps such as the tree, GitHub Pages or `gs://marin-public`
  would also give deploy-on-merge. This proposal keeps them in Marina so one
  mechanism covers static and backend apps, but the simpler host is a valid
  choice for the tree alone.
