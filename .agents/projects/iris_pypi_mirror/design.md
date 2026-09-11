# Iris PyPI Mirror

_Why are we doing this? What's the benefit?_

Recent github.com partial outages and PyPI throttling events have caused Iris
task installs to fail during the BUILDING phase, where every task runs
`uv sync` against public indexes. We want a pull-through cache that survives
public-index incidents and trims the long tail on cold-boot installs. Marin
already runs this exact pattern for GHCR images
([`ghcr-mirror`](https://github.com/marin-community/marin/blob/main/lib/iris/docs/image-push.md));
this design extends it to Python wheels via Google Artifact Registry remote
PyPI repos. See [#4897](https://github.com/marin-community/marin/issues/4897).

The 90% target: `pypi.org` and `pytorch-cpu` (the only pytorch index we use
on GCP — TPU and CPU jobs both resolve through it). GPU/CUDA wheels and
`git+https://github.com/...` deps are out of scope for v1 (see
[`research.md`](./research.md) §"Open decisions" for the deferrals).

**Co-prerequisite: publish `dupekit` to PyPI as `marin-dupekit`.** `dupekit`
is currently a top-level dependency that resolves via a `find-links` entry
pointing at
`github.com/marin-community/marin/releases/expanded_assets/...` (see
`pyproject.toml:28-31`). uv's `expanded_assets` URL handler is a uv-internal
HTML scraper, **not a PEP 503 Simple index** — Artifact Registry remote PyPI
repos cannot proxy it. So as long as `dupekit` is hosted on GitHub Releases,
github.com is on the install hot path no matter what we do for PyPI proper.
The fix is to publish to PyPI; once published, AR's `pypi-mirror` repo
proxies it like any other wheel. PyPI has no real namespaces, so we follow
the convention of hyphenated org prefixes (`google-cloud-*`, `azure-*`) and
publish as **`marin-dupekit`**. The Python import name stays `dupekit`
(controlled by `module-name` in `rust/dupekit/pyproject.toml`); only the
distribution name changes. We're treating this as part of the same project
rather than a follow-up — the AR mirror buys ~80% without it, but pulling
github.com fully off the install path requires both. See
[`spec.md`](./spec.md) §"PyPI publication for `marin-dupekit`" for the
concrete steps. (`kitoken` uses the same pattern but sits behind the
`kitoken` extra; out of scope for v1, same fix shape — would publish as
`marin-kitoken` — if/when it matters.)

## Challenges

_What's hard?_

Three real ones, none fundamental:

1. **uv has no transparent fallback.** If a configured index returns a
   transport error, uv halts resolution rather than retrying against another
   index. Wiring AR as the default index therefore makes AR a hard
   dependency on the install path. We accept this — the alternative (a
   wrapper that retries against upstream after N failures) reintroduces the
   failure mode we're trying to remove.
2. **`pytorch-cpu` is a `[[tool.uv.index]]`-named index in
   `lib/marin/pyproject.toml`, not the default index.** Overriding it
   without source edits requires `UV_INDEX_PYTORCH_CPU=<url>` (uv name-mangles
   to uppercase + hyphen→underscore); a single `UV_DEFAULT_INDEX` is not
   enough.
3. **Auth on workers must use the existing service-account identity** —
   no static credentials baked into images. uv needs
   `UV_KEYRING_PROVIDER=subprocess` plus
   `keyrings.google-artifactregistry-auth` available in the task base image
   so it can mint short-lived OAuth tokens off the GCE metadata server.

## Costs / Risks

- AR remote endpoints become critical-path for task install. An AR outage now
  blocks new tasks from building. We accept this on the assumption that AR
  is more reliable than `pypi.org`. **Partial-region AR failures are more
  common than global PyPI outages**; a single-region AR incident (e.g. `us`
  down, `europe` up) breaks all GCP-`us` workers until manually flipped to
  opt-out or the region recovers. Documented as a known asymmetry.
- Mirror cold-cache: AR caches only what's been requested once successfully.
  An outage that hits us before any pre-warm leaves new wheel versions
  unavailable. Mitigation: nightly canary already exercises the common
  resolution path, which warms the cache.
- **Cold-fanout pressure**: a fleet-wide ferry launching N workers
  simultaneously and all cold-pulling the same wheel can amplify upstream
  pressure or hit AR pull rate limits during a release. Daily ferry warms
  steady-state; new wheel versions in a PR cold-pull once and serve from
  cache thereafter.
- **UV cache invalidation**: workers reuse `/uv/cache` across tasks
  (env.py:85, mounted via `task_attempt.py:715`). After rollout, uv may
  repopulate cache entries on first sync because the resolved index URL
  changes. One-time cost, harmless.
- One-time infra setup: 4 AR repos (`us`/`europe` × `pypi-mirror` /
  `pytorch-cpu-mirror`), reusing existing 30-day cleanup policy and IAM
  grants.
- The `marin-resiliparse` index (GitHub Pages) is unmirrored; evals using
  `chatnoir-resiliparse` will still fail during a github.com outage. Same
  for `harbor` and `lm-eval` (`git+https`). These are accepted residual
  risks — see [`research.md`](./research.md).
- **PyPI publication adds a release-time dependency on PyPI.** Wheel pushes
  during a PyPI outage will fail; `pypi.org` install-side reads are mirrored
  via AR. Asymmetry accepted — read-path outages affect every task,
  write-path failures only block releases. **Caveat**: PyPI rejects
  re-uploads at the same version, so a partial-success push (some wheels
  uploaded, others failed) poisons the version. Retry then requires a
  cargo version bump, not just a workflow re-run; documented in
  [`spec.md`](./spec.md) §"Version handling".

## Design

_How are we doing this?_

**Infra (one-time, 4 AR repos).** Mirror the `ghcr-mirror` setup with
`repository-format=python`:

```bash
gcloud artifacts repositories create pypi-mirror \
  --project=hai-gcp-models --location=us \
  --repository-format=python --mode=remote-repository \
  --remote-python-repo=PYPI

gcloud artifacts repositories create pytorch-cpu-mirror \
  --project=hai-gcp-models --location=us \
  --repository-format=python --mode=remote-repository \
  --remote-python-custom-repo=https://download.pytorch.org/whl/cpu
```

Repeat for `--location=europe`. Apply the same `roles/artifactregistry.reader`
grant the worker SA already has for `ghcr-mirror`. Cleanup policy: 30 days,
matching the existing pattern documented in
[`lib/iris/docs/image-push.md`](https://github.com/marin-community/marin/blob/main/lib/iris/docs/image-push.md).

**Region routing.** Reuse `zone_to_multi_region()` from
[`lib/iris/src/iris/cluster/providers/gcp/bootstrap.py:40-52`](https://github.com/marin-community/marin/blob/main/lib/iris/src/iris/cluster/providers/gcp/bootstrap.py#L40-L52).
The same `_ZONE_PREFIX_TO_MULTI_REGION` map (`us`, `europe`) and same
`_UNSUPPORTED_ZONE_PREFIXES` (`asia`, `me`) apply. Three outcomes:
unmapped prefixes (e.g. a future `southamerica-`) return `None` → skip
mirror, no error; `asia`/`me` raise `ValueError` → caught at the worker
callsite and treated as skip-mirror with INFO log; CoreWeave / no-region
workers also skip silently. Net behavior: graceful fallback to public PyPI
for any non-`us`/`europe` worker.

**Iris wiring.** Add a `pypi_mirror.py` module next to `bootstrap.py` with a
single function:

```python
def build_pypi_mirror_env(multi_region: str, project: str = AR_PROJECT) -> dict[str, str]:
    base = f"https://{multi_region}-python.pkg.dev/{project}"
    return {
        "UV_DEFAULT_INDEX": f"{base}/pypi-mirror/simple/",
        "UV_INDEX_PYTORCH_CPU": f"{base}/pytorch-cpu-mirror/simple/",
        "UV_KEYRING_PROVIDER": "subprocess",
    }
```

Call it from
[`_prepare_container_config` in `lib/iris/src/iris/cluster/worker/task_attempt.py`](https://github.com/marin-community/marin/blob/main/lib/iris/src/iris/cluster/worker/task_attempt.py#L680-L700)
right after `IRIS_WORKER_REGION` is set on the env dict (line 696) and
before the `_task_env` and user `env_vars` merges (lines 698–699). This is
the natural seam: the worker knows its zone there, and a user who *also*
sets `UV_DEFAULT_INDEX` in their job env_vars will still win (escape hatch
for debugging mirror issues). `build_common_iris_env` itself is unchanged.

Default on. Users opt out per-job by setting `IRIS_PYPI_MIRROR=0`
(exact-string match — `"false"`, `"off"`, etc. do **not** opt out; matches
existing `IRIS_DEBUG_UV_SYNC` convention) in their
`EnvironmentConfig.env_vars`. CoreWeave (no `IRIS_WORKER_REGION`),
`asia`/`me` workers (`zone_to_multi_region` raises, caught with an INFO
log naming the region), and unmapped prefixes (returns `None`, also
INFO-logged) all skip the injection and resolve from public PyPI as
today. The INFO log is the operator's signal during an outage that a
worker is unprotected.

**Auth.** Add `keyrings.google-artifactregistry-auth` to the task base image
(`lib/iris/Dockerfile`, `task` stage). With `UV_KEYRING_PROVIDER=subprocess`,
uv shells out to it on each request and the metadata server mints a token
from the worker SA. No static creds, no per-task config.

**`pyproject.toml` left untouched** _by the mirror wiring_. Local dev keeps
resolving against upstream PyPI / `download.pytorch.org`. The mirror only
kicks in when the worker env vars are present, so we never need a separate
dev/prod manifest.

**dupekit → PyPI as `marin-dupekit`** (one-time, parallel workstream).
Register the `marin-community` PyPI organisation, rename the package in
`rust/dupekit/pyproject.toml` (`name = "marin-dupekit"`; the import name
stays `dupekit` via `module-name`), configure GitHub Actions trusted
publishing (OIDC, no static tokens), update `scripts/rust_package.py` to
push wheels + sdist to PyPI in addition to the existing GitHub Release,
and rename + remove the dupekit entries in `pyproject.toml:23,29`. uv's
resolver then picks `marin-dupekit` up from `pypi.org` (and through the
AR mirror in tasks). Each release must bump the cargo version since PyPI
disallows re-uploads. Concrete steps in [`spec.md`](./spec.md) §"PyPI
publication for `marin-dupekit`".

## Testing

_Agents make mistakes — how do we catch them?_

- **Unit**: test `build_pypi_mirror_env` URL shape and `ValueError` on
  unsupported region; test the gating logic in `_prepare_container_config`
  for the four cases (mirror enabled, opt-out, asia/me ValueError caught,
  no `IRIS_WORKER_REGION`).
- **Integration**: run the existing dev-cluster smoke job with the mirror
  enabled, verify task install succeeds and AR access logs show the pull.
  Re-run with `IRIS_PYPI_MIRROR=0` and verify task install resolves directly
  from `pypi.org`.
- **Auth path**: explicit test that a worker with no GCE metadata access
  fails the `uv sync` cleanly (not silently). This catches misconfigured
  IAM grants on the worker SA.
- **Outage simulation**: point `UV_DEFAULT_INDEX` at an obviously-bad URL
  on a test worker and confirm the build fails fast with a useful error
  (not a silent fallback) — guards against accidental fallback wiring
  slipping in later.
- **Rollout**: enable on a single worker pool first; let the daily ferry
  exercise it for one full cycle before flipping the default for all
  workers. Pre-warming happens implicitly via the daily run.
- **Reverse compatibility**: existing tasks should keep working with
  unchanged behavior when the env vars are not set (e.g. on CoreWeave or
  unsupported regions).

**Rollback levers.** Three escalating options, no new infrastructure
required:

1. Per-job: set `IRIS_PYPI_MIRROR=0` in `EnvironmentConfig.env_vars`.
2. Worker default: workers already merge `_task_env` (`task_attempt.py:698`);
   adding `IRIS_PYPI_MIRROR=0` to a worker's task_env disables the mirror
   for every task on that worker.
3. Code revert: flip the call site to a no-op or revert the PR. The wiring
   is one block in `_prepare_container_config`.

## Open Questions

- **pre-warming**: rely on the daily ferry to populate the cache, or
  proactively pre-pull a fixed wheel set after creating each repo? My
  default is "rely on the ferry." Worth a reviewer take.
- **`marin-resiliparse` deferral**: confirmed accepted residual risk for
  v1, but if reviewers want to mirror it (one extra `--remote-python-custom-repo`
  pointing at the GitHub Pages URL), the marginal cost is low.
- **git+https deferral path forward**: if we later need it, the cleanest fix
  is a GCS-backed bare repo plus `git config url.<gcs>.insteadOf
  https://github.com/...` injected into the same env builder. Worth a
  thumbs-up that this is the right shape if/when we get there.
- **Defensive PyPI registrations**: `marin-dupekit` is the only commit
  in v1, but once we publish under the prefix, squatters can grab
  `marin-kitoken`, `marin-iris`, `marin-zephyr`, etc. Worth deciding
  whether to upload empty placeholders for those names alongside the
  `marin-dupekit` first release. Cheap, irreversible.
- **Cleanup policy: 30 vs 90 days.** Copied verbatim from `ghcr-mirror`,
  but PyPI wheels are smaller and re-pullable, and a redeploy after a
  quiet period would benefit from a longer retention. Reviewer take?
