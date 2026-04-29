# Spec — `iris_pypi_mirror`

Concrete contracts implied by [`design.md`](./design.md). Reviewers should be
able to read this and answer "yes, that's the API I'd accept."

## Public Python API

### `lib/iris/src/iris/cluster/providers/gcp/pypi_mirror.py`

New module, sibling to `bootstrap.py`.

```python
AR_PROJECT: str = "hai-gcp-models"
PYPI_MIRROR_REPO: str = "pypi-mirror"
PYTORCH_CPU_MIRROR_REPO: str = "pytorch-cpu-mirror"

_SUPPORTED_MULTI_REGIONS: frozenset[str] = frozenset({"us", "europe"})


def build_pypi_mirror_env(multi_region: str, project: str = AR_PROJECT) -> dict[str, str]:
    """Build uv env vars that point dependency resolution at AR remote PyPI repos.

    Caller responsibilities (not enforced here, kept out so the helper is
    trivially testable):
    1. Verify ``IRIS_WORKER_REGION`` is set and resolves via
       ``zone_to_multi_region`` to a non-None continent.
    2. Verify the user has not set ``IRIS_PYPI_MIRROR=0`` in
       ``EnvironmentConfig.env_vars``.

    Args:
        multi_region: AR multi-region location. Must be in {"us", "europe"};
            other values raise. Pass the result of ``zone_to_multi_region``.
        project: GCP project hosting the AR repos. Defaults to ``AR_PROJECT``.

    Returns:
        Three-key dict containing ``UV_DEFAULT_INDEX``, ``UV_INDEX_PYTORCH_CPU``,
        and ``UV_KEYRING_PROVIDER``. URL form:
        ``https://{multi_region}-python.pkg.dev/{project}/<repo>/simple/``.

    Raises:
        ValueError: if ``multi_region`` is not in ``_SUPPORTED_MULTI_REGIONS``.
    """
```

The function takes `project` as an argument so tests can pass a fake project
string and assert URL shape; the worker call site passes the constant.

### Wiring point: `lib/iris/src/iris/cluster/worker/task_attempt.py`

The injection lives in `_prepare_container_config` (around line 696), **not**
in `build_common_iris_env`. Reason: `IRIS_WORKER_REGION` is layered on by the
worker callsite, not inside `build_common_iris_env`. New block goes between
the existing region assignment and the user env merge:

```python
# Existing (unchanged):
region_attr = self._worker_metadata.attributes.get(WellKnownAttribute.REGION)
if region_attr and region_attr.string_value:
    env["IRIS_WORKER_REGION"] = region_attr.string_value

# NEW: PyPI mirror injection. Reads opt-out from the user-supplied
# EnvironmentConfig.env_vars directly, so user `IRIS_PYPI_MIRROR=0` in
# env_vars suppresses injection (the value also flows through to the task
# via the env.update on line 699).
mirror_disabled = self.request.environment.env_vars.get("IRIS_PYPI_MIRROR") == "0"
if region_attr and region_attr.string_value and not mirror_disabled:
    try:
        multi_region = zone_to_multi_region(region_attr.string_value)
    except ValueError:
        logger.info(
            "pypi mirror skipped: zone %s is unsupported (asia/me/etc.)",
            region_attr.string_value,
        )
        multi_region = None
    if multi_region is None:
        logger.info(
            "pypi mirror skipped: region %s has no AR continent mapping",
            region_attr.string_value,
        )
    else:
        env.update(build_pypi_mirror_env(multi_region))

# Existing (unchanged):
env.update(self._task_env)
env.update(dict(self.request.environment.env_vars))
```

Contract notes:

- **Opt-out lookup is `== "0"`**; any other value (including unset, `"1"`,
  `"true"`, `"false"`, `"off"`) enables the mirror. Single canonical sentinel
  matches existing IRIS_DEBUG_UV_SYNC convention.
- **Source of truth for opt-out**: `self.request.environment.env_vars` is a
  `dict[str, str]` populated from `EnvironmentConfig.env_vars`. Read it
  **before** any env merging via `self.request.environment.env_vars.get("IRIS_PYPI_MIRROR")`.
- **URL trailing slash**: env-var values end with `/simple/` (with the
  trailing slash). uv tolerates either form for `UV_DEFAULT_INDEX` but
  `UV_INDEX_<NAME>` requires the trailing slash on some uv versions —
  including it unconditionally is safe and the explicit form.
- **asia/me workers**: catch `ValueError` from `zone_to_multi_region` and
  skip the mirror with no warning surface change to the user. These workers
  fall back to public PyPI, same as today. Logged at INFO level via the
  module logger.
- **CoreWeave / no-region workers**: `region_attr` is unset → outer `if`
  short-circuits → skip silently (same code path as today's GHCR rewrite
  skip).
- **User can override mirror URLs**: because `env.update(self.request.environment.env_vars)`
  runs after our injection, a user setting `UV_DEFAULT_INDEX` directly in
  their job env_vars will win. This is intentional — escape hatch for
  debugging a mirror issue without committing a code change.
- **`build_common_iris_env` signature unchanged.** All new logic lives in
  the worker callsite.

## Env-var contract

The mirror sets exactly these three env vars on the task:

| Var | Value | Source / notes |
| --- | --- | --- |
| `UV_DEFAULT_INDEX` | `https://{mr}-python.pkg.dev/{project}/pypi-mirror/simple/` | Set by `build_pypi_mirror_env`. Replaces `pypi.org`. |
| `UV_INDEX_PYTORCH_CPU` | `https://{mr}-python.pkg.dev/{project}/pytorch-cpu-mirror/simple/` | Set by `build_pypi_mirror_env`. uv name-mangling: `[[tool.uv.index]] name="pytorch-cpu"` → env var `UV_INDEX_PYTORCH_CPU` (uppercase, hyphens → underscores). Overrides the named index in `lib/marin/pyproject.toml`. |
| `UV_KEYRING_PROVIDER` | `subprocess` | Set by `build_pypi_mirror_env`. Tells uv to call `keyrings.google-artifactregistry-auth`. |

Inputs (user-controlled, never set by this design):

| Var | Source | Behavior |
| --- | --- | --- |
| `IRIS_PYPI_MIRROR` | `EnvironmentConfig.env_vars` | `"0"` opts out; anything else (incl. unset) enables. Read once at task launch in `_prepare_container_config`. |

Env vars **not** set by this design: `UV_INDEX_URL`, `UV_EXTRA_INDEX_URL`,
`PIP_INDEX_URL`. We rely solely on `uv`'s typed overrides. `UV_CACHE_DIR` is
left at its existing value (the bind-mounted `/uv/cache` set in env.py:85
plus the worker mount in `task_attempt.py:715`); switching the default index
may cause uv to repopulate cache entries on first sync after rollout, which
is benign one-time cost.

## Infra contract — Artifact Registry

Repos to provision (one-time, idempotent):

| Location | Repo name | Mode | Upstream |
| --- | --- | --- | --- |
| `us` | `pypi-mirror` | remote | `--remote-python-repo=PYPI` |
| `us` | `pytorch-cpu-mirror` | remote | `--remote-python-custom-repo=https://download.pytorch.org/whl/cpu` |
| `europe` | `pypi-mirror` | remote | `--remote-python-repo=PYPI` |
| `europe` | `pytorch-cpu-mirror` | remote | `--remote-python-custom-repo=https://download.pytorch.org/whl/cpu` |

Project: `hai-gcp-models`. Repository format: `python`. Cleanup policy: keep
recent versions, delete after 30 days unused — same policy attached to
`ghcr-mirror`, copied verbatim.

IAM: the worker service account already used for `ghcr-mirror` gets
`roles/artifactregistry.reader` on each new repo. No new SA, no new role
binding scope.

The four `gcloud artifacts repositories create` invocations live in a script
under `lib/iris/scripts/setup_pypi_mirror.py` (parallels the existing
`setup_iam.py`). The script is idempotent: it re-runs cleanly if a repo
exists.

## Dockerfile changes

`lib/iris/Dockerfile`, `task` stage only. Add one package via uv:

```dockerfile
RUN uv pip install --system keyrings.google-artifactregistry-auth
```

The `controller` and `worker` stages do **not** install the keyring package —
they don't run user `uv sync`. The `deps` stage (which builds the base image
itself) also does not need it; that build runs on a developer/CI host with
its own auth.

**Keyring CLI on PATH inside the build container.** uv with
`UV_KEYRING_PROVIDER=subprocess` shells out to `keyring` (the Python
`keyring` package's CLI), passing the index hostname as the service. The
`keyrings.google-artifactregistry-auth` package registers itself as a
backend; once it's `uv pip install --system`'d into the image, the
`keyring` CLI on PATH discovers it automatically. **Validation gate
(end-to-end auth test, mandatory)**: before the rollout PR merges, run
the existing dev-cluster smoke job with `IRIS_PYPI_MIRROR=1` on a worker
whose SA has `roles/artifactregistry.reader` but **no cached
`gcloud auth`** (i.e. relies entirely on the GCE metadata server). The
task install must succeed and AR access logs must show the pull. This
catches three failure modes in one shot: keyring CLI not on PATH,
metadata-server token issuance broken, AR IAM grant missing.

**Token TTL note.** GCE metadata-server tokens have a ~1h TTL.
`keyrings.google-artifactregistry-auth` caches per-URL with a TTL just
under the token's lifetime, so a typical ~30 s `uv sync` is unaffected.
Long resolves (>1 h) may hit a mid-stream 401; current Marin syncs are
nowhere near that bound, but documented for future reference.

## Errors

No new exception types. Behavior changes:

- `build_pypi_mirror_env` raises `ValueError` if `multi_region` is not in
  `_SUPPORTED_MULTI_REGIONS` (`{"us", "europe"}`). Caller wraps via
  `zone_to_multi_region` so this only fires on programmer error (passing a
  string that wasn't run through the canonical mapper).
- `zone_to_multi_region` raising `ValueError` for `asia`/`me` is **caught**
  in the worker callsite and treated as "skip mirror, log INFO." These
  workers continue to resolve from public PyPI as today — no regression.
- `uv sync` failures during BUILDING now surface AR-side errors (4xx from
  AR, transport errors during an AR outage) instead of pypi.org-side errors.
  No change to how Iris reports them — same BUILDING-state task failure,
  same log surface. AR returns 404 for "package exists upstream but not yet
  cached" in some failure modes; this maps to the same uv error class as a
  pypi.org 404 today, no special handling.
- **First-cold fetch latency**: AR's pull-through proxies the upstream on
  first request, then caches. The first install of a freshly released
  wheel after a deps PR lands incurs the upstream fetch time *plus* a
  small AR proxying overhead — slightly slower than a direct
  `pypi.org` install. Steady-state (subsequent installs from cache) is
  faster. Acceptable; flagged so operators don't misread a slow
  first-install as a regression.

## PyPI publication for `marin-dupekit`

`dupekit` currently ships via a GitHub Release + `find-links` URL
(`pyproject.toml:28-31`); this puts github.com on the install hot path.
The AR mirror cannot proxy this because uv's `expanded_assets` is an
HTML-scraping URL handler, not a PEP 503 Simple index. We move dupekit to
PyPI as part of v1.

**Naming**: PyPI has no real namespaces (PEP 752 not shipped). We follow
the de-facto convention of hyphenated org prefixes (`google-cloud-*`,
`azure-*`) and publish as **`marin-dupekit`**. The Python *import* name
stays `dupekit` — the top-level Python package directory is `dupekit/`
(at `rust/dupekit/dupekit/`, governed by `python-source = "."` in
`rust/dupekit/pyproject.toml:27`); the Rust extension submodule is
`dupekit._native` (controlled by `module-name`, line 26). Renaming
`[project].name` to `marin-dupekit` changes only the *distribution* name
(wheel filename `marin_dupekit-<v>-...whl` per PEP 427 hyphen→underscore;
PyPI project URL); it does not move the `dupekit/` directory or break
`import dupekit`. Register `marin-kitoken`, `marin-iris`, etc.
protectively as the need arises — this design only commits to
`marin-dupekit`.

### Secrets posture

**No long-lived PyPI credential is stored anywhere in the repo, GitHub
secrets, or GitHub variables.** Trusted publishing (OIDC) replaces the
classic `PYPI_API_TOKEN` model — at workflow runtime, GitHub mints an
OIDC ID token, PyPI validates it against the configured publisher
binding, and PyPI issues a one-shot upload token (TTL ~15 min) that dies
when the workflow ends. There is no `PYPI_API_TOKEN` to rotate, leak, or
scope.

What lives where:

| Item | Storage location | Owner |
| --- | --- | --- |
| Per-admin PyPI password | each admin's personal password manager | individual |
| Per-admin PyPI 2FA TOTP seed | each admin's authenticator app | individual |
| **Per-admin PyPI 2FA recovery codes** | **shared team password vault (1Password)** | all org admins |
| PyPI org admin role grants | PyPI server-side (assigned via web UI) | n/a |
| Trusted publisher binding `(marin/dupekit-wheels.yaml, pypi-publish) ↔ marin-dupekit` | PyPI project settings (public, visible at the project's publishing page) | n/a |
| GitHub OIDC trust config | `.github/workflows/dupekit-wheels.yaml` (committed) | n/a |
| Per-run PyPI upload token | RAM only, ~15 min TTL | n/a |

The only thing that *must* be in a shared vault is the **2FA recovery
codes** for the PyPI account(s) that own the `marin-community` org —
needed if all admins lose their authenticator devices simultaneously.
Store these in the team password manager under
`marin-community / pypi-org-recovery`.

**Explicitly rejected**: storing a long-lived `PYPI_API_TOKEN` in GitHub
Actions secrets. It rotates poorly, leaks easily (any workflow on the
repo can read it via `${{ secrets.* }}`), needs manual revocation if
exposed, and gets clunky to scope per-package as we add `marin-kitoken`
etc. Trusted publishing is strictly better for our case.

**Compromise scenarios**:

- **Workflow file tampering**: an attacker who edits `dupekit-wheels.yaml`
  cannot publish without also bypassing the `pypi-publish` environment's
  required-reviewer gate (a Marin admin must click Approve on the GitHub
  Actions UI). Trust is anchored in three independent things: branch
  protection on `main`, the environment's reviewer requirement, and the
  publisher binding's pinning to a specific workflow filename + env name.
- **Compromised GitHub repo write access**: same — the reviewer gate is
  the last line of defence. Set the gate to "human admins only," not a
  bot account.
- **Compromised admin PyPI account**: revoke their org role on PyPI
  (`https://pypi.org/manage/organization/marin-community/people/`); their
  personal trusted-publisher bindings on other PyPI projects are
  unaffected. No shared credential to rotate.

### One-time setup (manual)

1. **Register the publishing PyPI organisation.** On
   `https://pypi.org/manage/organizations/` create an organisation account
   under the name `marin-community` (preferred — matches the GitHub org)
   with at least two human admins who hold the recovery email. The org is
   the owning entity for all `marin-*` packages; per-project trusted
   publishers are configured under it.
2. **Reserve `marin-dupekit`.** Confirm availability via
   `https://pypi.org/pypi/marin-dupekit/json` (404 = available). After the
   first OIDC-published release, the project will be created automatically
   and inherit ownership from the org. Optional defensive registrations:
   create empty placeholder uploads for `marin-kitoken`, `marin-iris`,
   `marin-zephyr`, etc. so squatters can't grab them. Out of scope here;
   noted for ops.
3. **Configure trusted publishing (OIDC).** On
   `https://pypi.org/manage/project/marin-dupekit/settings/publishing/`
   add a pending publisher with:
   - Repository owner: `marin-community`
   - Repository name: `marin`
   - Workflow filename: `dupekit-wheels.yaml`
   - Environment name: `pypi-publish` (matches the GitHub Actions
     environment we'll add for gating)
   No long-lived API tokens; PyPI mints a short-lived token per workflow
   run via OIDC. This avoids storing any static credential in the repo.
4. **Create the `pypi-publish` GitHub Actions environment** in
   `https://github.com/marin-community/marin/settings/environments` with:
   - Required reviewer: at least one Marin admin (so a release isn't a
     silent push). Optional in v1; reviewers should LGTM.
   - Branch protection: only `main` may deploy.

### Distribution name change

`rust/dupekit/pyproject.toml:6` changes from `name = "dupekit"` to
`name = "marin-dupekit"`. Maturin's `module-name = "dupekit._native"`
(line 26) is **unchanged** — this controls the import path, not the
distribution name. Verify after the rename that
`uv build --package marin-dupekit` produces wheels named
`marin_dupekit-<version>-...whl` and that `import dupekit` still works
inside a fresh venv.

### Workflow changes

`.github/workflows/dupekit-wheels.yaml` — change the `release` job:

- Add `environment: pypi-publish` and
  `permissions: { id-token: write, contents: write }` so OIDC is available
  and the existing `gh release create` keeps working.
- After the existing `python scripts/rust_package.py --skip-build` step
  (which still creates the GitHub Release for now — see "Cutover" below),
  add a step that calls `pypa/gh-action-pypi-publish@release/v1` pointing
  at the artifacts in `dist/` (already populated by the upstream
  `actions/download-artifact` step at line 72-75).

The PyPI publish step is gated identically to the existing release step
(`github.ref == 'refs/heads/main'` and `should_run == 'true'`), so PR
builds remain build-only.

### Version handling

PyPI disallows re-uploads at the same version. Each release must bump
`rust/dupekit/Cargo.toml:version` (semver: patch for fixes, minor for
features). The current release script (`scripts/rust_package.py:142-148`)
already reads version from `Cargo.toml`; no script change required for
versioning itself. **What does change**: the script's
`update_pyproject(tag, version)` step (line 318) currently rewrites
`find-links` to a new GitHub-Releases URL **and** the `dupekit >= ...`
version pin. After cutover the `find-links` rewrite is dropped and the
pin substitution targets `marin-dupekit >= <version>` instead.

### Cutover (single PR — atomic rename + consumer update)

The rename and consumer update **must land in a single PR**. If we land
the rename first (alone), the workflow produces wheels named
`marin_dupekit-*.whl` while the root `pyproject.toml` still pins
`dupekit >= 0.1.0` against the GitHub-Releases `find-links` URL — `uv
lock` on `main` breaks immediately. Atomic merge avoids the broken
intermediate state.

The PR contains:

1. **Producer rename**:
   - `rust/dupekit/pyproject.toml`: `name = "dupekit"` → `name = "marin-dupekit"`.
     `module-name = "dupekit._native"` unchanged. Bump
     `rust/dupekit/Cargo.toml:version` to a new patch (PyPI no-reupload).
2. **Workflow**:
   - `.github/workflows/dupekit-wheels.yaml`: add `environment: pypi-publish`,
     `permissions: { id-token: write, contents: write }`, and the
     `pypa/gh-action-pypi-publish@v1.12.4` step (pin a specific version, not
     `release/v1`, to avoid silent action breaking changes).
3. **Release script**:
   - `scripts/rust_package.py`: drop the `find-links` rewrite from
     `update_pyproject` (lines 318-345); change the version-pin regex to
     match `"marin-dupekit\s*>=\s*[^"]*"`.
4. **Consumer**:
   - Root `pyproject.toml`: replace `"dupekit >= 0.1.0"` (line 23) with
     `"marin-dupekit >= <new-version>"`; remove the dupekit `find-links`
     entry (line 29). Keep the `kitoken` line untouched.
5. **Lockfile**: run `uv lock`; the lockfile entry for the package should
   switch from
   `source = { registry = "https://github.com/.../expanded_assets/..." }`
   to `source = { registry = "https://pypi.org/simple" }` with dist name
   `marin-dupekit`.

**Pre-merge gate (mandatory)**: build the renamed package locally and run
the import smoke check. The reviewer can replicate:

```bash
uv build --package marin-dupekit
uv pip install --system dist/marin_dupekit-*.whl
python -c "import dupekit; from dupekit import _native; print(_native.__file__)"
```

This proves (a) the wheel name follows PEP 427 (`marin_dupekit-*.whl`),
(b) the top-level Python package is still importable as `dupekit`, and
(c) the Rust extension is loadable. Without this gate we only learn the
import works post-merge.

**Post-merge validation**: confirm the wheel + sdist appear on
`https://pypi.org/project/marin-dupekit/` once `main` runs the workflow,
and that an Iris smoke task with `IRIS_PYPI_MIRROR=1` resolves
`marin-dupekit` through AR (look for the AR domain in uv's `--verbose`
output, or in AR access logs).

The GitHub Release continues to be produced by the workflow for now (no
harm; humans may still want to download wheels off a release page).
Removal of the GitHub Release flow is a follow-up — out of scope here.

## File paths

| File | Status | Purpose |
| --- | --- | --- |
| `lib/iris/src/iris/cluster/providers/gcp/pypi_mirror.py` | NEW | `build_pypi_mirror_env`, repo-name constants, `AR_PROJECT` |
| `lib/iris/src/iris/cluster/worker/task_attempt.py` | MODIFY | Conditional env-var injection in `_prepare_container_config` after `IRIS_WORKER_REGION` is set |
| `lib/iris/Dockerfile` | MODIFY | `task` stage adds `keyrings.google-artifactregistry-auth` |
| `lib/iris/scripts/setup_pypi_mirror.py` | NEW | Idempotent provisioning script (4 repos + IAM grant + cleanup policy) |
| `lib/iris/docs/image-push.md` | MODIFY | One-paragraph cross-reference to the new mirror, or split into a sibling `pypi-mirror.md` |
| `lib/iris/tests/test_pypi_mirror.py` | NEW | Unit tests for `build_pypi_mirror_env`, opt-out behavior, region propagation |
| `rust/dupekit/pyproject.toml` | MODIFY | Rename `name = "dupekit"` → `name = "marin-dupekit"`. `module-name` unchanged (import name stays `dupekit`). |
| `.github/workflows/dupekit-wheels.yaml` | MODIFY | Add `environment: pypi-publish`, OIDC permissions, and `pypa/gh-action-pypi-publish` step in the `release` job |
| `scripts/rust_package.py` | MODIFY | Drop `find-links` rewrite from `update_pyproject` (lines 318-345); change version-pin regex to match `marin-dupekit`. |
| `pyproject.toml` | MODIFY | Rename top-level dep `dupekit >= 0.1.0` → `marin-dupekit >= 0.1.0` (line 23); remove the dupekit `find-links` entry (line 29). `kitoken` stays. |
| `uv.lock` | REGENERATED | After `pyproject.toml` edit, `uv lock` rewrites the registry source from GitHub-Releases to `pypi.org/simple` and the dist name to `marin-dupekit`. |

No changes to: any workspace `pyproject.toml` (only the root's
`find-links` entry shifts), or any other `lib/*/pyproject.toml`. Local dev
resolves dupekit from PyPI after cutover (same as any other public
package).

## Out of scope (explicit non-commitments)

- **`pytorch-cu128` mirroring.** GPU is rare on GCP; revisit if/when it
  isn't. Adding it later is one repo + one env var.
- **`marin-resiliparse` mirroring** (GitHub Pages index). Accepted residual
  risk; mirroring would be one `--remote-python-custom-repo` away.
- **`git+https://github.com/...` deps** (`harbor`, `lm-eval`). Evals using
  these will fail to install during a github.com outage. Future fix shape:
  GCS-backed bare repos + `git config url.<gcs>.insteadOf` injected by the
  same env builder.
- **CoreWeave workers.** No `IRIS_WORKER_REGION` continent match → no env
  vars injected → CoreWeave keeps resolving public `pypi.org`. They are
  unprotected during a PyPI outage; this is acceptable for v1.
- **Fallback to upstream on AR errors.** Explicitly rejected in `design.md`;
  hard-fail is the contract.
- **Pre-warming.** Implicit via daily ferry; no explicit pre-pull script.
- **First-class per-pool override** for `IRIS_PYPI_MIRROR`. Operators can
  already set `IRIS_PYPI_MIRROR=0` in a worker's `_task_env` (rollback
  lever #2 in `design.md`), but there is no dedicated config field;
  per-job is the only formal v1 surface.
- **`marin-kitoken` PyPI publication.** Lives in a separate repo
  (`marin-community/kitoken`) and only resolves under the `kitoken` extra,
  which Iris tasks don't pull. Same fix shape as dupekit — publish as
  `marin-kitoken` with trusted publishing, drop the `find-links` line at
  `pyproject.toml:30`. Deferred to a follow-up.
- **Removal of the dupekit GitHub Release flow.** The workflow keeps
  cutting GitHub Releases after PyPI cutover; pruning that path is a
  cleanup task, not part of v1.
