# Releasing Packages to PyPI

Marin publishes ten distributions to [PyPI](https://pypi.org/). Two GitHub
Actions workflows build and publish them:

- [`dupekit-release-wheels.yaml`](https://github.com/marin-community/marin/blob/main/.github/workflows/dupekit-release-wheels.yaml)
  builds `marin-dupekit` (pure Python) and its native companion
  `marin-dupekit-native` (the Rust dedup kernels), in lockstep.
- [`marin-release-libs-wheels.yaml`](https://github.com/marin-community/marin/blob/main/.github/workflows/marin-release-libs-wheels.yaml)
  builds the eight pure-Python libs, driven by
  [`scripts/python_libs_package.py`](https://github.com/marin-community/marin/blob/main/scripts/python_libs_package.py).

The **distribution name** (what you `pip install`) carries a `marin-` prefix so
the names don't collide on PyPI, which has no namespaces. The **import name**
(what you `import`) is unchanged.

| Distribution | Import | Source |
| --- | --- | --- |
| `marin-core` | `marin` | `lib/marin` |
| `marin-iris` | `iris` | `lib/iris` |
| `marin-fray` | `fray` | `lib/fray` |
| `marin-haliax` | `haliax` | `lib/haliax` |
| `marin-levanter` | `levanter` | `lib/levanter` |
| `marin-rigging` | `rigging` | `lib/rigging` |
| `marin-zephyr` | `zephyr` | `lib/zephyr` |
| `marin-finelog` | `finelog` | `lib/finelog` |
| `marin-dupekit` | `dupekit` | `lib/dupekit` |
| `marin-dupekit-native` | `dupekit_native` | `lib/dupekit/rust` |

All publishing uses **OIDC trusted publishing**. There is no API token stored
in the repository, in GitHub secrets, or anywhere else. At workflow runtime
GitHub mints a short-lived OIDC token, PyPI validates it against the
publisher binding configured on each project, and PyPI issues a one-shot
upload token that expires when the run ends.

## How releases happen

| Trigger | Result |
| --- | --- |
| Daily schedule (06:00 UTC) | Nightly dev release: `<base>.dev<YYYYMMDDhhmm>`, published. |
| Push a `marin-libs-v<X.Y.Z>` tag | Stable release at version `<X.Y.Z>`, published. |
| `workflow_dispatch` (manual mode) | Build-only smoke; nothing is published. |
| Pull request touching the build script or workflow | Build-only smoke; nothing is published. |

`marin-dupekit` follows the same shape with `dupekit-v<X.Y.Z>` tags.

The eight libs always share one version per build, and each published wheel
pins its sibling `marin-*` dependencies to that exact version.

### Versioning

The `version` declared in each lib's `pyproject.toml` (or
`lib/haliax/src/haliax/__about__.py`) is only a **floor**. Nightlies are
resolved to one patch above `max(declared version, latest stable on PyPI)`,
so a `.dev` build always sorts above the current stable and `pip install
--pre` / `uv` prefer it. Because the script reads the latest stable from
PyPI, the declared version never needs re-bumping after a release.

To cut a stable release, pick the next [SemVer](https://semver.org/) version
and push the tag — no `pyproject.toml` edit required:

```bash
git tag marin-libs-v0.2.0
git push origin marin-libs-v0.2.0
```

PyPI rejects re-uploading an existing `(name, version)` pair, so every stable
release must use a fresh version.

## One-time PyPI setup

This is performed once by a PyPI admin who owns the `marin-community`
organization. It cannot be automated.

### 1. Organization

Ensure a `marin-community` [PyPI organization](https://pypi.org/manage/organizations/)
exists with at least two human admins. Every `marin-*` project is owned by it.

### 2. Clear the placeholder releases

Every `marin-*` lib already exists on PyPI with a single placeholder release:
`0.99` for seven of them (`marin-core`, `marin-iris`, `marin-fray`,
`marin-haliax`, `marin-levanter`, `marin-rigging`, `marin-zephyr`) and `0.1.0`
for `marin-finelog`. Delete that **release** from each project (project page →
*Manage project* → *Releases* → per-release *Options* → *Delete*). The project
itself stays — its name, ownership, and any configured trusted publisher are
preserved; the first real publish simply adds a new release.

PyPI permanently retires a deleted `(name, version)` pair — it can never be
re-uploaded. The libs therefore ship starting at **`0.2.0`**, which avoids
both retired placeholder versions (`0.99` and `0.1.0`), so no project has to
be deleted wholesale.

(`marin-core` is the distribution name for the top-level package; the
unprefixed `marin` name is not used.)

### 3. Configure a trusted publisher for each of the eight libs

Every project already exists (step 2 removes only the placeholder release,
not the project), so open each one's per-project publishing page and add the
publisher there:

```
https://pypi.org/manage/project/<name>/settings/publishing/
```

(If a project were ever fully deleted, you would instead add a **pending
publisher** from `https://pypi.org/manage/account/publishing/`, which creates
the project on the first matching upload — not needed here.)

Add a publisher with these values — identical for all eight bindings except
the project name:

| Field | Value |
| --- | --- |
| PyPI project name | one of `marin-core`, `marin-iris`, `marin-fray`, `marin-haliax`, `marin-levanter`, `marin-rigging`, `marin-zephyr`, `marin-finelog` |
| Repository owner | `marin-community` |
| Repository name | `marin` |
| Workflow filename | `marin-release-libs-wheels.yaml` |
| Environment name | `pypi-publish` |

Configure **all eight before the first publish run**. The publish job uploads
the whole `dist/` directory in one batch; a single missing binding fails the
upload partway and can poison a version (see [Troubleshooting](#troubleshooting)).

`marin-dupekit` and its native companion `marin-dupekit-native` each already
have a binding for the `dupekit-release-wheels.yaml` workflow and `pypi-publish`
environment; leave them untouched.

### 4. The `pypi-publish` GitHub Actions environment

Both release workflows publish through the `pypi-publish`
[deployment environment](https://github.com/marin-community/marin/settings/environments).
It already exists for `marin-dupekit`. Recommended settings:

- **Deployment branches and tags**: restrict to `main` and the release tags
  (`marin-libs-v*`, `dupekit-v*`).
- **No required reviewer**. The nightly runs unattended; a reviewer gate
  would block every nightly on a manual approval click. Trust is anchored by
  branch protection on `main` plus the publisher binding pinning a specific
  workflow filename and environment.

## Secrets posture

No long-lived PyPI credential exists in the repository, in GitHub secrets, or
in GitHub variables.

- The trusted-publisher binding is public information, shown on each PyPI
  project's publishing page.
- The GitHub OIDC trust config lives in the committed workflow files.
- The per-run PyPI upload token lives only in RAM and expires (~15 min) when
  the workflow ends.
- The only secret that must be shared is the **2FA recovery codes** for the
  PyPI account(s) that administer the `marin-community` organization. Store
  these in the team password vault.

Storing a classic `PYPI_API_TOKEN` in GitHub Actions secrets is explicitly
rejected: it rotates poorly, any workflow on the repo can read it, and it
needs manual revocation if exposed.

## Troubleshooting

- **`gh-action-pypi-publish` fails with 403 for one project.** Its
  trusted-publisher binding is missing or a field does not match. Re-check
  the four fields in step 3 — they must match the workflow exactly.
- **A version is "poisoned".** PyPI rejects re-uploads, so a run that
  uploaded some wheels and then failed leaves that version partially
  published and unrepeatable. Do not retry the same version: pick the next
  one and re-tag.
- **A nightly is not picked up by `pip install --pre`.** The dev build must
  sort above the latest stable. Confirm the stable on PyPI is not ahead of
  what `scripts/python_libs_package.py --resolve-only` computes.
