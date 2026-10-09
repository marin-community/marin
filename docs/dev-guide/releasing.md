# Releasing Packages to PyPI

The [package release workflow](https://github.com/marin-community/marin/blob/main/.github/workflows/marin-release-libs-wheels.yaml)
builds and publishes Marin distributions to PyPI. Its release registry,
`PACKAGES` in `scripts/ci/package_release.py`, defines the distributions,
source paths, build jobs, and release families. Distribution names start with
`marin-`; Python import names come from each package's manifest.

The general Python family shares one version per release. Each wheel pins its
Marin dependencies in that family to the exact release version. The root uv
workspace supplies editable packages for development; published wheels depend
on PyPI packages. Native companions have separate build jobs and versions.

Publishing uses PyPI trusted publishers and GitHub OIDC. The workflow does not
store a PyPI API token.

## How releases happen

| Trigger | Result |
| --- | --- |
| Native source change merged to `main` | Package-specific dev release, followed by an automatic compatibility-floor and `uv.lock` pull request. |
| Daily pure-library schedule (06:00 UTC) | Coupled pure-library development release, published. |
| Push a package release tag | Stable release at the tag version, published. |
| `workflow_dispatch` (manual mode) | Build-only smoke; nothing is published. |
| Pull request touching package build inputs | Build the selected family; native wheels are installed and imported; nothing is published. |

Stable tags are `marin-libs-v<X.Y.Z>`, `dupekit-v<X.Y.Z>`,
`finelog-v<X.Y.Z>`, and `iris-native-v<X.Y.Z>`.

Native implementation pull requests compile their changed Rust sources in
`unified-unit` and in the package release workflow. The follow-up dependency
pull request changes only the consumer compatibility floor and `uv.lock`, so
CI exercises the newly published wheels before they become the repository
default. The shared update branch serializes releases from different native
packages and uses a GitHub App token so its pull request triggers normal CI.

To add a distribution, update `PACKAGES` and configure its trusted publisher
for this workflow. Add a build driver only if the package uses a new build
system.

### Versioning

Declared package versions are floors. Development releases use one patch above
the greatest supported declared or published version in the family. The
GitHub run ID is the unique, retry-stable development serial. Looking across
every distribution in a coupled family keeps a legacy development release or
one diverged project from sorting above the new release.

After PyPI accepts a complete native family release, automation raises its
consumer dependency floor and locks that exact registry version. A targeted
lock validation rejects unrelated package churn. The general Python family
does not need a follow-up pull request because the checkout resolves its
workspace packages locally. Published wheels refer to PyPI distributions.

To cut a stable release, pick the next [SemVer](https://semver.org/) version
and push the tag — no `pyproject.toml` edit required:

```bash
git tag marin-libs-v0.2.0
git push origin marin-libs-v0.2.0
```

PyPI rejects re-uploading an existing `(name, version)` pair, so every stable
release must use a fresh version.

## One-time PyPI setup

An admin configures each new PyPI project before its first release.

### 1. Organization

Create each project under the
[`marin-community` PyPI organization](https://pypi.org/manage/organizations/).
Keep at least two human organization admins.

### 2. Configure a trusted publisher for each distribution

Open each project's publishing page and add the publisher there:

```
https://pypi.org/manage/project/<name>/settings/publishing/
```

(A pending publisher can also create a project on first upload, but it does
not reserve the project name or assign it to the organization. Create any new
project under the organization first.)

Add a publisher with these values, choosing the workflow that publishes the
project:

| Field | Value |
| --- | --- |
| PyPI project name | the distribution name in `PACKAGES` |
| Repository owner | `marin-community` |
| Repository name | `marin` |
| Workflow filename | `marin-release-libs-wheels.yaml` |
| Environment name | `pypi-publish` |

Every distribution produced by the workflow needs its own binding. The publish
job remains in this top-level workflow because
[PyPI Trusted Publishing does not currently support naming a reusable workflow](https://docs.pypi.org/trusted-publishers/troubleshooting/#reusable-workflows-on-github)
as the publisher.

When moving a project from an older release workflow, update its publisher
binding to this workflow and the `pypi-publish` environment.

### 3. The `pypi-publish` GitHub Actions environment

The release workflow publishes through the `pypi-publish`
[deployment environment](https://github.com/marin-community/marin/settings/environments).
It already exists for `marin-dupekit`. Recommended settings:

- **Deployment branches and tags**: restrict to `main` and the release tags
  (`marin-libs-v*`, `dupekit-v*`, `finelog-v*`, `iris-native-v*`).
- **No required reviewer**. Automated releases run unattended; a reviewer gate
  would block them on a manual approval click. Trust is anchored by
  branch protection on `main` plus the publisher binding pinning a specific
  workflow filename and environment.

## Secrets posture

The workflow obtains a short-lived upload token through OIDC. Do not add a
`PYPI_API_TOKEN` GitHub secret. Store organization admin 2FA recovery codes in
the team password vault.

## Troubleshooting

- **`gh-action-pypi-publish` fails with 403 for one project.** Its
  trusted-publisher binding is missing or a field does not match. Re-check
  the fields in step 2; they must match the workflow exactly.
- **A family release stopped after some files uploaded.** Rerun the failed
  jobs in the same workflow run. The version is stable across reruns, and the
  preflight accepts only remote files whose hashes match the complete family
  manifest. The publisher skips those matching files and uploads the rest.
- **A development release is not picked up by `pip install --pre`.** The dev build must
  sort above the latest stable. Confirm the stable on PyPI is not ahead of
  what `scripts/ci/package_release.py plan` computes for the family.
