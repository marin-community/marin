# Iris native release contract

## Decision

Treat the Python package requirement and `uv.lock` as the deployment contract.
Publishing a wheel is necessary but does not update a checkout that installs
from the repository lockfile.

For a Python change that requires a new native API:

1. Publish a native nightly from the PR commit with the
   `iris-native-release-wheels.yaml` workflow.
2. In the same PR, raise the `marin-iris-native` lower bound to that nightly and
   refresh `uv.lock`. This makes the PR testable and makes its eventual merge
   safe to deploy.
3. After the PR merges, create the next `iris-native-vX.Y.Z` tag on the commit
   in `main`. The tag publishes the stable wheels.
4. Refresh the requirement and lockfile to the stable version in a small
   follow-up PR.

Do not tag an unmerged PR commit. Marin PRs are squash-merged, so such a tag
would permanently identify a commit that is not an ancestor of `main`.

## Current state

- PyPI stable: `marin-iris-native==0.1.0`.
- PR #7598 nightly:
  `marin-iris-native==0.1.1.dev202607241746`, published from rebased commit
  `db5ef9411c`.
- Before this PR's dependency update, `lib/iris/pyproject.toml` accepted
  `marin-iris-native >= 0.1.0.dev0` and the root `uv.lock` selected `0.1.0`.
- PR #7598 requires the new `NativeProxy.rpc_metrics_json` property, so allowing
  or locking `0.1.0` is not valid after this PR merges.

Release workflow run
[#30114319841](https://github.com/marin-community/marin/actions/runs/30114319841)
published the PR nightly for macOS arm64, Linux x86_64, and Linux aarch64,
plus its source distribution.

## Why the two-stage release is preferable

The release workflow supports scheduled nightlies, explicit stable releases,
and stable tags. A workflow dispatch from an unmerged branch could publish
`0.1.1` immediately, but that would make the stable artifact's source revision
an unmerged PR commit and leave no canonical stable tag on `main`.

Using a PR nightly first keeps the dependency update atomic:

- CI and reviewers consume the exact native implementation required by the
  Python code.
- A checkout of the merged commit receives a compatible wheel through
  `uv sync`.
- The stable tag remains a statement about code present in `main`.

The temporary nightly pin is operationally harmless. Stable `0.1.1` sorts after
the `0.1.1.dev...` build under PEP 440, so the follow-up lock refresh naturally
replaces it.

## Release procedure

### Before merge

1. Rebase the PR onto `origin/main` and run native wheel CI.
2. Dispatch `iris-native-release-wheels.yaml` on the PR branch in `nightly`
   mode.
3. Wait for all three platform wheels and the source distribution to publish.
4. Verify the new version on the PyPI simple index.
5. Set the Iris dependency floor to that exact nightly and run:

   ```bash
   uv lock --upgrade-package marin-iris-native
   uv sync --all-packages
   ```

6. Confirm the lockfile selects the new nightly and rerun the Python tests that
   exercise the native proxy.

### After merge

1. Fetch `main` and identify the merge commit containing PR #7598.
2. Choose the next unused stable version, currently `0.1.1`.
3. Create and push an annotated `iris-native-v0.1.1` tag on that `main` commit.
4. Wait for the release workflow and verify all distributions on PyPI.
5. Open a dependency-only PR that changes the floor to
   `marin-iris-native >= 0.1.1` and refreshes `uv.lock`.

## Longer-term improvement

The stable workflow should open the dependency-only PR automatically after a
successful tag publication. Until that exists, every native stable release has
two explicit completion conditions: the distributions are visible on PyPI, and
the repository lockfile selects the new version.
