# Workflow Scripts Research

Issue: https://github.com/marin-community/marin/issues/5067

## Framing

Marin has accumulated enough GitHub Actions workflows that workflow purpose, naming, and local reproducibility are no longer obvious from `.github/workflows/`. The proposal under discussion is to make workflows thin triggers around reproducible Python scripts, likely under `scripts/workflows/`, using `scripts/rust_package.py` as a rough model. The intended outcome is consistent workflow and step naming, less duplicated YAML, fewer third-party logic actions where Marin can reasonably own the behavior, and workflow scripts that share a consistent CLI style based on Click and uv.

## Issue Context

Issue #5067 asks for a consistent naming scheme for workflows and tests, plus a "minimal YAML" policy where workflow logic is primarily defined in independently runnable Python scripts. The issue explicitly calls out consolidation across related workflows, such as one parameterized smoke test instead of separate GCP and CoreWeave implementations.

An existing issue comment triaged the current state as roughly 33-34 workflows and about 4.4K YAML lines. The main problems identified were naming drift, duplicated inline shell logic, and no index mapping workflow names to triggers and gate scale. The comment also proposed staged cleanup PRs, with open questions around domain-first versus type-first naming, allowed repo-infra domains, pytest marker rollout, branch-protection coordination, and whether GCP/CoreWeave smoke tests can share one helper.

Related issue #5065, the Infra Stability Epic, frames the broader goal as making Iris and Zephyr gates easier to reason about and operate.

## In-Repo Findings

### Existing Script Patterns

- `scripts/rust_package.py` is the strongest existing model for workflow-owned logic in Python. It documents usage and prerequisites at the top of the file, defines repo-root constants, runs subprocesses explicitly, and exposes clear command-line flags in `main()`:
  - `https://github.com/marin-community/marin/blob/53bdadbad1b1ae45d3aeaba3f33ab7f5d850359a/scripts/rust_package.py#L5`
  - `https://github.com/marin-community/marin/blob/53bdadbad1b1ae45d3aeaba3f33ab7f5d850359a/scripts/rust_package.py#L351`
- `.github/workflows/dupekit-wheels.yaml` already demonstrates a minimal-YAML pattern: the workflow checks out code, installs the toolchain, and delegates build/release behavior to `scripts/rust_package.py`.
  - `https://github.com/marin-community/marin/blob/53bdadbad1b1ae45d3aeaba3f33ab7f5d850359a/.github/workflows/dupekit-wheels.yaml#L56`
  - `https://github.com/marin-community/marin/blob/53bdadbad1b1ae45d3aeaba3f33ab7f5d850359a/.github/workflows/dupekit-wheels.yaml#L77`
- `scripts/python_libs_package.py` follows a similar release/build orchestration pattern for Python libraries.
- `scripts/canary/validate_canary_metrics.py` is already workflow-called validation logic and should be evaluated for reuse or migration under the proposed workflow-script layout.
- `scripts/gcp-ssh`, `scripts/iris/dev_tpu.py`, and `scripts/debug/decode_tokens.py` show existing Click-based command/group style in the repo.
- `scripts/logscan.py` is a current example of a standalone uv script using `#!/usr/bin/env -S uv run --script` and PEP 723 inline dependencies:
  - `https://github.com/marin-community/marin/blob/53bdadbad1b1ae45d3aeaba3f33ab7f5d850359a/scripts/logscan.py#L1`
  - `https://github.com/marin-community/marin/blob/53bdadbad1b1ae45d3aeaba3f33ab7f5d850359a/scripts/logscan.py#L6`

### Representative Pain Points

- Workflow filename and displayed-name conventions drift across `.yaml` and `.yml`, underscore and hyphen naming, and suffixes such as `run_tests`, `itest`, `launch_small_fast`, `unit-tests`, and plain `tests`.
- `marin-canary-ferry.yaml` includes a copied Iris job polling loop in YAML:
  - `https://github.com/marin-community/marin/blob/53bdadbad1b1ae45d3aeaba3f33ab7f5d850359a/.github/workflows/marin-canary-ferry.yaml#L101`
- The same workflow embeds a long failure-diagnostics shell block with `gcloud compute ssh`, Docker logs, and inline Python/SQLite:
  - `https://github.com/marin-community/marin/blob/53bdadbad1b1ae45d3aeaba3f33ab7f5d850359a/.github/workflows/marin-canary-ferry.yaml#L159`
- Similar submit/wait/diagnose patterns appear in `marin-datakit-smoke.yaml`, `zephyr-shuffle-itest.yaml`, `iris-cloud-smoke-gcp.yaml`, and `iris-coreweave-ci.yaml`.
- `dupekit-wheels.yaml` uses `peter-evans/create-pull-request@v7` for PR creation:
  - `https://github.com/marin-community/marin/blob/53bdadbad1b1ae45d3aeaba3f33ab7f5d850359a/.github/workflows/dupekit-wheels.yaml#L80`
  No action matching the example `rando-user/pull-request@7` was found in the current checkout.

### Tooling Constraints

- The repo requires Python `>=3.11`:
  - `https://github.com/marin-community/marin/blob/53bdadbad1b1ae45d3aeaba3f33ab7f5d850359a/pyproject.toml#L10`
- Root dependencies are intentionally kept small:
  - `https://github.com/marin-community/marin/blob/53bdadbad1b1ae45d3aeaba3f33ab7f5d850359a/pyproject.toml#L21`
- Current repo configuration excludes `scripts/` from ruff formatting/linting and pyrefly type checking:
  - `https://github.com/marin-community/marin/blob/53bdadbad1b1ae45d3aeaba3f33ab7f5d850359a/pyproject.toml#L75`
  - `https://github.com/marin-community/marin/blob/53bdadbad1b1ae45d3aeaba3f33ab7f5d850359a/pyproject.toml#L139`
  The design should decide whether new workflow scripts inherit that exclusion or define stricter checks.

## Related Designs

- `.agents/projects/ferry_framework.md` frames ferries as high-signal integration workflows and proposes launch/monitor helpers. Its run-record fields overlap with workflow-script output/reporting needs.
- `.agents/projects/20260212_iris_testing_design.md` motivates fast local Iris E2E testing and includes pytest marker conventions and `uv run pytest ...` examples that can inform gate naming.
- `.agents/projects/20260203_iris_job_name_design.md` is not workflow-specific, but is useful precedent for canonical naming and no-backward-compat migration framing.

## Prior Art

- GitHub reusable workflows are the native mechanism for reusing complete jobs or job graphs. They live directly in `.github/workflows/`, use `workflow_call`, and can be invoked from the same repo with `./.github/workflows/<file>`. GitHub positions them as the way to avoid copy-pasting whole workflow structures.
- GitHub distinguishes reusable workflows from composite actions: reusable workflows can contain jobs, choose runners, and show each job/step in logs, while composite actions package steps inside a job. For Marin's smoke/ferry orchestration, reusable workflows fit shared job topology; Python scripts fit reusable behavior inside a step.
- GitHub's security guidance says third-party actions can access secrets and repository tokens, and recommends pinning third-party actions to full-length SHAs, auditing action source, and using tags only for trusted creators. This supports treating third-party logic actions as design-reviewed dependencies rather than casual YAML conveniences.
- PEP 723 defines inline metadata for single-file Python scripts, including `requires-python`, `dependencies`, and optional `[tool]` metadata. This is the standard shape used by uv scripts.
- uv supports `uv run` for project-backed commands, `uv run --script` and uv shebangs for PEP 723 scripts, adjacent lockfiles for scripts, and `exclude-newer` metadata for stronger reproducibility.
- Click is the established Python CLI library for command groups, options, argument parsing, help output, and composable command-line interfaces. Using Click for all new workflow scripts would make local execution and GitHub Actions invocation consistent.

Sources:

- GitHub reusable workflows: https://docs.github.com/en/actions/how-tos/reuse-automations/reuse-workflows
- GitHub reusable workflows vs composite actions: https://docs.github.com/en/actions/sharing-automations/avoiding-duplication
- GitHub Actions secure use reference: https://docs.github.com/en/actions/security-for-github-actions/security-guides/security-hardening-for-github-actions
- uv scripts guide: https://docs.astral.sh/uv/guides/scripts/
- PEP 723: https://peps.python.org/pep-0723/
- Click documentation: https://click.palletsprojects.com/

## Design Questions

- Canonical workflow names should use domain-first `domain-type[-variant]` ordering. This matches the existing dominant convention and keeps team-owned workflows adjacent.
- `scripts/workflows/` should default to repo-local `uv run python scripts/workflows/...` scripts. Standalone PEP 723 uv scripts remain appropriate for tools that are intentionally usable outside a Marin checkout or environment, such as notification and ops helpers.
- New workflow scripts should be brought under ruff and pyrefly even though broader `scripts/**` remains excluded today.
- Native/trusted setup primitives are acceptable: `actions/*`, GitHub-owned actions, cloud setup/auth actions, Docker setup/login/build actions, and `astral-sh/setup-uv`. Other third-party actions should be pinned to full commit SHAs. No separate review registry is needed for this design; pinning is the policy.
- Repeated Iris submit/wait/diagnostics behavior should be exposed as Python scripts first. Reusable workflow consolidation can wait until script extraction shows whether duplicated job topology remains.
- Branch protection can be checked with `gh api repos/marin-community/marin/branches/main/protection`. As of this research pass, main requires these classic contexts: `lint-and-format`, `build-docs`, `marin-tests`, `levanter-tests`, `levanter-entry-tests`, `levanter-torch-tests`, `haliax-tests`, `iris-tests`, `zephyr-tests`, `fray-tests`, and `marin-itest`. The repo also has an active `protect main` ruleset, but the required status checks are visible through classic branch protection. If the token has admin permission, the contexts can be updated with `gh api --method PATCH repos/marin-community/marin/branches/main/protection`; workflow/job renames should include that command in the rollout step instead of hand-waving about manual settings.

## Resolved Design Decisions

- Use `ops` for repository automation workflows such as stale issue handling, CodeQL, nightshift jobs, Claude automation, and workflow maintenance. It is clearer than `repo` because these workflows are operations-owned automation, not a product domain.
- Do not create a shared package under `scripts/workflows/` at the start. Use concrete modules such as `scripts/workflows/iris_job.py`, `scripts/workflows/iris_diagnostics.py`, and `scripts/workflows/pull_request.py`. Add `scripts/workflows/lib/` only when at least two modules need the same non-trivial helper and the helper has a stable contract.
