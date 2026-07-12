# Research — `iris_pypi_mirror`

Mitigation for github.com / pypi.org outages that block Iris task installation.
Linked issue: [marin#4897](https://github.com/marin-community/marin/issues/4897).

## TL;DR of findings

The user's instinct is correct but the term is imprecise: the right shape is a
**Google Artifact Registry (GAR) remote PyPI repo** plus per-named-index
remotes for the pytorch wheel servers, deployed once per continent. This is the
exact pattern the repo already uses for GHCR images (`ghcr-mirror`), and there
is already a detailed half-page proposal posted by an earlier agent run as a
comment on issue #4897 (2026-04-18) that we should treat as the starting
draft.

"GitHub" is mostly a red herring for v1: only two transitive `git+https`
deps exist (`harbor`, `lm-eval`), both behind optional extras. The cold-boot
failures we've actually seen are PyPI / pytorch wheel resolution. GAR has no
git remote, so true `git+https://github.com/...` mirroring would require
separate work and is **out of scope for the 90% solution**.

## In-repo references

### Current install path (what fails when PyPI/GitHub is down)

- `lib/iris/src/iris/cluster/runtime/entrypoint.py:44-109` — `build_runtime_entrypoint`
  emits the `uv sync --frozen --link-mode symlink` and `uv pip install` setup
  commands. No index flags are passed today; uv resolves against
  `pypi.org` plus whatever `[[tool.uv.index]]` entries each workspace
  `pyproject.toml` declares.
- `lib/iris/src/iris/cluster/runtime/docker.py:324-403` — `DockerRuntime.build()`
  runs the setup commands inside a build container during the BUILDING task
  state. This is the blocking hot path for cold boots.
- `lib/iris/src/iris/cluster/worker/task_attempt.py:567,618,740-759` — task
  attempts call `handle.build()` which executes the `uv sync` step.
- `lib/iris/Dockerfile:45-73` — base image build runs `uv sync --package
  marin-iris` against PyPI at `docker build` time. Pre-baked once, so this is
  not in the per-task hot path.

### Existing GHCR pull-through (the pattern to clone)

- `lib/iris/src/iris/cluster/providers/gcp/bootstrap.py:26-72` — defines
  `_ZONE_PREFIX_TO_MULTI_REGION = {"us": "us", "europe": "europe"}`,
  `zone_to_multi_region(zone)` mapping, `GHCR_MIRROR_REPO = "ghcr-mirror"`,
  and `rewrite_ghcr_to_ar_remote(image_tag, multi_region, project)`.
- `lib/iris/src/iris/cluster/providers/gcp/workers.py:282-293` —
  `resolve_image()` applies the rewrite at VM creation time; CoreWeave path
  bypasses.
- `lib/iris/docs/image-push.md` — narrative docs for the existing pattern,
  including the 30-day cleanup policy and IAM grants
  (`roles/artifactregistry.reader` on the worker SA).

### Indexes that need to be mirrored

From `lib/marin/pyproject.toml:189-217`:

- Default index: `pypi.org` (implicit).
- Named index `pytorch-cpu` → `https://download.pytorch.org/whl/cpu` (used by
  `cpu` and `tpu` extras — i.e. all our TPU workers).
- Named index `pytorch-cu128` → `https://download.pytorch.org/whl/cu128`
  (only for `--extra gpu` on Linux).
- Named index `marin-resiliparse` →
  `https://marin-community.github.io/chatnoir-resiliparse/simple` (small,
  hosted on GitHub Pages — at risk during a github.com outage).

### Env layer

- `lib/iris/src/iris/cluster/runtime/env.py:46-135` — `build_common_iris_env`
  is the single point where worker-derived env vars get layered onto a task
  attempt. `UV_PYTHON_INSTALL_DIR` is already set here; this is where
  `UV_DEFAULT_INDEX`, `UV_INDEX_*`, and `UV_KEYRING_PROVIDER` should go too.
  `IRIS_WORKER_REGION` is layered on by the worker path (path-specific
  addition mentioned in the docstring).

### git+https deps (out of scope for v1)

From `uv.lock`:

- `harbor` → `git+https://github.com/marin-community/harbor.git@354692d…`
  (only resolved with `--extra harbor`).
- `lm-eval` →
  `git+https://github.com/stanford-crfm/lm-evaluation-harness.git@d5e3391…`
  (only resolved with `--extra evalchemy`).

Iris core (controller / worker / task base) has no hard `git+https` deps.

## Prior work

- **Issue #4897 itself** has a thorough proposal in the agent comment
  (2026-04-18) — call out AR Python remote repos with `--remote-python-repo
  PYPI` for pypi.org and `--remote-python-custom-repo` for the pytorch wheel
  servers, IAM via existing worker SA + `keyrings.google-artifactregistry-auth`
  in the task base image, opt-in via `IRIS_PYPI_MIRROR=1` until soaked.
- **Issue #4571** ("Modal-like startup times") tracks broader cold-boot
  performance and quantifies the build phase at 5–30 s; mirror would shave
  the tail when public indexes throttle.
- No prior design doc in `.agents/projects/` on this topic.

## Prior art (web pass)

Consolidated digest:

- **Standard mitigations**: PyPI mirrors (`bandersnatch` full mirror,
  `devpi`/`proxpi` caching proxies), git mirrors (`git clone --mirror` to
  GCS, Gitea pull-mirror), container mirrors (GAR remote, ECR pull-through,
  Harbor proxy-cache projects).
- **GAR specifics**: remote repositories support PyPI (with custom upstream),
  Docker Hub, Maven, npm, Debian/CentOS. Virtual repos unify upstreams. **No
  native git remote, no GHCR upstream** — git+https URLs need a separate
  solution. Continent multi-region (`us`, `europe`) keeps egress free for
  same-continent VMs.
- **uv knobs**: `UV_DEFAULT_INDEX` overrides the default PyPI URL;
  `UV_INDEX_<name>` overrides each named `[[tool.uv.index]]` without editing
  source. `UV_KEYRING_PROVIDER=subprocess` lets uv shell out to
  `keyrings.google-artifactregistry-auth` for short-lived OAuth tokens off
  the GCE metadata server.
- **uv fallback**: uv does **not** auto-fall-back across indexes on transport
  errors. A dead AR endpoint halts resolution, so the mirror cannot be wired
  as a "best-effort" layer in front of pypi.org without manual logic.
- **Failure modes others report**: stale snapshots after upstream
  yank/force-push, cross-region egress costs, cold cache hit on first pull
  during an outage (mirror only caches what's been requested once
  successfully — pre-warming matters).
- **Reference implementations**: `proxpi`, Pulp (Red Hat), StableBuild (pinned
  PyPI snapshots), public ECR pull-through writeups for EKS.

## Open decisions for design

These need user input before drafting:

1. **Default-on vs opt-in.** The earlier agent comment proposed
   `IRIS_PYPI_MIRROR=1` opt-in until soak. Default-on gets us 100% of the
   resilience benefit immediately but means a misconfigured AR remote takes
   down all task installs. Opt-in is conservative but does nothing until
   we flip the flag.
2. **Coverage**: mirror PyPI + `pytorch-cpu` (cheap, covers all TPU jobs)
   only, vs. also `pytorch-cu128` (large wheels, mostly used by GPU jobs)?
   `marin-resiliparse` is on GitHub Pages — strictly speaking that's at
   risk in a github outage too, but it's a tiny package; could either mirror
   it (custom upstream URL) or accept the risk.
3. **Git mirror scope.** Confirm we are deferring `git+https://github.com/...`
   coverage entirely for v1. The two affected deps (`harbor`, `lm-eval`) sit
   behind optional extras, so the failure mode is "evals can't start during
   a github outage" — acceptable for the 90% solution, but worth naming
   explicitly in the doc.
4. **Failure semantics** when AR is unreachable. Since uv won't auto-fall-
   back, options are (a) accept that an AR outage now blocks task installs
   (added blast radius), (b) wrap uv with a retry/fallback shim that flips
   to upstream after N failures, (c) keep the flag opt-in so the user can
   disable per-job. (a) is by far the simplest.
