# Migration Plan: Levanter TPU Workflows → Marin TPU Testing Environment

**Date:** 2025-11-04
**Status:** Planning Phase

## Executive Summary

This document outlines a plan to migrate Levanter's TPU testing workflows from on-demand GCP TPU VM creation to Marin's self-hosted TPU CI infrastructure. This migration will reduce workflow complexity, improve test execution speed, and reduce costs.

**Key Metrics:**
- Current: 19 workflow files
- Target: 12-14 workflow files (reduction of 5-7 files)
- Expected benefits: Faster tests, lower costs, easier maintenance

## Current State Analysis

### Levanter Workflows (11 files)

#### TPU Workflows (3 files) - **TARGET FOR MIGRATION**
1. **`levanter-tpu_unit_tests.yaml`**
   - Creates/destroys TPU VMs on-demand using GCP
   - Matrix testing across:
     - JAX versions: 0.6.2, 0.7.2, 0.8.0
     - TPU types: v5p-8, v4-8, v5litepod-4, v6e-8
     - Zones: us-central1-a, us-central2-b, europe-west4-a, europe-west4-b
   - Uses `infra/spin-up-vm.sh` for VM creation
   - Runs tests with: `pytest lib/levanter/tests -m 'not entry and not ray'`

2. **`levanter-gpt2_small_itest.yaml`**
   - Uses `infra/launch.py` to create v4-32 TPU VM
   - Runs GPT-2 small integration test
   - Uses Docker image from GitHub Container Registry

3. **`levanter-launch_small_fast.yaml`**
   - Launches training jobs on v4-32 TPUs
   - Uses `infra/launch.py` for orchestration
   - Runs actual training workloads (not just tests)

#### CPU Test Workflows (3 files) - **TARGET FOR CONSOLIDATION**
4. **`levanter-run_tests.yaml`**
   - Runs on GitHub Actions ubuntu runners
   - Matrix: python 3.11, JAX 0.6.2/0.7.2/0.8.0
   - Tests: `pytest tests -m "not entry and not slow and not ray"`

5. **`levanter-run_ray_tests.yaml`**
   - Runs Ray-specific tests
   - Single version (python 3.11)
   - Tests: `pytest tests -m "ray"`

6. **`levanter-run_entry_tests.yaml`**
   - Runs entry point tests
   - Single version (python 3.11)
   - Tests: `pytest tests -m "entry"`

#### Docker Workflows (2 files) - **TARGET FOR CONSOLIDATION**
7. **`levanter-docker-base-image.yaml`**
   - Builds base Docker image for TPU
   - Pushes to GitHub Container Registry (ghcr.io)
   - Tagged as `levanter-base:latest` and `levanter-base:YYYYMMDD`

8. **`levanter-docker-cluster-image.yaml`**
   - Builds cluster Docker image
   - Depends on base image workflow
   - Tagged as `levanter-cluster:latest`

#### Other Workflows (3 files) - **KEEP AS-IS**
9. **`levanter-run_pre_commit.yaml`** - Pre-commit hooks
10. **`levanter-check_lockfile.yaml`** - uv.lock validation
11. **`levanter-publish_dev.yaml`** - PyPI dev builds

### Marin Workflows (8 files)

1. **`marin-unit-tests.yaml`** - **KEY REFERENCE**
   - Has two jobs:
     - `test`: CPU tests on ubuntu runners
     - `tpu-test`: **Runs on `[tpu-ci]` self-hosted runners**
   - TPU test configuration:
     - Uses Docker: `$REGION-docker.pkg.dev/hai-gcp-models/marin-ci/tpu-ci:latest`
     - Mounts `/dev/vfio` for TPU access
     - Sets `JAX_PLATFORMS=tpu`, `PJRT_DEVICE=TPU`
     - Runs: `pytest tests/tpu -m tpu_ci`

2. **`marin-build-docker-images.yaml`** - **TARGET FOR CONSOLIDATION**
   - Builds `marin_cluster` image
   - Pushes to multiple regional Google Artifact Registries
   - Regions: europe-west4, us-central1, us-central2, us-east1, us-east5, us-west4

3-8. Other Marin workflows (docs, metrics, quickstart, lint, CodeQL, leaderboard)

9. **`stale.yml`** - Shared workflow for stale issue management

### Key Infrastructure Files

**Docker Images:**
- `docker/marin/Dockerfile.tpu-ci` - TPU CI runner image (includes GitHub Actions runner + uv + dependencies)
- `docker/marin/Dockerfile.cluster` - Marin cluster deployment image
- `docker/levanter/Dockerfile.base` - Levanter base TPU image
- `docker/levanter/Dockerfile.incremental` - Levanter incremental TPU image
- `docker/levanter/Dockerfile.cluster` - Levanter cluster image

**Test Markers:**
- Marin tests use `@pytest.mark.tpu_ci` for TPU-specific tests
- Levanter tests use `@pytest.mark.entry`, `@pytest.mark.ray`, `@pytest.mark.slow`

## Key Differences: Levanter vs Marin TPU Testing

| Aspect | Levanter (Current) | Marin (Target) |
|--------|-------------------|----------------|
| **Infrastructure** | On-demand GCP TPU VMs | Persistent self-hosted `[tpu-ci]` runners |
| **Test Execution** | SSH into VM, run tests | Docker container with TPU access |
| **Startup Time** | 5-10 minutes (VM creation) | <1 minute (runner already running) |
| **Cost** | Pay per VM creation/hour | Fixed cost (persistent runners) |
| **Flexibility** | Any TPU type/zone | Limited to runner's TPU type |
| **Cleanup** | Must delete VMs (or pay forever) | Automatic (container cleanup) |
| **Use Case** | Good for training jobs | Perfect for unit/integration tests |

## Proposed Consolidated Structure

### New/Modified Workflows

#### 1. **`levanter-tests.yaml`** (NEW - Consolidates 4 workflows)

Replaces:
- `levanter-run_tests.yaml`
- `levanter-run_ray_tests.yaml`
- `levanter-run_entry_tests.yaml`
- `levanter-tpu_unit_tests.yaml`

```yaml
name: Levanter - Tests

on:
  push:
    branches: [main]
    paths:
      - lib/levanter/**
      - uv.lock
      - .github/workflows/levanter-tests.yaml
  pull_request:
    paths:
      - lib/levanter/**
      - uv.lock
      - .github/workflows/levanter-tests.yaml

jobs:
  cpu-unit-tests:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11"]
        jax-version: ["0.6.2", "0.7.2", "0.8.0"]
    steps:
      # Run: pytest tests -m "not entry and not slow and not ray and not tpu"

  cpu-ray-tests:
    runs-on: ubuntu-latest
    steps:
      # Run: pytest tests -m "ray"

  cpu-entry-tests:
    runs-on: ubuntu-latest
    steps:
      # Run: pytest tests -m "entry"

  tpu-tests:
    runs-on: [tpu-ci]  # Self-hosted TPU CI runners
    if: github.event.pull_request.head.repo.full_name == github.repository
    strategy:
      fail-fast: false
      matrix:
        jax_version: ["0.6.2", "0.7.2", "0.8.0"]
    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Run TPU tests in Docker
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          # Similar to marin-unit-tests.yaml tpu-test job
          # But installs specific JAX version: uv run --with "jax[tpu]==${{ matrix.jax_version }}"
          # Runs: pytest lib/levanter/tests -m "tpu_ci"
```

**Benefits:**
- Single workflow for all Levanter tests
- Clear CPU vs TPU separation
- Maintains JAX version compatibility testing
- Uses efficient self-hosted infrastructure
- Eliminates need to create/destroy VMs

#### 2. **`docker-images.yaml`** (NEW - Consolidates 3 workflows)

Replaces:
- `levanter-docker-base-image.yaml`
- `levanter-docker-cluster-image.yaml`
- `marin-build-docker-images.yaml`

```yaml
name: Build Docker Images

on:
  workflow_dispatch:
  push:
    branches: [main]
    paths:
      - docker/**
      - uv.lock

jobs:
  marin-cluster-images:
    runs-on: ubuntu-latest
    steps:
      # Current marin-build-docker-images.yaml logic
      # Builds: marin_cluster (multi-region)

  marin-tpu-ci-image:
    runs-on: ubuntu-latest
    steps:
      # Build: docker/marin/Dockerfile.tpu-ci
      # This is NEW - currently not automated

  levanter-tpu-images:
    runs-on: ubuntu-latest
    steps:
      # Current levanter docker workflows logic
      # Builds: levanter-base, levanter-tpu, levanter-cluster
```

**Benefits:**
- Single place for all Docker builds
- Easier to trigger rebuilds
- Can add dependencies between jobs if needed
- Clear organization by project

#### 3. **`levanter-gpt2_small_itest.yaml`** (MODIFIED)

Current approach: Creates v4-32 VM using `infra/launch.py`

**Option A: Migrate to `[tpu-ci]` runners** (if runners have v4-32 or similar)
```yaml
jobs:
  integration_test:
    runs-on: [tpu-ci]
    steps:
      # Run in Docker container
      # No VM creation needed
```

**Option B: Keep VM creation but simplify** (if `[tpu-ci]` runners are smaller TPUs)
```yaml
jobs:
  integration_test:
    runs-on: ubuntu-latest
    steps:
      # Keep current approach for larger TPU needs
      # But streamline the script
```

#### 4. **`levanter-launch_small_fast.yaml`** (KEEP)

**Recommendation: Keep as-is**
- This is for actual training jobs, not just tests
- Needs v4-32 or larger TPUs
- On-demand creation makes sense here

### Workflows to Keep Unchanged

- `levanter-run_pre_commit.yaml`
- `levanter-check_lockfile.yaml`
- `levanter-publish_dev.yaml`
- `marin-unit-tests.yaml`
- `marin-quickstart.yaml`
- `marin-lint-and-format.yaml`
- `marin-docs.yaml`
- `marin-metrics.yaml`
- `marin-update-leaderboard.yml`
- `marin-codeql.yml`
- `stale.yml`

## Detailed Migration Steps

### Phase 1: Prepare Levanter Tests for TPU CI

**Step 1.1: Add TPU CI pytest markers to Levanter tests**
- Identify which tests need TPU hardware
- Add `@pytest.mark.tpu_ci` decorator
- Ensure tests work in Docker environment
- Test locally if possible

**Step 1.2: Update TPU CI Docker image**
- Modify `docker/marin/Dockerfile.tpu-ci` to include Levanter dependencies
- Add Levanter to the workspace sync: `uv sync --frozen --extra tpu --extra gcp --package levanter`
- Test image builds successfully
- Deploy to artifact registry

**Affected files:**
- `lib/levanter/tests/**/*.py`
- `docker/marin/Dockerfile.tpu-ci`

### Phase 2: Create Consolidated Test Workflow

**Step 2.1: Create `levanter-tests.yaml`**
- Create new workflow file
- Implement `cpu-unit-tests` job (from `levanter-run_tests.yaml`)
- Implement `cpu-ray-tests` job (from `levanter-run_ray_tests.yaml`)
- Implement `cpu-entry-tests` job (from `levanter-run_entry_tests.yaml`)
- Implement `tpu-tests` job (adapted from `marin-unit-tests.yaml` + matrix)

**Step 2.2: Test new workflow**
- Create test PR to trigger workflow
- Verify all jobs run successfully
- Verify JAX version matrix works for TPU tests
- Verify test markers work correctly

**Affected files:**
- `.github/workflows/levanter-tests.yaml` (NEW)

### Phase 3: Consolidate Docker Workflows

**Step 3.1: Create `docker-images.yaml`**
- Create new unified workflow
- Implement `marin-cluster-images` job (from `marin-build-docker-images.yaml`)
- Implement `marin-tpu-ci-image` job (NEW - automates TPU CI image builds)
- Implement `levanter-tpu-images` job (from Levanter docker workflows)
- Set up proper job dependencies if needed

**Step 3.2: Test Docker builds**
- Trigger workflow manually
- Verify all images build successfully
- Verify images are pushed to correct registries
- Test that TPU CI runners can pull the new image

**Affected files:**
- `.github/workflows/docker-images.yaml` (NEW)

### Phase 4: Update Integration Test Workflow

**Step 4.1: Decide on approach for `levanter-gpt2_small_itest.yaml`**
- Determine if `[tpu-ci]` runners have sufficient TPU resources
- If yes: Migrate to self-hosted runners
- If no: Keep VM creation but document why

**Step 4.2: Implement chosen approach**
- Modify workflow file
- Test integration test runs successfully
- Document any limitations

**Affected files:**
- `.github/workflows/levanter-gpt2_small_itest.yaml` (MODIFIED)

### Phase 5: Cleanup and Documentation

**Step 5.1: Remove deprecated workflows**

Delete the following files:
- `.github/workflows/levanter-run_tests.yaml`
- `.github/workflows/levanter-run_ray_tests.yaml`
- `.github/workflows/levanter-run_entry_tests.yaml`
- `.github/workflows/levanter-tpu_unit_tests.yaml`
- `.github/workflows/levanter-docker-base-image.yaml`
- `.github/workflows/levanter-docker-cluster-image.yaml`
- `.github/workflows/marin-build-docker-images.yaml`

**Step 5.2: Update documentation**
- Update README or contributing guide
- Document new workflow structure
- Document how to run TPU tests locally/in CI
- Document JAX version testing strategy

**Step 5.3: Update workflow triggers**
- Ensure workflows that depend on "Run Tests" workflow now depend on "Levanter - Tests"
- Update `levanter-publish_dev.yaml` dependencies
- Update any other workflow_run triggers

**Affected files:**
- `README.md` or `CONTRIBUTING.md`
- `.github/workflows/levanter-publish_dev.yaml` (workflow_run dependency)
- Any other workflows with dependencies

## Expected Outcomes

### Quantitative Improvements

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| **Total workflow files** | 19 | 12-14 | -5 to -7 files |
| **Levanter test workflows** | 4 separate files | 1 consolidated file | -3 files |
| **Docker workflows** | 3 separate files | 1 consolidated file | -2 files |
| **TPU test startup time** | 5-10 minutes | <1 minute | ~90% faster |
| **TPU VM cost** | Per-test VM creation | Fixed runner cost | Lower (amortized) |

### Qualitative Improvements

✅ **Faster CI feedback loops**
- TPU tests start immediately (no VM creation wait)
- Developers get test results much faster

✅ **More cost-efficient**
- No per-test VM creation costs
- Reuses persistent TPU CI infrastructure
- Reduced GCP API calls

✅ **Cleaner workflow organization**
- Single file for all Levanter tests
- Clear separation of CPU vs TPU tests
- Easier to understand what tests run when

✅ **Easier maintenance**
- Fewer workflow files to update
- Centralized Docker build configuration
- Consistent TPU testing approach across projects

✅ **Better resource utilization**
- Leverages existing Marin TPU CI infrastructure
- Unified Docker image management
- Shared TPU testing patterns

✅ **Maintains flexibility**
- Still tests multiple JAX versions
- Still can create VMs for training jobs when needed
- Still have separate CPU/TPU test paths

## Special Considerations

### JAX Version Matrix Testing

**Challenge:** Levanter tests against 3 JAX versions (0.6.2, 0.7.2, 0.8.0) but TPU CI Docker image has one frozen version.

**Solution:** Use `uv run --with "jax[tpu]==X.Y.Z"` to test different versions:
```bash
uv run --with "jax[tpu]==0.6.2" pytest lib/levanter/tests -m tpu_ci
```

**Trade-off:**
- Slower than pre-installed versions
- But still faster than creating VMs
- Can optimize later by pre-installing multiple JAX versions in Docker image

### Large TPU Requirements

**Current:** `levanter-launch_small_fast.yaml` uses v4-32 TPUs

**Decision tree:**
```
Is this a unit/integration test?
├─ Yes → Use [tpu-ci] self-hosted runners
└─ No → Is this a training job?
   ├─ Yes → Use on-demand VM creation (infra/launch.py)
   └─ No → Re-evaluate requirements
```

**Recommendation:**
- Keep `levanter-launch_small_fast.yaml` for training jobs
- Keep VM creation infrastructure (infra/launch.py, spin-up-vm.sh)
- Use self-hosted runners only for tests

### Docker Image Strategy

**TPU CI Image** (`docker/marin/Dockerfile.tpu-ci`):
- Purpose: Run tests on self-hosted TPU CI runners
- Should include: Both Marin and Levanter test dependencies
- Update frequency: When dependencies change
- Location: Google Artifact Registry (multiple regions)

**Levanter TPU Images** (`docker/levanter/Dockerfile.*`):
- Purpose: Deploy training jobs, run on TPU VMs
- Should include: Levanter + dependencies optimized for training
- Update frequency: After successful test runs
- Location: GitHub Container Registry

**Key:** These are separate images for separate purposes!

### Self-Hosted Runner Capacity

**Question:** Can `[tpu-ci]` runners handle concurrent Levanter + Marin test runs?

**Considerations:**
- Current Marin TPU tests: timeout 10 minutes
- Proposed Levanter TPU tests: 3 JAX versions in matrix
- Total potential concurrent jobs: Marin + (Levanter × 3 versions)

**Mitigation strategies:**
1. Use `concurrency` groups to limit parallel runs
2. Set up additional `[tpu-ci]` runners if needed
3. Monitor runner capacity and queue times
4. Consider priority queuing (Marin high priority, Levanter can wait)

### Migration Risk Mitigation

**Parallel running period:**
- Keep old workflows during migration
- Add `-legacy` suffix to old workflow files
- Run both old and new workflows in parallel
- Compare results for 1-2 weeks
- Remove legacy workflows after validation

**Rollback plan:**
- Keep old workflow files in git history
- Document how to revert if needed
- Have GCP VM creation scripts ready
- Monitor test pass rates closely

## Timeline Estimate

| Phase | Tasks | Estimated Time | Depends On |
|-------|-------|---------------|------------|
| **Phase 1** | Prepare Levanter tests | 2-3 days | - |
| **Phase 2** | Create consolidated test workflow | 1-2 days | Phase 1 |
| **Phase 3** | Consolidate Docker workflows | 1 day | - |
| **Phase 4** | Update integration tests | 1 day | Phase 1, 2 |
| **Phase 5** | Cleanup and docs | 1 day | Phase 2, 3, 4 |
| **Testing & Validation** | Parallel running, monitoring | 1-2 weeks | Phase 5 |
| **Total** | | **~3-4 weeks** | |

## Success Criteria

✅ **Functionality:**
- [ ] All Levanter tests pass on new infrastructure
- [ ] All JAX versions tested successfully
- [ ] No regression in test coverage
- [ ] TPU tests complete within timeout

✅ **Performance:**
- [ ] TPU test startup time < 2 minutes (vs 5-10 currently)
- [ ] Total CI time reduced by 20%+
- [ ] No increase in test flakiness

✅ **Code Quality:**
- [ ] Workflow count reduced by 25%+
- [ ] Clear documentation of new structure
- [ ] No duplicate logic across workflows

✅ **Operations:**
- [ ] TPU CI runners remain stable
- [ ] Docker images build successfully
- [ ] No increase in runner queue times

## Next Steps

1. **Review this plan** with team
2. **Decide on approach** for integration tests (self-hosted vs VM)
3. **Identify tests** that need `@pytest.mark.tpu_ci` marker
4. **Update TPU CI Docker image** with Levanter dependencies
5. **Begin Phase 1** implementation

## Questions for Discussion

1. **What TPU resources do `[tpu-ci]` runners have?**
   - TPU type (v4-8, v5p-8, etc.)?
   - Can they handle v4-32 workloads?
   - How many concurrent runs can they support?

2. **What's the policy on runner capacity?**
   - Should we limit concurrent Levanter jobs?
   - Should we set up priority queuing?
   - Do we need additional runners?

3. **Should we automate TPU CI Docker image builds?**
   - Currently manual (workflow_dispatch only)
   - Should we auto-build on dependency changes?
   - How often do we want to update the image?

4. **Migration timeline:**
   - Can we do parallel running for safety?
   - How long should validation period be?
   - When can we remove legacy workflows?

5. **Testing strategy:**
   - Should we add smoke tests for new workflows?
   - How do we validate JAX version matrix works?
   - Should we test Docker image changes separately first?

---

**Document Version:** 1.0
**Last Updated:** 2025-11-04
**Author:** Claude (AI Assistant)
**Status:** Awaiting Review
