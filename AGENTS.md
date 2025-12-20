# Agent Guidelines for Marin

## How to Use This Guide

- Start with the shared practices below; if you discover missing guidance, expand this document so the next agent benefits.
- When you uncover directory-specific guidance, add it to the relevant subproject manual so the next agent stays aligned.
- Consult the subproject manuals when working in submodule trees:
  * `lib/levanter/AGENTS.md` for Levanter-specific conventions.
  * `lib/marin/AGENTS.md` for Marin-specific conventions
- When a recipe exists, follow it—the agent-friendly playbooks live in `docs/recipes/`. Some live in the individual `lib/*/docs` directories.

## Shared Workflow Playbooks

- Begin with the agent-friendly recipes in `docs/recipes/`.
- The first step for dataset addition is schema inspection. See the [add_dataset.md](docs/recipes/add_dataset.md) recipe for details.
- You can help organize experiments using the [organize_experiments.md](docs/recipes/organize_experiments.md) recipe.
- Follow the rules and examples in each recipe to ensure compatibility and automation-friendliness.

## Shared Coding Practices

### Tooling

- Assume Python >=3.11.
- Always use `uv run` for Python entry points. If that fails, try `.venv/bin/python` directly.
- Run `uv run python infra/pre-commit.py --all-files` before sending changes; formatting and linting are enforced with `ruff`.
- Keep type hints passing under `uv run mypy`; configuration lives in `pyproject.toml`.

### Communication & Commits

- NEVER SAY "You're absolutely right!"
- You never credit yourself in commits.
- NEVER EVER EVER credit yourself in commit messages.

### Code Style

- Put all imports at the top of the file. Avoid local imports unless technically necessary (for example, to break circular dependencies or guard optional dependencies).
- Prefer top-level functions when code does not mutate shared state; use classes to encapsulate data when that improves clarity.
- Prefer top-level Python tests and fixtures.
- Use early returns (`if not x: return None`) when they reduce nesting.
- Do not introduce ad-hoc compatibility hacks like `hasattr(m, "old_attr")`; update the code consistently instead.
- Do not use `from future import ...` statements.
- Document public APIs with concise Google-style docstrings.

### Error Handling

- Let exceptions propagate by default.
- Only catch exceptions when you can add meaningful context and re-raise, or when you are intentionally altering control flow.
- NEVER EVER SWALLOW EXCEPTIONS unless specifically requested by the user.

### Documentation

- Keep MkDocs content in sync with code. Docs live in `docs/` or in the subproject's `docs/` directory; use Markdown and mkdocs-style links when referencing symbols.
- Public-facing modules and APIs need concise Google-style docstrings; align terminology across code and docs.

### Deprecation

- Unless specifically requested, do not introduce deprecation or fallback paths—update all call sites instead.

## Comments

You write detailed comments when appropriate to describe code behavior as a
whole, e.g. at the module or class level, or when describing some subtle
behavior.

You don't generate comments that merely restate the code, e.g.

<bad>
     # Use in-memory rollout queue
    rollout_queue = InMemoryRolloutQueue()
</bad>

<good>
# We have found that each instance of a FlightServer can provide approximately 1GB/s
# of throughput. As our typical VMs run with 200Gbps NICs, running 16 parallel servers
# should be sufficient to saturate the network.
</good>

## Planning

- When planning, you produce detailed plans including code snippets.
- You ask questions up front when building a plan instead of guessing.
- When a request feels too large for one pass, capture a plan (for example in `.agents/projects/` when the subproject provides one) before pausing.

## Testing

- Always fix tests if you broke them.
- Do not fix tests by relaxing tolerances or hacking around them.
- Run the appropriate tests for your changes (for example, `uv run pytest` under the relevant directory); consult subproject guides for preferred markers.
- Use pytest features like fixtures and parameterization to avoid duplication and write clean code.

## Environment

- Prefer to use `uv` when possible. If you can't (for instance, due to sandbox restrictions) you can use `.venv/bin/python`

> This file will be expanded as agent workflows and best practices evolve.


## Infrastructure Note: Ray TPU Worker Disk Cleanup
This document summarizes the investigation and resolution of the "No space left on device" error encountered on the us-central1-vllm cluster on 2025-12-20.

### Issue Summary
Symptoms: Ray jobs failed during runtime environment setup with OSError: [Errno 28] No space left on device during pip install. Root Cause: The /tmp/gcsfuse_mount directory on several TPU workers was not correctly mounted to GCS, causing model downloads to write directly to the local boot disk (100GB). Specifically, /tmp/gcsfuse_mount/models was consuming ~62GB per worker.

### Investigative Commands
1. Identify Workers with Low Disk Space
Use the provided cleanup script in dry-run mode to scan the cluster:

```bash
uv run scripts/ray/cleanup_disk.py --config infra/marin-us-central1-vllm.yaml --threshold 10 --dry-run
```

2. Inspect a Specific Worker
SSH into a suspected worker to verify disk usage and mount status:

```bash
# Check disk usage
gcloud compute tpus tpu-vm ssh <tpu_name> --zone us-central1-a --command "df -h /"
# Check if gcsfuse is actually mounted
gcloud compute tpus tpu-vm ssh <tpu_name> --zone us-central1-a --command "mount | grep gcsfuse"
# Find largest directories in /tmp
gcloud compute tpus tpu-vm ssh <tpu_name> --zone us-central1-a --command "sudo du -sh /tmp/* | sort -h"
```

### Resolution Commands
1. Delete Misplaced Files
If /tmp/gcsfuse_mount is verified to be a local directory (not a mount), delete it:

```bash
gcloud compute tpus tpu-vm ssh <tpu_name> --zone us-central1-a --command "sudo rm -rf /tmp/gcsfuse_mount"
```

2. Re-initialize Ray Worker
After clearing space, the Ray container should be restarted to ensure a clean state:

```bash
uv run scripts/ray/cluster.py --config infra/marin-us-central1-vllm.yaml init-worker <tpu_name>
```

### Prevention & Maintenance
* Threshold Monitoring: Run `scripts/ray/cleanup_disk.py` periodically to detect workers dropping below 20% free space.
* Mount Verification: Ensure gcsfuse mounting logic in infra/*.yaml setup commands includes a check to ensure the mount point isn't written to if the mount fails.
* Cleanup Cron: Ensure the automated cleanup job is running on the head node:
```bash
uv run scripts/ray/cluster.py --config infra/marin-us-central1-vllm.yaml start-cleanup
```