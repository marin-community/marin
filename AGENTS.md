# Agent Guidelines for Marin

## How to Use This Guide

- Start with the shared practices below; if you discover missing guidance, expand this document so the next agent benefits.
- When you uncover directory-specific guidance, add it to the relevant subproject manual so the next agent stays aligned.
- Consult the subproject manuals when working in submodule trees:
  * the **Levanter** section below for Levanter-specific conventions.
  * the **Marin** section below for Marin-specific conventions
  * the **Iris** section below for Iris-specific conventions

## Shared Workflow Playbooks

- Begin with the agent-friendly recipes in `docs/recipes/`.
- For PR descriptions, testing, specifications, and review workflow, follow [pull-request.md](docs/recipes/pull-request.md).
- The first step for dataset addition is schema inspection. See the [add_dataset.md](docs/recipes/add_dataset.md) recipe for details.
- You can help organize experiments using the [organize_experiments.md](docs/recipes/organize_experiments.md) recipe.
- For long-running benchmark/research threads, follow [agent_research.md](docs/recipes/agent_research.md).
- For canary/daily ferry proposal, launch, and monitoring workflow, follow [ferries.md](docs/recipes/ferries.md).
- When making significant changes to Grug/Grugformer, follow [change_grug.md](docs/recipes/change_grug.md).
- For profiling ingestion and agent-driven optimization workflows, follow [agent_profiling.md](docs/recipes/agent_profiling.md).
- Follow the rules and examples in each recipe to ensure compatibility and automation-friendliness.

## Shared Coding Practices

### Tooling

- Assume Python >=3.11.
- Always use `uv run` for Python entry points. If that fails, try `.venv/bin/python` directly.
- Run `./infra/pre-commit.py --all-files` before sending changes; formatting and linting are enforced with `ruff`.
- Keep type hints passing under `uv run pyrefly`; configuration lives in `pyproject.toml`.

### Communication & Commits

- NEVER SAY "You're absolutely right!"
- You never credit yourself in commits.
- NEVER EVER EVER credit yourself in commit messages.

### Agent-Generated GitHub Activity

- When an agent creates a PR or an issue using the user's auth token, it must add the `agent-generated` label.
- When an agent comments on a PR or issue using the user's auth token, the comment must begin with an `🤖` emoji unless the exact comment text was explicitly approved by the user. If it cannot be at the very beginning for formatting or workflow reasons, it should come as soon as possible.

### Code Style

- Put all imports at the top of the file. Avoid local imports unless technically necessary (for example, to break circular dependencies or guard optional dependencies).
- Prefer top-level functions when code does not mutate shared state; use classes to encapsulate data when that improves clarity.
- Prefer top-level Python tests and fixtures.
- Separation of responsibilities: when a change introduces a new subsystem (e.g., serving/inference, data access, evaluation), encapsulate lifecycle/configuration in a dedicated module and have callers depend on the interface rather than re-implementing setup/teardown details.
- Disprefer internal mutation of function arguments, especially config dataclasses; prefer returning a modified copy (e.g., via `dataclasses.replace`) so call sites remain predictable and side effects are explicit.
- Use early returns (`if not x: return None`) when they reduce nesting.
- Do not introduce ad-hoc compatibility hacks like `hasattr(m, "old_attr")`; update the code consistently instead.
- Document public APIs with concise Google-style docstrings.
- Prefer small, concrete helpers over abstraction that adds indirection without reuse.
- When defaults depend on environment/resource type, resolve them once and fail fast on unknown/ambiguous inputs rather than silently guessing.
- Keep environment detection logic minimal and explicit; avoid multi-key heuristics unless they are clearly required.
- Prefer single strong signals over sprawling defensive checks when detecting environment state (e.g., check the one variable that must be set rather than many optional ones).
- In marin we generally prefer logging over `print` statements. `print` is fine for debugging and "scripts".

### Error Handling

- Let exceptions propagate by default.
- Only catch exceptions when you can add meaningful context and re-raise, or when you are intentionally altering control flow.
- NEVER EVER SWALLOW EXCEPTIONS unless specifically requested by the user.

### Documentation

- Keep MkDocs content in sync with code. Docs live in `docs/` or in the subproject's `docs/` directory; use Markdown and mkdocs-style links when referencing symbols.
- Public-facing modules and APIs need concise Google-style docstrings; align terminology across code and docs.
- Write docs for readers who do not have conversational context: include enough problem framing, assumptions, commands, and results that the document stands on its own.

### Deprecation

**NO BACKWARD COMPATIBILITY**: Do NOT add deprecation warnings, fallback paths, or compatibility shims. Update all call sites instead. Only add backward compatibility if the user explicitly requests it.

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
- Avoid “tautological” tests that merely restate implementation logic as asserts; prefer tests that validate externally-observable behavior, integration points, or realistic failure modes.
- Run the appropriate tests for your changes (for example, `uv run pytest` under the relevant directory); consult subproject guides for preferred markers.
- Use pytest features like fixtures and parameterization to avoid duplication and write clean code.

PREFER:

- Integration style tests which exercise behavior and test the output

DO NOT:

- Create tests which validate obvious features: if a type exists, a constant has a value, etc.


## Environment

- Prefer to use `uv` when possible. If you can't (for instance, due to sandbox restrictions) you can use `.venv/bin/python`


## Levanter

# Levanter Agent Guidelines

Levanter is incorporated into `src/levanter`. Start with the shared instructions in `the root AGENTS.md`; the notes below cover Levanter-specific conventions.

## Workflow Notes

* Build on the shared practices in `the root AGENTS.md`; this file only captures Levanter-specific expectations.
* Capture multi-session Levanter work in `.agents/projects/` inside this directory so partial progress is easy to resume.
* Keep Levanter-specific recipes in `docs/recipes/` and expand them when you uncover repeatable tasks.

## Recipes

* [Porting a Model to Levanter](docs/recipes/port-models.md): A guide for porting model architectures to the Levanter ecosystem using Haliax and Equinox.

## Documentation

* Levanter documentation lives in `docs/levanter/`. Update `mkdocs.yml` when adding or reorganizing sections.
* Use mkdocs-style symbol links (for example `[full.path.object][]`) so cross-references work in the generated site.
* Add a short intent paragraph to new recipes and link them from relevant guides or README sections.

## Testing

* Run `uv run pytest tests -m "not entry and not slow and not ray"` to execute the default Levanter suite.
* Do not relax numerical tolerances without prior agreement from a human. Prefer `assert_allclose` with 1e-4 for complex modules and 1e-5 for simpler ones unless a specific case demands otherwise.
* Batch sizes should generally be a low multiple (e.g. 1 or 2) of the number of `len(jax.devices())` to ensure multi-device correctness.
* Mark long-running tests with `@pytest.mark.slow` so they stay out of the default suite.
* Always protect PyTorch-dependent tests with `@skip_if_no_torch` so they skip cleanly when PyTorch is unavailable.
* **CI tip:** use `astral-sh/setup-uv` to install `uv`, run `uv python install`, and then run `uv sync`/`uv pip` to guarantee the expected Python version in workflows.

### TPU VMEM flags for kernel tuning

* For TPU Pallas benchmark/tuning runs, set `LIBTPU_INIT_ARGS` by TPU generation:
  * `v5p`/`v5e`: `--xla_tpu_scoped_vmem_limit_kib=50000`
  * `v6e`: `--xla_tpu_scoped_vmem_limit_kib=98304`
* Do not set a special scoped VMEM limit flag on `v4` unless a user explicitly asks for it.

## Design Preferences

* **Named tensors:** Levanter relies heavily on [Haliax](https://github.com/stanford-crfm/haliax). Represent arrays with `NamedArray` and explicit `Axis` objects, and operate over named axes whenever possible.
* **Generic code:** Many utilities use Python generics and dataclasses. Prefer reusable functions/classes parameterized with `TypeVar` instead of hard-coding concrete types.
* **Configurations:** Configuration files are dataclasses loaded via `draccus`. Keep them declarative and typed.
* **Datasets:** Represent datasets as `AsyncDataset` or `SyncDataset` in `levanter.data.dataset`. Favor asynchronous datasets and ensure they support slicing, shuffling, and mapping. Prefer `AsyncDataset` unless there is a concrete reason not to.
* **Logging and tracking:** Use the existing tracker hooks (for example Weights & Biases or TensorBoard) for metrics instead of ad-hoc logging.
* **Reproducibility:** Aim for deterministic training; avoid nondeterminism unless explicitly needed.
* Prefer `Stacked` with `fold` or `scan` over hand-written loops to improve compile times and gradient-checkpointing behavior.

## JIT Safety

* Avoid data-dependent Python control flow inside jitted code.
* Do not rely on dynamic shapes or dynamic lengths when indexing.
* Use `debug.print` if you need to inspect values.
* Choose jit-safe variants like `jnp.where` or `hax.where` when branching on data.
* Do not call `jax.default_backend()` or `jax.devices()` at module import time; resolve backend/device lazily inside runtime functions.

Any method inside an `equinox.Module`, any function decorated with `jax.jit` or one of its variants (for example `eqx.filter_jit` or `jax.named_jit`), and any helpers they call must follow these jit-safety rules.

## Additional Tips

* Use `NamedArray` and `Axis` for model parameters and activations.
* Levanter uses Equinox and Haliax—avoid Flax and Haiku layers in new code.
* Prefer functional-style JAX code with explicit PRNG keys.
* Avoid hard-coding dataset paths; surface them through configuration.
* Maintain compatibility with both GPU and TPU backends when extending the library.

## Marin

# Vendored Marin Agent Guidelines

!!! note
In this file, we use leading `/` to refer to paths relative to the repository root.

The `/src/marin` directory contains a vendored copy of Marin. Read the shared instructions in `/AGENTS.md` first; the notes below call out directory-specific expectations.

## Workflow Notes


* Build on the shared practices in `/AGENTS.md`; capture only `/src/marin`-specific expectations here.
* Keep this guide updated whenever you learn a `/src/marin`-specific best practice.
* Capture packaging or release checklists in `/docs/recipes/` so other agents can repeat them.

## Packaging and Code Layout

* The packaging metadata lives in `/pyproject.toml`. Update version pins, extras, or dependency groups together with the corresponding upstream change.
* When you add new modules under `/src/marin/`, follow the shared docstring and import conventions from the root guide and ensure the files remain automation-friendly.

## Data Access

* Do not special-case Google Cloud Storage unless absolutely necessary. Use `fsspec.open` and other fsspec helpers so code stays filesystem-agnostic.
* You generally should not copy data artifacts (for example `.json` or `.parquet` files) to the local filesystem when writing Python code—stream them through fsspec instead.
* Avoid hard-coding GCS paths such as `gs://marin-us-central2/foo/bar`. Prefer referencing pipeline steps; if you must inject a literal path, wrap it with `InputName.hard_coded` and call out the follow-up risk. This is **critical** for large files or directory trees.
* Again, NEVER EVER EVER load GCS files from across region if they are more than a few MB.

## Testing

* Re-run the relevant tests with `uv run pytest` targeting any suites that exercise the vendored package (for example a future `tests` directory) before submitting changes.
* Do not relax tolerances or add hacks to silence failures—fix the underlying issue or coordinate on the shared guide.

Add any additional directory-specific conventions here as they emerge.

## Iris

# Iris Agent Notes

- Use Connect/RPC for APIs and dashboards. Do not use `httpx` or raw HTTP.
- After changing `.proto` files, regenerate via `scripts/generate_protos.py`.
- Prefer shallow, functional code that returns control quickly; avoid callback-heavy or inheritance-driven designs.

The Iris testing policy lives in `TESTING.md`.

## Docs

Read the docs for the area you are changing. If docs disagree with code, update the docs (or add a task) in the same PR.

Key docs (kept intentionally short):

- `README.md` — overview + quick start
- `OPS.md` — operating / troubleshooting a live cluster
- `docs/autoscaler-v2.md` — autoscaler design + terminology
- `docs/controller-flow.md`, `docs/worker-flow.md` — controller/worker lifecycle
- `docs/task-states.md` — task state machine + retry semantics
- `docs/coreweave.md` — CoreWeave platform + `runtime=kubernetes` behavior

Resource model note: CPU demand is fungible and can route to any group; GPU/TPU demand is non-fungible and must match device type (and optionally variant). Priority configuration determines whether CPU spillover lands on accelerator groups.

## Imports

Avoid `TYPE_CHECKING`. Use real imports. If you hit a cycle:

- Prefer refactoring when sensible.
- Otherwise use a `Protocol` at the boundary.

## RPC / API Accessibility

Any functionality exposed by the worker or controller dashboards must also be available via RPC.
Dashboards should be a thin UI over the RPC API, not a second implementation path.

## Concurrency Model

Platform operations (`terminate`, `create_slice`, etc.) shell out via `subprocess.run` and are thread-safe.
For concurrent independent platform operations, use `concurrent.futures.ThreadPoolExecutor` (not asyncio) and apply hard timeouts so the CLI cannot hang indefinitely.

## Planning

Prefer spiral plans over linear plans: each stage should be independently testable (proto → server stub → client wiring → end-to-end test), then iterate.

### Light Worker Mode (CoreWeave + runtime=kubernetes)

When `runtime: kubernetes` is configured, worker Pods are intentionally "light":
- Worker Pod must not request `nvidia.com/gpu` or `rdma/ib`.
- Task Pods created by `src/iris/cluster/runtime/kubernetes.py` request accelerators per task.
- Worker Pod still uses the scale-group `nodeSelector` and `hostNetwork: true`.
- Worker Pod passes control-plane env needed for task-pod creation (for example
  `IRIS_SERVICE_ACCOUNT_NAME`, and `IRIS_S3_SECRET_NAME` when S3 is enabled).

Quick verification:
- Worker create log shows `resource_limits=none`.
- `kubectl get pod <worker> -o jsonpath='{.spec.containers[0].resources}'` is empty.
- Task pod specs include GPU limits when task resources request GPUs.

**Disk layout**: CoreWeave bare-metal nodes have a 15 GB RAM disk (`/dev/ram0`) as the root
filesystem and a multi-TB NVMe RAID (`/dev/md127`) mounted at `/mnt/local`. Bind mounts expose
it as `/var/lib/containerd`, `/var/lib/kubelet`, `/opt`, etc. The `cache_dir` must point to the
NVMe (e.g. `/mnt/local/iris-cache`) — the default `/var/cache/iris` lands on the tiny RAM disk
and will fill up immediately when installing CUDA packages.

All K8s resources (RBAC, ConfigMap, shared NodePools, Deployment, Service) are created
automatically by `iris cluster start` via `CoreweavePlatform.start_controller()`. RBAC
manifests (Namespace, ServiceAccount, ClusterRole, ClusterRoleBinding) are defined in
`CoreweavePlatform.ensure_rbac()` — no separate YAML files needed.

## Key Modules

### Time Utilities

Use `iris.time_utils` for all time-related operations instead of raw `datetime` or `time`:

| Class | Purpose |
|-------|---------|
| `Timestamp` | Point in time (epoch-based). Use for created_at, timestamps in logs, etc. |
| `Duration` | Time interval. Use for timeouts, intervals, configuration values. |
| `Deadline` | Monotonic deadline for timeout checks. Use in polling loops. |
| `Timer` | Elapsed time measurement. Use for performance tracking. |
| `ExponentialBackoff` | Retry/polling with backoff. Use `wait_until()` for condition polling. |

Example:
```python
from iris.time_utils import Timestamp, Duration, Deadline

created_at = Timestamp.now()
timeout = Duration.from_seconds(30.0)
deadline = Deadline.from_now(timeout)
deadline.wait_for(condition)

while not deadline.expired():
    if condition():
        break
    time.sleep(0.1)
```

### Deployment Topology

The controller is a plain GCE VM with no zone affinity to workers — it can run
in any zone and serve workers across all regions.

**When changing the controller zone**, update in `examples/marin.yaml`:
- `controller.gcp.zone` — the GCE zone
- Image tags use `ghcr.io/marin-community/...` format. The controller and
  autoscaler automatically rewrite these to AR remote repos for the VM's
  continent at boot time.

**Docker registries**: Bootstrap scripts in `src/iris/cluster/platform/bootstrap.py` auto-detect
AR image tags and configure `gcloud auth configure-docker`. AR remote repos
proxy GHCR — see `docs/image-push.md` for setup.

### Multi-Region Image Push/Pull

Images are pushed only to **GHCR** (`ghcr.io/marin-community/`). GCP VMs pull
from **Artifact Registry remote repositories** that act as pull-through caches
for GHCR. See `docs/image-push.md` for full details.

**Push**: `iris build push` and `iris cluster start` push to GHCR only.

**Pull**: The autoscaler and controller bootstrap automatically rewrite GHCR
image tags to the AR remote repo for the VM's continent:
- `ghcr.io/org/image:v1` → `us-docker.pkg.dev/project/ghcr-mirror/org/image:v1`

Set `defaults.worker.docker_image` to a `ghcr.io/...` tag. Non-GHCR tags
(`docker.io`, existing AR tags) pass through unchanged.

**Bundle storage** (`controller.bundle_prefix`) is a GCS URI with no zone
affinity — globally accessible.

**Zone validation**: `src/iris/cluster/config.py` validates that every scale group zone
appears in `platform.gcp.zones`. Multi-zone scale groups are auto-expanded by
`_expand_multi_zone_groups()`.

## Testing

See `TESTING.md` for the testing policy and commands.

## Haliax

# Haliax LLM Agent Guidelines

This document summarizes important conventions for contributing code or documentation to the Haliax
repository. Follow these notes when implementing new features or fixing bugs.

## General Guidelines

* **Get better.** Whenever you discover something missing from these guidelines, or the requester
  suggests a better way to do something, please update this document. The goal is to make it easier for
  everyone to contribute and maintain the codebase. Generally speaking, you should add bullets or new sections.
  Be sure to do this when directed to. For example, if directed that you should never relax tolerances in
  floating point tests, add that to the list.
* **Playbooks.** Sometimes, there are repeatable tasks (e.g. porting models) for which we follow a standard set of steps.
  Please reference `.playbooks/` to see what playbooks are available, or see the list below. If you want to add a playbook
  write a markdown doc named e.g. `.playbooks/add-types.md` and add a pointer to it in the list below.

## Playbook

- Adding Haliax-style tensor typing annotations are described in @.playbooks/add-types.md
- [Wrapping standard JAX functions](.playbooks/wrap-non-named.md) so they operate on `NamedArray`

## Code Style

* **Python version**: the project targets Python >=3.10.
* **Formatting and Linting**: We use `./infra/pre-commit.py` (ruff, black, license headers) to keep files consistent.
* **Typing**: the code base uses `mypy` for static type checking. `mypy` is run by the same `infra/pre-commit.py` entrypoint and the
  configuration is found in `pyproject.toml`.
* **Run `./infra/pre-commit.py --all-files`** before committing. The CI workflows run the same checks.
* **Use `uv run` for commands.** When running tools like `pytest` or other scripts, invoke them via `uv run` so the development dependencies are active.
* **Doc Strings**: All public functions, classes, and modules should have docstrings, unless
  their purpose is painfully obvious. Use
  [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for
  consistency.
* **Commenting**: Use comments to explain why something is done a certain way, especially if it is not
  immediately obvious. Avoid commenting on every line of code; focus on the intent and purpose of
  complex logic. Demarcating logical groups of code with comments is encouraged, unless it is better
  to refactor the code into smaller functions or classes.
* **Mkdocs**: We use [Mkdocs](https://www.mkdocs.org/) for documentation. The main documentation is in
  the `docs` directory. Use Markdown for writing docs, and follow the existing structure. When linking to
  symbols, prefer using mkdocs-style links (e.g. With a custom title: `[full.path.object2][]` or
  `[Object 1][full.path.object1]`)
* **Documentation**: When adding new features, ensure that the documentation is updated accordingly.
  This includes updating the Mkdocs files and any relevant docstrings. If you add a new module or
  significant functionality, consider adding a dedicated section in the documentation. When you
  wrap a new JAX function, add a reference to it in `docs/api.md` so users can discover it.

## Testing

* Tests are executed with `pytest`. The default workflow runs ` XLA_FLAGS=--xla_force_host_platform_device_count=8 PYTHONPATH=tests:src:. uv run pytest tests`.
* In general, never relax tolerances in floating point tests unless specifically discussed with the
  team. Use `assert_allclose` with appropriate tolerances for numerical comparisons. We typically use
  1e-4 for more complex modules, and 1e-5 for simpler ones.
* Always mark tests that depend on pytorch with `@skip_if_no_torch` to ensure they are skipped
  when PyTorch is not available. This is particularly important for tests that require PyTorch-specific
  functionality.


## Design Preferences

* **Generic code**: many utilities are written with Python generics and dataclasses. Where possible,
  write reusable functions or classes that operate over TypeVars instead of hard coding concrete types.
* **Reproducibility**: Haliax aims for determinism where possible. Avoid sources of
  nondeterminism unless explicitly required.
* Prefer Stacked with fold or scan over writing custom loops, for better compile times and gradient checkpointing support
* For configuration, we prefer frozen dataclasses over dictionaries.

## Library conventions
- Haliax revolves around `NamedArray` and named shapes, either via Axis objects or "shape dicts" (e.g. `{"batch": 42, "embed": 16}).
  Prefer APIs that accept axes or axis names rather than hard‑coding positional dimensions. In particular, use AxisSpec and AxisSelection where possible.
- Utilities should be written so they work with arbitrary axis names. Avoid relying on
  fixed axis orders when possible.
- Use the provided modules in `haliax.nn` or Equinox when building neural network layers.
- Type annotations can use named shapes shorthand provided in `haliax.haxtyping`: `ht.f32[NamedArray, "batch"]`
  for a float32 array with a "batch" axis, or `ht.Float[NamedArray, "batch"]` for any floating point dtype.

## Documentation
- Public functions and modules require docstrings. If behavior is non‑obvious, add examples in `docs/`.
- For a concise overview of Haliax aimed at LLM agents, see [docs/primer.md](docs/primer.md).
