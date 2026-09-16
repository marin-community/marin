## Verdict: no blocker

The wrapper preserves everything you asked about. Details, with the specific mechanism for each:

**Artifact fingerprints — preserved.** `ArtifactStep.fingerprint_payload()` (`lib/marin/src/marin/execution/lazy.py:213`) hashes only `canonical_json(build_config(...))`; resources are not in the payload, and `path()` is `{prefix}/{name}/{version}`. The wrapper mutates the `JobRequest` at `submit` time, downstream of `data_steps()`/`expected_fingerprint`, so neither the subset fingerprint nor the 28 training fingerprints move. Equally important: `code_pins()` (`launch_tpp10_domain_sweeps.py:45`) enumerates four named files plus `ASSETS.iterdir()` — it does **not** scan the package directory, so adding `resume_tpp10_domain_sweeps.py` leaves `extension_code_sha256` and therefore `plan_sha256` unchanged. `main()`'s archived-plan equality check at `launch_tpp10_domain_sweeps.py:376` will pass.

**Callable payload — untouched.** `dataclasses.replace(request, resources=...)` keeps the same `entrypoint` object (`Entrypoint.from_callable(lambda: self.fn(*args, **kwargs))`), `environment`, `max_retries_failure=0`, `max_retries_preemption=100`, `max_task_failures=0`, `priority`, `processes_per_task`, and forwards `adopt_existing`. `JobRequest.__post_init__` re-runs but `replicas` is already resolved to `1`, so it stays `1`.

**Runtime/source pins — unchanged.** `dependency_groups_for_resources` keys off `resources.device` only and is computed in `RemoteCallable.__call__` before the wrapper sees the request; `convert_environment(..., request.resources.device)` still sees `CpuConfig` → same `JAX_PLATFORMS=cpu` extras. `self._iris` is the parent's own `IrisClient`, so workspace/bundle resolution is identical to `FrayIrisClient.from_iris_client`. The in-job recipe guards (`prepare_subset`'s `file_sha256(prepare_starcoder_tpp10.py)` check) are hash-of-named-file, not hash-of-bundle, so the extra wrapper file in the bundle is inert.

**central1 + on-demand — preserved.** `replace(request.resources, ram="4g")` copies `cpu=2`, `disk="32g"`, `preemptible=False`, `regions=(REGION,)`, `zone=ZONE`. `convert_constraints` (`fray/iris_backend.py:135`) therefore still emits the non-preemptible, region, and zone constraints.

**TPU requests — never intercepted.** The `isinstance(request.resources.device, CpuConfig)` gate excludes `launch.TPU`, `uncheatable.TPU`, and `pod.resources`, all of which are `ResourceConfig.with_tpu(...)`.

**The guard actually fires.** `StepRunner.run` explicitly captures `_current_client_var` on the calling thread and re-establishes it inside each worker thread (`step_runner.py:222`, `:389-401`), and `asyncio.to_thread`/`asyncio.Task` copy the context — so `current_client()` returns your subclass on every submission path in `submit()`. Name match holds: `remote(original.prepare_subset, resources=CPU, ...)` passes no `name`, so `RemoteCallable.__call__` builds `prepare_subset-<uuid4 hex8>`, which survives `sanitize_job_name` unchanged and satisfies `startswith("prepare_subset-")`. The `request.resources != preparation.CPU` arm compares the identical `CPU` object, so it is trivially satisfied.

**Only one CPU submission will occur.** `original_data.verify_caches` and `pending_training_steps` are status-only reads; `uncheatable.prepare_caches` tokenizes in-process on the parent; `_launch_step` returns `None` for `STATUS_SUCCESS` deps, and `VERSION = "2026.09.11"` is non-mutable so nothing rebuilds. The five completed caches are skipped.

### Two things to be aware of (not corrections)

1. **The guard is fail-closed on the wrong CPU task.** If FineMath `raw` or `parent` turned out not to be `STATUS_SUCCESS`, their jobs are named `prepare_parquet-…`/`prepare_parent-…` and the wrapper would raise `ValueError` *before* submitting, aborting the resume rather than silently running them at 4 GiB. That matches your stated intent; just know that "5 of 6 successful" is load-bearing, and a mismatch surfaces as an abort, not a fallback.

2. **`ram` is the container limit, not a soft target.** 4 GiB against a 1.35 GiB measured peak leaves ~3x for the interpreter, uv-synced deps, and the 8 MiB GCS chunk buffers — comfortable, but it is an OOM-kill boundary, and `max_retries_failure=0` means a breach fails the job rather than retrying.
