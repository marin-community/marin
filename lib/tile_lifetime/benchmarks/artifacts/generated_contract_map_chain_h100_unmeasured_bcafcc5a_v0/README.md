# Generated Contract/Map H100 replay: unmeasured

## Status

This artifact records an infrastructure bootstrap failure. It contains no GPU
benchmark result and makes no correctness or performance claim.

The intended bounded replay was the standalone generated two-Contract scalar-
Map training component at Shuttle revision
`bcafcc5ab13677146af36644b4f12b008790b676`. The allocation controller used
the separately pinned Iris holder revision
`eafa4d49f7c55fbf2abb26b5d92c1ac7d093f9fb`, because the measured Shuttle
revision's Iris client predates the cluster minimum.

The requested allocation was one H100, one CPU, 32 GB host memory, 50 GB disk,
and batch priority. The one controller-admitted holder failed during Iris setup
before the benchmark's compile/link/load preflight:

```text
error: Group `dev` is not defined in any project's `dependency-groups` table
```

The benchmark process never started. Therefore this artifact has no installed
GPU JAX/toolchain observation, generated-source binary, correctness values,
determinism hashes, handler counts, timing samples, or latency ratio. The
expected versions remain requirements, not measurements: JAX, JAXLIB,
`jax-cuda13-plugin`, and `jax-cuda13-pjrt` 0.11.0.

## Release proof

The controller reported exactly one matching job in `JOB_STATE_FAILED`, no
active matching jobs, and a single setup failure. An explicit terminate request
was issued. Thirty seconds later the exact task-label pod query returned no pod,
and the local dev-GPU session state was absent. No reservation remains active.

## Future authorized replay prerequisite

Use a clean detached checkout for the measured Shuttle source and a separately
pinned current Iris holder client. The narrow holder submission workspace must
remain below the 25 MB controller bundle limit while retaining the task
bootstrap metadata expected by Iris. For the minimal holder used here, that
means defining the `dev` dependency group (an empty group is sufficient for the
holder command) or using an equivalent small bootstrap project. Transfer the
exact Shuttle archive after allocation and record both source revisions.

Do not interpret the failed holder as a component failure and do not reuse this
directory as the destination for a future measurement.

