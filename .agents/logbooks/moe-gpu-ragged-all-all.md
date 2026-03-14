# MoE GPU Ragged All-to-All: Research Logbook

## Scope
- Goal: reproduce the key Expert Parallelism comparison style from #2710 on GPU, with the same focused Grug MoE harness, and measure whether a `ragged_all_to_all` dispatch/return path is competitive with the current ring EP path on GPU.
- Primary metric(s): end-to-end `forward` and `forward_backward` wall time, plus `tokens/s`, for apples-to-apples kernel comparisons on the same GPU host and shape.
- Constraints:
  - Do not start, stop, or restart the CoreWeave Iris cluster without explicit user approval.
  - Keep comparisons fixed-shape and fixed-routing unless the axis being changed is the point of the run.
  - Reuse the sealed `grug-moe-ep-ring-20260307` benchmark harness as the starting point.
- Experiment issue: https://github.com/marin-community/marin/issues/3633

## Baseline
- Date: 2026-03-13
- Code refs:
  - `lib/levanter/scripts/bench/bench_moe_hillclimb.py`
  - `lib/levanter/src/levanter/grug/grug_moe.py`
  - MaxText reference: `src/maxtext/layers/moe.py` `ragged_all_to_all` path around the `ragged_all_all` mode
- TPU reference baseline from #2710 / follow-on seal:
  - fixed shape: `tokens=32768`, `hidden=2048`, `mlp_dim=768`, `experts=128`, `topk=8`, `shared_expert_dim=2048`
  - current ring EP on `v5p-16`: `EP=1 24.289 ms`, `EP=2 23.440 ms`, `EP=4 20.124 ms`, `EP=8 17.847 ms`
- Fixed GPU baseline case for repeated comparison:
  - shape: `tokens=32768`, `hidden=2048`, `mlp_dim=768`, `experts=128`, `topk=8`, `shared_expert_dim=2048`
  - distributions: `random`, `runs`
  - kernels: `current`, `ragged_a2a`
  - EP sweep: `1,2,4,8` when local GPU device count allows it

## Initial Hypotheses
- GPU may shift the EP communication tradeoff enough that `ragged_all_to_all` is more competitive than it was on TPU.
- The existing harness `ragged_a2a` kernel is the closest in-repo analog to the MaxText `ragged_all_all` path and should be benchmarked before inventing a new kernel variant.
- If `ragged_a2a` regresses on GPU, the regression is more likely to come from local sort/permute overhead or buffer sizing than from the collective primitive alone.

## Stop Criteria
- Enough evidence to answer whether `ragged_a2a` is better, neutral, or worse than `current` on the fixed GPU baseline case.
- If promising, enough follow-up evidence to say whether the effect holds across `topk in {2, 8}` or `distribution in {random, runs}`.
- If blocked by infrastructure, leave a fully reproducible command bundle and a documented blocker state.

## Experiment Log
### 2026-03-13 15:40 - Kickoff and artifact recovery
- Hypothesis: the sealed `grug-moe-ep-ring-20260307` tag is the right starting point because it still contains both the focused benchmark harness and the prior research logbook.
- Command:
  ```bash
  git worktree add -b research/moe-gpu-ragged-all-all ~/marin-wt/moe-gpu-ragged-all-all grug-moe-ep-ring-20260307
  git show grug-moe-ep-ring-20260307:lib/levanter/scripts/bench/bench_moe_hillclimb.py | sed -n '1,320p'
  git show grug-moe-ep-ring-20260307:.agents/logbooks/grug-moe-ep-ring.md | sed -n '1,260p'
  ```
- Config:
  - source snapshot: `grug-moe-ep-ring-20260307`
  - new branch: `research/moe-gpu-ragged-all-all`
- Result:
  - Worktree created successfully.
  - Confirmed the harness already contains a `ragged_a2a` kernel that uses `jax.lax.ragged_all_to_all` in both dispatch and return.
- Interpretation:
  - The first GPU comparison can be done without inventing a new kernel name; the main remaining question is whether the current harness runs cleanly on GPU and whether the existing `ragged_a2a` path is the intended MaxText analog.
- Next action:
  - Create the experiment issue.
  - Do local harness inspection and smoke validation in this worktree.

### 2026-03-13 15:41 - CW cluster status check
- Hypothesis: the CoreWeave cluster might already be available for GPU job submission.
- Command:
  ```bash
  KUBECONFIG=~/.kube/coreweave-iris uv run iris --config=lib/iris/examples/coreweave.yaml cluster status
  kubectl --kubeconfig ~/.kube/coreweave-iris get nodes -o wide
  kubectl --kubeconfig ~/.kube/coreweave-iris get all -n iris
  kubectl --kubeconfig ~/.kube/coreweave-iris get svc -n iris -o wide
  ```
- Config:
  - guide: `~/llms/cw_ops_guide.md`
- Result:
  - Two CW nodes are `Ready`.
  - `iris cluster status` cannot resolve `iris-controller-svc`.
  - `iris` namespace currently has no resources.
- Interpretation:
  - The CoreWeave environment has live Kubernetes nodes but the Iris cluster itself is not currently up, so GPU job submission is blocked without a cluster start/recovery action.
- Next action:
  - Finish local setup and validation.
  - Leave the GPU benchmark matrix ready to run once the cluster is up or the user approves bringing it up.

### 2026-03-13 15:48 - Local host-device smoke for `ragged_a2a`
- Hypothesis: before depending on GPU infrastructure, the sealed harness should at least prove that `ragged_a2a` is wired correctly in the current JAX stack and only backend support is at issue.
- Command:
  ```bash
  XLA_FLAGS=--xla_force_host_platform_device_count=8 uv run python lib/levanter/scripts/bench/bench_moe_hillclimb.py \
    --tokens 256 \
    --hidden 64 \
    --mlp-dim 64 \
    --experts 8 \
    --topk 2 \
    --shared-expert-dim 64 \
    --ep-list 1,2,4,8 \
    --kernel current \
    --bench-pass forward \
    --iters 1 \
    --warmup 0

  XLA_FLAGS=--xla_force_host_platform_device_count=8 uv run python lib/levanter/scripts/bench/bench_moe_hillclimb.py \
    --tokens 256 \
    --hidden 64 \
    --mlp-dim 64 \
    --experts 8 \
    --topk 2 \
    --shared-expert-dim 64 \
    --ep-list 1,2,4,8 \
    --kernel ragged_a2a \
    --bench-pass forward \
    --iters 1 \
    --warmup 0
  ```
- Config:
  - local backend: forced 8 host CPU devices
  - JAX: `0.8.0`
  - jaxlib: `0.8.0`
- Result:
  - `current` completed for `EP=1,2,4,8`.
  - `ragged_a2a` completed at `EP=1`, then failed at the first real collective case with:
    - `UNIMPLEMENTED: HLO opcode ragged-all-to-all is not supported by XLA:CPU ThunkEmitter`
- Interpretation:
  - The harness wiring is present and the `ragged_a2a` path is reachable.
  - CPU is not a viable backend for validating or benchmarking `jax.lax.ragged_all_to_all`.
  - A real GPU or TPU backend is required to evaluate the MaxText-like path.
- Next action:
  - Run the same comparison on GPU once the CoreWeave Iris controller is available.
  - Initial GPU command bundle:
    ```bash
    iris job run \
      --gpu H100x8 --cpu 32 --memory 256g --disk 256g \
      --extra gpu \
      -e MARIN_PREFIX "s3://marin-us-west-04a/marin" \
      -e WANDB_API_KEY "${WANDB_API_KEY}" \
      -e HF_TOKEN "${HF_TOKEN}" \
      -- /bin/bash -lc '
      set -euo pipefail
      cd /workspace/marin
      for pass in forward forward_backward; do
        for dist in random runs; do
          for kernel in current ragged_a2a; do
            uv run python lib/levanter/scripts/bench/bench_moe_hillclimb.py \
              --tokens 32768 \
              --hidden 2048 \
              --mlp-dim 768 \
              --experts 128 \
              --topk 8 \
              --shared-expert-dim 2048 \
              --distribution "$dist" \
              --kernel "$kernel" \
              --ep-list 1,2,4,8 \
              --bench-pass "$pass" \
              --iters 3 \
              --warmup 1
          done
        done
      done'
    ```

### 2026-03-13 15:53 - CW cluster bring-up attempt
- Hypothesis: with explicit user approval, the benchmark worktree can bring the CoreWeave Iris cluster up directly via `iris cluster start`, after which GPU jobs can be submitted from the same worktree.
- Command:
  ```bash
  KUBECONFIG=~/.kube/coreweave-iris uv run iris --config=lib/iris/examples/coreweave.yaml cluster start
  ```
- Config:
  - worktree SHA pinned by Iris build step: `6f96d7c60`
  - images built and pushed:
    - `ghcr.io/marin-community/iris-worker:6f96d7c60`
    - `ghcr.io/marin-community/iris-controller:6f96d7c60`
    - `ghcr.io/marin-community/iris-task:6f96d7c60`
- Result:
  - All three images built and pushed successfully.
  - Startup then failed during `CoreweavePlatform.start_controller()` immediately after namespace/RBAC reconciliation with:
    - `R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY environment variables are required for S3-compatible object storage`
  - Local checks showed:
    - `R2_ACCESS_KEY_ID=unset`
    - `R2_SECRET_ACCESS_KEY=unset`
    - `op whoami` reported not signed in.
- Interpretation:
  - The remaining blocker is not CoreWeave or Iris itself; it is operator credentials for the configured `s3://marin-test/iris/bundles` bundle store.
  - Re-running `iris cluster start` should be sufficient once `op read` can supply the R2 credentials.
- Next action:
  - Unlock/sign into 1Password on the operator machine.
  - Export `R2_ACCESS_KEY_ID` and `R2_SECRET_ACCESS_KEY` via `op read`.
  - Re-run `iris cluster start`, then submit the H100 benchmark job.

### 2026-03-13 16:12 - Cluster and credentials re-check from current operator state
- Hypothesis: the earlier cluster-start blocker is stale, and the current machine state may already have the required cluster and object-store credentials available for fresh shells without another 1Password prompt.
- Command:
  ```bash
  KUBECONFIG=~/.kube/coreweave-iris kubectl get ns
  KUBECONFIG=~/.kube/coreweave-iris kubectl get pods -n iris -o wide
  uv run iris --config=lib/iris/examples/coreweave.yaml cluster status
  zsh -lc 'for v in WANDB_API_KEY HF_TOKEN R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY; do if [[ -n "${(P)v:-}" ]]; then echo "$v=set"; else echo "$v=unset"; fi; done'
  ```
- Config:
  - cluster config: `lib/iris/examples/coreweave.yaml`
  - kubeconfig: `~/.kube/coreweave-iris`
- Result:
  - The `iris` namespace is active and the CoreWeave Iris cluster is up.
  - `iris cluster status` reports the controller healthy with `2/2 healthy` workers, including one ready `h100-8x` worker.
  - Fresh shells see `WANDB_API_KEY`, `HF_TOKEN`, `R2_ACCESS_KEY_ID`, and `R2_SECRET_ACCESS_KEY` via a `launchctl`-backed shell export path, so repeated operator prompts are no longer needed.
- Interpretation:
  - The earlier blocker was real at the time but is no longer the current state.
  - GPU submission can proceed immediately from this machine.
- Next action:
  - Submit a smoke task from the sealed worktree.
  - If the normal `iris job run` path is flaky, fall back to a raw `LaunchJobRequest`.

### 2026-03-13 16:18 - `iris job run` setup mismatch and raw Iris fallback
- Hypothesis: a standard `iris job run` from the sealed worktree should be enough for the GPU benchmark; if not, an equivalent raw request should still prove bundle staging and task execution on H100.
- Command:
  ```bash
  uv run iris --config=lib/iris/examples/coreweave.yaml job run \
    --gpu H100x8 --cpu 1 --memory 8g --disk 16g \
    -- /bin/bash -lc 'cd /app && pwd && test -f pyproject.toml'

  uv run python - <<'PY'
  from pathlib import Path
  from iris.cluster.client.bundle import BundleCreator
  from iris.cluster.config import IrisConfig
  from iris.cluster.types import EnvironmentSpec, JobName, ResourceSpec, gpu_device
  from iris.rpc import cluster_pb2
  from iris.rpc.cluster_connect import ControllerServiceClientSync

  cfg = IrisConfig.load("lib/iris/examples/coreweave.yaml")
  platform = cfg.platform()
  controller_address = cfg.controller_address() or platform.discover_controller(cfg.proto.controller)
  bundle = BundleCreator(Path("/Users/romain/marin-wt/moe-gpu-ragged-all-all")).create_bundle()
  request = cluster_pb2.Controller.LaunchJobRequest(
      name=JobName.root("romain", "raw-gpu-smoke-20260313-232018").to_wire(),
      replicas=1,
      resources=ResourceSpec(cpu=32, memory="256g", disk="256g", device=gpu_device("H100", 8)).to_proto(),
      environment=EnvironmentSpec(extras=["gpu"], python_version="3.11").to_proto(),
      bundle=bundle.proto,
      runtime=cluster_pb2.RuntimeEntrypoint(
          setup_commands=[
              "cd /app",
              "uv sync --quiet --link-mode symlink --python 3.11 --all-packages --no-group dev --extra gpu",
          ],
          run_command=cluster_pb2.CommandEntrypoint(
              argv=[
                  "bash",
                  "-lc",
                  "cd /app && source .venv/bin/activate && python - <<'INNER'\nimport jax\nprint('DEVICES', jax.devices())\nprint('DEVICE_COUNT', jax.device_count())\nINNER",
              ]
          ),
      ),
  )
  with platform.tunnel(address=controller_address) as controller_url:
      client = ControllerServiceClientSync(address=controller_url, timeout_ms=30000)
      print(client.launch_job(request))
  PY
  ```
- Config:
  - worktree: `/Users/romain/marin-wt/moe-gpu-ragged-all-all`
  - sealed snapshot: `grug-moe-ep-ring-20260307`
- Result:
  - The normal `iris job run` path staged the worktree but failed in task setup with `No pyproject.toml found in current directory or any parent directory`.
  - A raw inspect task proved the staged repo under `/app` did include `pyproject.toml`.
  - A raw H100 smoke task successfully initialized JAX and reported 8 CUDA devices.
- Interpretation:
  - The cluster, bundle upload, and task image are all functioning.
  - The remaining problem is specific to the convenience CLI setup path, not GPU availability, so the benchmark can proceed via a raw request without changing the measurement target.
- Next action:
  - Launch the full fixed-shape H100x8 comparison sweep with the raw request path.

### 2026-03-13 16:23 - H100x8 fixed-shape GPU comparison completed
- Hypothesis: on GPU, `ragged_a2a` might become competitive with the current ring EP path, especially at higher EP sizes where communication dominates.
- Command:
  ```bash
  cd /app
  uv sync --quiet --link-mode symlink --python 3.11 --all-packages --no-group dev --extra gpu
  source .venv/bin/activate
  for topk in 2 8; do
    for dist in random runs; do
      for kernel in current ragged_a2a; do
        python lib/levanter/scripts/bench/bench_moe_hillclimb.py \
          --tokens 32768 \
          --hidden 2048 \
          --mlp-dim 768 \
          --experts 128 \
          --topk "$topk" \
          --shared-expert-dim 2048 \
          --distribution "$dist" \
          --kernel "$kernel" \
          --ep-list 1,2,4,8 \
          --bench-pass forward_backward \
          --iters 3 \
          --warmup 1
      done
    done
  done
  ```
- Config:
  - job id: `/romain/raw-gpu-bench-20260313-232319`
  - host: one CoreWeave `h100-8x` worker
  - benchmark mode: `forward_backward`
  - fixed shape: `tokens=32768 hidden=2048 mlp_dim=768 experts=128 shared_expert_dim=2048`
- Result:
  - `topk=2`, `distribution=random`
    - `current`: `52.802 / 35.494 / 18.644 / 11.044 ms`
    - `ragged_a2a`: `52.828 / 60.547 / 30.888 / 16.272 ms`
  - `topk=2`, `distribution=runs`
    - `current`: `52.992 / 36.509 / 19.263 / 10.986 ms`
    - `ragged_a2a`: `53.594 / 62.138 / 33.226 / 18.999 ms`
  - `topk=8`, `distribution=random`
    - `current`: `215.920 / 139.003 / 66.969 / 34.307 ms`
    - `ragged_a2a`: `213.717 / 234.108 / 116.771 / 59.646 ms`
  - `topk=8`, `distribution=runs`
    - `current`: `210.857 / 134.295 / 70.494 / 34.212 ms`
    - `ragged_a2a`: `219.390 / 242.572 / 117.599 / 61.277 ms`
- Interpretation:
  - `EP=1` behaves as the expected control: the two paths are effectively equal.
  - For every measured GPU point with `EP > 1`, `current` beats `ragged_a2a`.
  - Across the completed `EP > 1` matrix, `ragged_a2a` is about `1.47x` to `1.81x` slower than `current`.
  - The result direction is consistent across both tested routing distributions and both tested `topk` settings on this fixed H100x8 case.
- Next action:
  - Update `#3633` with the completed table and conclusion.
  - Keep `current` as the recommended path for this workload unless a more MaxText-faithful implementation or materially different shape shows a different regime.
