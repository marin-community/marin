# Environment — how to run on your H100

You have a dedicated **1×H100 dev pod**. A helper script `./h100` runs commands on it.

## Running code on the H100

```bash
./h100 python -c "import jax; print(jax.devices())"   # should print a CudaDevice (H100)
./h100 python bench_fp8_ragged_dot.py                 # your microbench
./h100 python -m pytest lib/haliax/tests/test_fp8_ragged.py -x   # your tests
./h100 nvidia-smi
```

How it works: `./h100 <cmd>` syncs your editable source (`lib/haliax/` plus scripts at the
worktree root) onto the pod's `/app` workspace, then runs `<cmd>` inside the pod's GPU
virtualenv on the H100. Edit files normally here in the worktree; they are re-synced on every
call. The pod already has the full marin repo at `/app` (at origin/main) with `jax[cuda13]`
and a working GPU/cuDNN install — **you do not need to run `uv sync` or install anything**.

## Notes

- Keep jobs microbench-scale and quick. Iterate freely.
- Do **not** touch the cluster lifecycle (no stop/restart/scale). The coordinator owns the pod.
- If you hit any of these — `No PTX compilation provider`, `Can't find libdevice`, a cuDNN
  version mismatch at compile, a `uv` python-version rejection, or an OOM / exit-137 — that is a
  **known cluster-image issue, not your bug**. Note it in `NOTES.md` and tell the coordinator;
  do not build elaborate workarounds for it.
- For anything else that blocks you, see the "This is solvable — finish it" section of `BRIEF.md`:
  write down where you're stuck and ask the coordinator for a hint. Don't give up.
