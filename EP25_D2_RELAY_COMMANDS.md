# EP25 Direction 2 MXFP8 Relay Command

Run this command from
`/home/marin/projects/marin/.worktrees/ep25-d2-bakeoff`.

Submit only this numerical job. Do not submit the stale v2 EP4 or rack jobs,
and do not stage later ladder jobs until this numerical gate is green.

```bash
ep25_pythonpath=$(find "$PWD/lib" -mindepth 2 -maxdepth 2 -type d -name src -print | paste -sd:)
env IRIS_USER=mwittmann PYTHONPATH="$ep25_pythonpath:$PWD" \
  .venv/bin/iris --cluster=marin job run --no-wait \
  --target-cluster cw-us-east-08a --priority interactive \
  --job-name ep25d2-mxfp8-numerics-20260725-v3 \
  --enable-extra-resources --gpu GB200x1 --cpu 16 --memory 96GB \
  --extra gpu \
  -- python experiments/grug/moe/standalone/check_mxfp8_expert_mlp.py \
  --out /tmp/ep25d2-mxfp8-numerics-20260725-v3.json
```

The script prints `CUTLASS_ENV_SENTINEL` before kernel compilation. Harvest
that line even if compilation fails. It identifies the imported CUTLASS
module and extension, installed CUTLASS distributions, CUDA payload owner,
`CUDA_TOOLKIT_PATH`, `LD_LIBRARY_PATH`, and the libNVVM path selected by
`cuda.pathfinder`.

Gate: the job must exit zero, all reported relative Frobenius errors must be
below 0.1, and both empty-expert weight-gradient checks must be exactly zero.
Operational friction does not close the MXFP8 direction; a failed v3 is a
checkpoint escalation under the amended round-6 fleet policy.
