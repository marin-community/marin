# Grug MoE JAX Benchmark Reproduction Guide

This guide reproduces the final benchmark and short-run quality checks from the Grug MoE JAX `0.8.0` vs `0.9.2` experiment.

It is written against:

- branch: `research/grug-moe-jax-regression`
- commit: `7686a6b9c384cc62acf9424307c1fc3be210ee04`

If you run a different commit, do not expect exact agreement with the tables below.

## What This Reproduces

- `v5p-8` steady-state training benchmark, 3 runs per stack
- `v4-8` steady-state training benchmark, 3 runs per stack
- one-step quality check on `v5p-8`

The benchmark harness is `scripts/grug_moe_jax_bench.py`. It runs the Grug MoE baseline config from `experiments/grug/moe/launch.py`, disables checkpointing, writes a JSON tracker log, and prints a final JSON report block to stdout.

## Expected Final Results

### `v5p-8`

| Stack | Tok/s mean | Duration mean (s) | MFU mean |
| --- | ---: | ---: | ---: |
| `jax==0.8.0`, `jaxlib==0.8.0`, `libtpu==0.0.24` | `184207.81` | `0.711545` | `22.56346` |
| `jax==0.9.2`, `jaxlib==0.9.2`, `libtpu==0.0.38` | `192244.34` | `0.681800` | `23.54784` |

### `v4-8`

| Stack | Tok/s mean | Duration mean (s) | MFU mean |
| --- | ---: | ---: | ---: |
| `jax==0.8.0`, `jaxlib==0.8.0`, `libtpu==0.0.24` | `108145.62` | `1.211997` | `22.10989` |
| `jax==0.9.2`, `jaxlib==0.9.2`, `libtpu==0.0.38` | `111580.33` | `1.174688` | `22.81210` |

### `v5p-8` one-step quality check

Expected old-vs-new absolute diffs:

- loss: `8.58e-6`
- router load-balancing loss: `6.20e-6`
- router z-loss: `1.72e-5`
- total parameter norm: `0.0`

## Preconditions

- You have this repo checked out locally.
- You can allocate Iris dev TPUs.
- Your local machine has `uv`.
- The worker can resolve the Llama 3.1 tokenizer.

If the worker does not have a cached tokenizer snapshot and cannot access Hugging Face, this experiment will fail. In the original run, I used:

```bash
/home/ubuntu/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b
```

If that path does not exist on your worker, fall back to `meta-llama/Meta-Llama-3.1-8B` and make sure `HF_TOKEN` is available.

## 1. Pin The Exact Code

```bash
cd ~/dev/marin-wt/tpu-dep-hell
git fetch origin
git checkout research/grug-moe-jax-regression
git pull --ff-only origin research/grug-moe-jax-regression
git checkout 7686a6b9c384cc62acf9424307c1fc3be210ee04
```

Optional sanity check:

```bash
git rev-parse HEAD
```

Expected output:

```text
7686a6b9c384cc62acf9424307c1fc3be210ee04
```

## 2. Allocate A Dev TPU

Use the Iris config from this repo:

```bash
export IRIS_CONFIG=lib/iris/examples/marin.yaml
```

### `v5p-8`

Open terminal A and keep it running:

```bash
cd ~/dev/marin-wt/tpu-dep-hell
uv run python scripts/iris/dev_tpu.py \
  --config "$IRIS_CONFIG" \
  --tpu-name grug-jax-v5p-repro \
  allocate \
  --tpu-type v5p-8 \
  --zone us-central1-a \
  --sync-path .
```

Open terminal B and connect:

```bash
cd ~/dev/marin-wt/tpu-dep-hell
uv run python scripts/iris/dev_tpu.py \
  --config "$IRIS_CONFIG" \
  --tpu-name grug-jax-v5p-repro \
  connect
```

### `v4-8`

When you are ready for the `v4-8` portion, use a separate session name:

```bash
cd ~/dev/marin-wt/tpu-dep-hell
uv run python scripts/iris/dev_tpu.py \
  --config "$IRIS_CONFIG" \
  --tpu-name grug-jax-v4-repro \
  allocate \
  --tpu-type v4-8 \
  --zone us-central2-b \
  --sync-path .
```

Then connect from another terminal:

```bash
cd ~/dev/marin-wt/tpu-dep-hell
uv run python scripts/iris/dev_tpu.py \
  --config "$IRIS_CONFIG" \
  --tpu-name grug-jax-v4-repro \
  connect
```

## 3. Set Up The Worker Shell

Run this once on the TPU worker after connecting:

```bash
source ~/.local/bin/env
cd ~/marin
source .venv/bin/activate

export TOKENIZER=/home/ubuntu/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b
if [ ! -d "$TOKENIZER" ]; then
  export TOKENIZER=meta-llama/Meta-Llama-3.1-8B
fi
```

Install two helper shell functions:

```bash
set_stack() {
  pip install -U "jax==$1" "jaxlib==$1" "libtpu==$2"
}

run_bench() {
  local run_id=$1
  local tpu_type=$2
  local steps=$3
  local warmup=$4
  rm -rf /tmp/grug-moe-jax-bench /tmp/grug-moe-jax-bench-logs
  mkdir -p /tmp/grug-moe-jax-bench /tmp/grug-moe-jax-bench-logs
  python scripts/grug_moe_jax_bench.py \
    --run-id "$run_id" \
    --steps "$steps" \
    --warmup-steps "$warmup" \
    --batch-size 32 \
    --tpu-type "$tpu_type" \
    --tokenizer "$TOKENIZER" | tee "/tmp/${run_id}.out"
}
```

## 4. Reproduce The `v5p-8` Results

### Quality check and 3x throughput runs

```bash
set_stack 0.9.2 0.0.38
run_bench grug-jax-092-v5p-math    v5p-8 1 0
run_bench grug-jax-092-v5p-7step   v5p-8 7 2
run_bench grug-jax-092-v5p-7step-b v5p-8 7 2
run_bench grug-jax-092-v5p-7step-c v5p-8 7 2

set_stack 0.8.0 0.0.24
run_bench grug-jax-080-v5p-math    v5p-8 1 0
run_bench grug-jax-080-v5p-7step   v5p-8 7 2
run_bench grug-jax-080-v5p-7step-b v5p-8 7 2
run_bench grug-jax-080-v5p-7step-c v5p-8 7 2

set_stack 0.9.2 0.0.38
```

What to expect:

- The one-step run uses `steps=1`, `warmup=0`.
- The throughput runs use `steps=7`, `warmup=2`.
- Measured steps are `2..6`.
- Step `0` and step `1` are still in the warmup/compile regime and should not be used for the final throughput claim.

## 5. Reproduce The `v4-8` Results

Run the same shape on a `v4-8` worker:

```bash
set_stack 0.9.2 0.0.38
run_bench grug-jax-092-v4-7step-a v4-8 7 2
run_bench grug-jax-092-v4-7step-b v4-8 7 2
run_bench grug-jax-092-v4-7step-c v4-8 7 2

set_stack 0.8.0 0.0.24
run_bench grug-jax-080-v4-7step-a v4-8 7 2
run_bench grug-jax-080-v4-7step-b v4-8 7 2
run_bench grug-jax-080-v4-7step-c v4-8 7 2

set_stack 0.9.2 0.0.38
```

## 6. Inspect The Artifacts

Each run should:

- exit `0`
- write stdout capture to `/tmp/<run-id>.out`
- write tracker events to `/tmp/grug-moe-jax-bench-logs/<run-id>.log`

The harness prints a JSON report between:

- `<run-id>_REPORT_START`
- `<run-id>_REPORT_END`

Example extraction:

```bash
python - <<'PY' /tmp/grug-jax-092-v5p-7step.out
import json
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text()
match = re.search(r"_REPORT_START\n(.*?)\n[^\n]*_REPORT_END", text, re.S)
if not match:
    raise SystemExit("report block not found")

report = json.loads(match.group(1))
print("versions:", report["versions"])
print("tok/s mean:", report["steady_state"]["tokens_per_second"]["mean"])
print("duration mean:", report["steady_state"]["duration"]["mean"])
print("mfu mean:", report["steady_state"]["mfu"]["mean"])
print("finish summary keys:", sorted((report["finish_summary"] or {}).keys())[:10])
PY
```

## 7. Aggregate The 3x Runs

If you want a quick local aggregation, use this script on the TPU worker after all runs are done:

```bash
python - <<'PY'
import json
import pathlib
import re
import statistics

RUNS = {
    "v5p-8 0.8.0": [
        "/tmp/grug-jax-080-v5p-7step.out",
        "/tmp/grug-jax-080-v5p-7step-b.out",
        "/tmp/grug-jax-080-v5p-7step-c.out",
    ],
    "v5p-8 0.9.2": [
        "/tmp/grug-jax-092-v5p-7step.out",
        "/tmp/grug-jax-092-v5p-7step-b.out",
        "/tmp/grug-jax-092-v5p-7step-c.out",
    ],
    "v4-8 0.8.0": [
        "/tmp/grug-jax-080-v4-7step-a.out",
        "/tmp/grug-jax-080-v4-7step-b.out",
        "/tmp/grug-jax-080-v4-7step-c.out",
    ],
    "v4-8 0.9.2": [
        "/tmp/grug-jax-092-v4-7step-a.out",
        "/tmp/grug-jax-092-v4-7step-b.out",
        "/tmp/grug-jax-092-v4-7step-c.out",
    ],
}

def load_report(path):
    text = pathlib.Path(path).read_text()
    match = re.search(r"_REPORT_START\n(.*?)\n[^\n]*_REPORT_END", text, re.S)
    if not match:
        raise RuntimeError(f"report block not found: {path}")
    return json.loads(match.group(1))

for label, paths in RUNS.items():
    reports = [load_report(path) for path in paths]
    tok = [r["steady_state"]["tokens_per_second"]["mean"] for r in reports]
    dur = [r["steady_state"]["duration"]["mean"] for r in reports]
    mfu = [r["steady_state"]["mfu"]["mean"] for r in reports]
    print(label)
    print("  tok/s mean :", statistics.fmean(tok))
    print("  tok/s stdev:", statistics.stdev(tok))
    print("  dur mean   :", statistics.fmean(dur))
    print("  dur stdev  :", statistics.stdev(dur))
    print("  mfu mean   :", statistics.fmean(mfu))
    print("  mfu stdev  :", statistics.stdev(mfu))
PY
```

## 8. Interpreting Success

I would treat the experiment as successfully reproduced if all of the following are true:

- all benchmark runs exit `0`
- each run emits both the JSON log file and the final stdout report
- the `0.9.2` stack remains ahead of the `0.8.0` stack on both `v5p-8` and `v4-8`
- your throughput numbers are within roughly `1%` of the expected means
- the one-step quality check shows only tiny drift, on the order listed above

## 9. Common Failure Modes

- Missing tokenizer access:
  - fix by using the cached tokenizer snapshot path or exporting `HF_TOKEN`
- Wrong commit:
  - fix by checking out `7686a6b9c384cc62acf9424307c1fc3be210ee04`
- TPU contention or preemption:
  - rerun the affected benchmark on a fresh dev TPU
- Using the short `steps=2`, `warmup=1` shape:
  - do not use that for the final claim; it under-samples steady state on `v4-8`

## 10. Cleanup

Release the dev TPU sessions after you finish:

```bash
cd ~/dev/marin-wt/tpu-dep-hell

uv run python scripts/iris/dev_tpu.py \
  --config "$IRIS_CONFIG" \
  --tpu-name grug-jax-v5p-repro \
  release

uv run python scripts/iris/dev_tpu.py \
  --config "$IRIS_CONFIG" \
  --tpu-name grug-jax-v4-repro \
  release
```
