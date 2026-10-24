# Nightly end-to-end gates

The nightly runs synchronous OPD, Grug Megatron training and the asynchronous
CatCount learning canary on H100s. Colocated synchronous RL is not a production
mode; CatCount covers learning. All policy updates use Megatron and the frozen
root environment.

| file | role |
| --- | --- |
| `run_opd_h100.sh` | run synchronous OPD with separate policy, rollout, and teacher roles |
| `run_grug_megatron.sh` | run Grug parity, training, and serving gates on four H100s |
| `run_cat_count_h100.sh` | submit the CatCount coordinator and gate sampled learning on four H100s |
| `gate.py` | score a training run against its spec |
| `specs/cat-count-canary-qwen2.5-0.5b-async.json` | CatCount sampled learning and per-step mechanism requirements |

## How the gate sees the run

The trainer writes every tracker payload to stdout as a `WANDB_MIRROR` line.
The gate reads these metrics from the run log:

```
WANDB_MIRROR kind=train step=2 metrics={"policy/policy_loss": 0.41, "reward/avg_raw_reward": 0.25, ...}
```

`gate.py` parses those, counts distinct training steps, and checks the final payload
against the spec's required metrics and bounds. A spec can also require evidence across
the run: finite values at every step, minimum observation counts, first-to-last-window
improvement, and a minimum number of observations above or below a threshold. Training and evaluation
payloads are separate streams. A selected observation must exist. Duplicate
payloads for one stream and step count once;
conflicting copies fail. The gate exits non-zero with one line per violation.

## Metric gates

Each `metric_gates` row selects a `kind` (`train` or `eval`) and a metric.
Every selected payload must contain a finite value. `min_observations` counts
payloads, so an evaluation every five steps has fewer observations than the
training stream. `step` selects a numbered step, `first`, or `last`;
`max_step` includes observations through that completed step.

A `bounds` range applies to every observation by default. With `minimum_count`,
it requires that many observations inside the range. Endpoint inclusion is
controlled by `inclusive_minimum` and `inclusive_maximum`. A `trend` compares
the mean of the first and last `window` observations; too few observations fail.
Negative `min_improvement` values permit a bounded decrease.
The gate returns typed failures for missing, nonfinite or out-of-range observations. Top-level `finite_metrics`
and `bounds` check the final training payload.

```json
[
  {"kind": "eval", "metric": "eval/train/avg_score",
   "min_observations": 1, "step": 0},
  {"kind": "eval", "metric": "eval/train/avg_score",
   "min_observations": 2, "trend": {"window": 1, "min_improvement": 0.2}}
]
```

CatCount requires sampled training-prompt reward in [0.10, 0.45] at step 0,
then at least one score ≥ 0.65 by step 30. The launcher evaluates every five
steps and stops at the first qualifying score. It requires
`policy/dp_weight_checksum_mismatch` = 0 after every optimizer step. The
synchronous lane is a manual launcher option outside CI.

## CatCount nightly

`cat-count-h100` submits a 4-CPU, 16-GB-memory, 8-GB-disk coordinator and two
worker tasks with two H100s and 65 CPUs each. It uses seed 17, behavior clipping
and staleness 2, without checkpoints or HF export. The runner clones Marin main,
sets the external MarinSkyRL source to the commit under test, and runs Marin's
`config/update-external.py MarinSkyRL`. That pin supplies both the launcher
package and the GPU runtime. Scheduled and manually dispatched workflows run
all three lanes. The CatCount job summary reports dashboard readiness without
affecting its learning result.

Each attempt has a 20-minute deadline including queue time. A failure with no
native training row is reported as `INFRASTRUCTURE_FAILURE` and retried once.
A failed job after training starts, or a failed metric gate, is `GATE_FAILURE`
and is not retried. The script records each attempt's wall time and exit status;
`OK against ...cat-count-canary-qwen2.5-0.5b-async.json` is the passing gate line.
The workflow uploads the combined native log and cancels its own named jobs
in the shared cleanup step.

## Ray startup collisions

Ray instances on the same node can read each other's temporary session state.
The second instance fails at startup with "Session name ... does not match persisted value. Perhaps there was an error connecting to Redis."

The Grug tests start Ray through `initialize_ray`; CatCount starts it through
the Marin task runtime. Both target `cw-rno2a`. A Ray startup error with that
message indicates two instances sharing a host; inspect their placement.

## Running it by hand

The gate is pure stdlib and runs anywhere, against any run log:

```bash
python3 skyrl-train/ci/marin_nightly/gate.py \
    --log nightly-run.log \
    --spec skyrl-train/ci/marin_nightly/specs/cat-count-canary-qwen2.5-0.5b-async.json \
    --wall-clock-seconds 900
```

The Grug lane runs `tests/gpu/test_grug_megatron.py`, the Levanter parity oracle
and the two-GPU CP2
FlashAttention forward/backward smoke with the frozen Megatron runtime closure;
see `docs/grug-megatron-training.md` for the Grug tests.
