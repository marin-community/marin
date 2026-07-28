# Capacity-factor resolution for the B200/GB200 MoE queue

## TL;DR

Before this change, the EP64 scale launcher resolved `SCALE_CAPACITY_FACTOR`
into `GrugModelConfig.capacity_factor`, and model construction passed that value
to `MoEExpertMlp`. The launcher fallback was `1.0`, so `1.0` won when the
environment variable was unset. The `1.25` default in
`levanter.grug._moe.common` did not win on this path because the model always
passed an explicit value.

`DEFAULT_EP_CAPACITY_FACTOR = 1.0` is now the only default definition.
`launch_cw_scale.build_scale_model()` is the only
`SCALE_CAPACITY_FACTOR` reader. The resolved value remains a field of
`GrugRunConfig.model`, which `_run_grug_local()` logs as hyperparameters before
model initialization. A rack leg can grep its hparams for
`"capacity_factor": <expected value>`.

One historical job was mislabeled:
`/mwittmann/ep25d1-qbon-cf115-350-0726-0313` was named as a cf1.15 leg but ran
at cf1.0. The record was repaired before the current D-7 frontier was written.
The replacement cf1.15 leg and the cf1.05/cf1.0625 legs have recorded in-log
provenance. No current D-7 number is known to be invalidated by the `1.25`
library default.

## The three implementations

The three implementations were developed on different lineages. Only
`595958b83` is an ancestor of this branch at `f53f781ce`; `3e149490f` and
`54bbe3d23` are not.

| SHA | Resolution path | Location in that commit |
|---|---|---|
| `595958b83` | Launcher reads the environment, stores the result in the model config, and model construction passes the config field. | `experiments/grug/moe/launch.py:115`, `experiments/grug/moe/launch_cw_scale.py:153`, `experiments/grug/moe/model.py:143,459` |
| `3e149490f` | `MoEMLP.init()` reads the environment while constructing the expert MLP. | `experiments/grug/moe/model.py:454-456` |
| `54bbe3d23` | Module import reads the environment into `_DEFAULT_EP_CAPACITY_FACTOR`; model construction later passes that cached value. | `experiments/grug/moe/model.py:55-59,625` |

The two pre-change constants on this branch also had independent origins:
`experiments/grug/moe/model.py:48` was `1.0` from `4337f40c76`, while
`lib/levanter/src/levanter/grug/_moe/common.py:19` was `1.25` from
`7ae9a9e754`. The latter was the library default for direct
`MoEExpertMlp.init()` and `moe_mlp()` calls.

## Runtime ground truth before the change

The EP64 training path on this branch was:

1. `launch_cw_scale.build_scale_model()` called
   `env_float("SCALE_CAPACITY_FACTOR", 1.0)`.
2. The result was stored in `GrugModelConfig.capacity_factor`.
3. `build_scale_checkpoint()` embedded that model config in
   `GrugMoeLaunchConfig`.
4. `run_grug_moe_trial()` embedded the same model config in `GrugRunConfig`.
5. `_run_grug_local()` logged the complete `GrugRunConfig` through
   `levanter.tracker.log_configuration(config)` before initializing the model.
6. `MoEMLP.init()` passed `cfg.capacity_factor` to `MoEExpertMlp.init()`.

Therefore:

- unset `SCALE_CAPACITY_FACTOR` produced an effective EP64 capacity factor of
  `1.0`;
- a non-empty environment value produced that float;
- the library's `1.25` default was bypassed on this training path;
- the environment was not read a second time during model construction on this
  branch.

Before `595958b83`, the same model construction site passed the experiment
module's `_DEFAULT_EP_CAPACITY_FACTOR = 1.0` directly. An environment variable
could not change it. Pre-knob EP64 scale bundles on this lineage therefore ran
at `1.0`, not `1.25`.

## Reconciliation

The canonical definition is now
`lib/levanter/src/levanter/grug/_moe/common.py:19`:

```python
DEFAULT_EP_CAPACITY_FACTOR = 1.0
```

`levanter.grug.grug_moe` publicly re-exports the constant and uses it for its
low-level defaults. `experiments/grug/moe/model.py` imports it for
`GrugModelConfig`. `experiments/grug/moe/launch_cw_scale.py` imports it for the
unset environment fallback. No model file reads `SCALE_CAPACITY_FACTOR`.

The resolved capacity factor remains in the logged model config. In the JSON
tracker this is an `event="hparams"` record containing
`hparams.model.capacity_factor`; W&B receives the same dataclass-derived
hyperparameters. This supplies the provenance gate required by each rack leg.

The regression tests exercise behavior instead of comparing literals:

- the unset scale launcher and direct Grug MoE construction resolve the same
  capacity factor;
- after the launcher resolves `1.0625`, changing the environment to `1.25`
  before model construction does not change the expert MLP's capacity factor.

The first test failed before the change with `1.0 != 1.25`. The second would
fail if either the `3e149490f` construction-time override or the `54bbe3d23`
module-time override were reintroduced beside the config route.

## Existing experiment provenance

The local experiment record establishes one actual mismatch and its repair:

- `be9db6d7c` records that
  `/mwittmann/ep25d1-qbon-cf115-350-0726-0313` lacked `capacity_factor` in its
  logged hparams because its bundle predated `595958b83`. Despite the job name,
  it ran at cf1.0 and was reused as the same-draw m=0 baseline.
- `392a3d0c8` records the replacement
  `/mwittmann/ep25d1-cf115-m0-350-0726-0434` with
  `"capacity_factor": 1.15` in-log.
- `246f720c5` records
  `/mwittmann/ep25d1-spill3-cf105-350-0726-0603` with
  `"capacity_factor": 1.05`.
- `528ef5765` records
  `/mwittmann/ep25d1-spill3-cf10625-350-0726-0730` with
  `"capacity_factor": 1.0625`.
- `5fb76121d` consolidates these reproduction pointers and explicitly marks the
  first job as cf1.0.

The initial cf1.15 label was wrong, but the corrected queue does not use it as
a cf1.15 measurement. The current D-7 frontier uses the replacement and the
provenance-checked cf1.05/cf1.0625 legs. The code trace also rules out the
shared `1.25` default for pre-knob EP64 scale runs on this lineage.

This audit does not independently recover hyperparameter logs for every legacy
job cited elsewhere in the omnibus. Jobs predating `595958b83` do not contain a
`capacity_factor` field, so their value must be established from their exact
bundle. For the EP64 scale lineage traced here, those bundles passed the model's
explicit `1.0`. A legacy number produced by another entry point that called
`MoEExpertMlp.init()` or `moe_mlp()` without an explicit capacity factor could
have used `1.25`; no such training entry point was found in the queue trace.
