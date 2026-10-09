# Placement and current examples

Use these examples for the decision they illustrate, not as templates to copy whole.

| Need | Location | Current example |
| --- | --- | --- |
| Reusable tokenization behavior and its result type | `lib/marin/src/marin/processing/tokenize/` | `tokenize.py` provides the work function and `TokenizedCache`. |
| Dataset identity, source pins, and selection | `experiments/datasets/` | `nemotron.py` builds keyed handles; mixture weights belong to the consuming experiment. |
| Experiment-specific DAG and tests | Beside the experiment | `experiments/post_training/task_curation/harbor_export.py` binds normalized inputs to Harbor export; `tests/test_harbor_export.py` checks export behavior and the RL smoke graph. |
| Custom analysis and training stages | `experiments/` for binding; `lib/` for reusable work | `experiments/references/reference_scaling_suite.py` declares dependencies, versions, resources, and a result-dependent training config. |

`experiments/references/reference_training_pipeline.py` shows a two-pass workflow when the second graph depends on data discovered by the first. Do not make every pipeline two-pass. The reference pipeline and task curation both keep workflow decisions in `experiments/**`, but also contain concrete work tied to one experiment; promote that work only when it has a real reuse case.

The lazy-artifact migration in [#6649](https://github.com/marin-community/marin/pull/6649) replaced import-time `ExecutorStep` globals with handles built at run time. [Dataset catalog work](https://github.com/marin-community/marin/issues/6777#issuecomment-4848352552) established `experiments/datasets/` as the home for dataset definitions. Current `lib/marin/src/marin/experiment/{data,train,evaluation}.py`, `lib/marin/src/marin/rl/skyrl.py`, and older `StepSpec` factories in `lib/marin/src/marin/datakit/` still build workflow objects. Treat them as existing exceptions to migrate when touching their API, not as placement guidance for new code.
