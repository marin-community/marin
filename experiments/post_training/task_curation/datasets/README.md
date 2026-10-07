# Dataset declarations

Each source module exposes `pipeline() -> RlDataPipeline`. The declaration owns
its canonical source name, pinned input, file selection, intended use and choice
of reusable conversion policy. [sources.py](../sources.py) maps names to these
factories. It constructs metadata without downloading data or submitting jobs.

## Binding a source

[shared.py](shared.py) provides `hf_pipeline` for pinned Hub inputs and
`executable_pipeline` for converters that require a selected execution image.
Reusable normalization, checks and review rubrics come from
[taskcompendium.datasets](../../../../lib/taskcompendium/src/taskcompendium/datasets/README.md).

```python
from experiments.post_training.task_curation.sources import rl_data_pipelines

source = rl_data_pipelines()["MarinSkyRL:math500"]
recipe = source.recipe(runtime)
step = source.bind(config, runtime)
```

`config` is a `SourcePipelineConfig`; `runtime` is a `SourceRuntimeConfig`.
`recipe(runtime)` resolves the selected inputs, policy and grader requirements.
`bind(config, runtime)` wraps the reusable source procedure in an ArtifactStep
with acquisition dependencies, output location and artifact identity.

For sources with an optional `runtime_binding`, `runtime.images[source_key]`
selects the backend and immutable grader image. The binding declares the TaskSpec
grader, private resources and control suite during conversion. Without that
entry, the source keeps its unbound grading contract. Executable converters
require an explicit runtime entry. Credentials alone never select an environment.

At execution time the runtime reads the serialized TaskSpec, stages private
files and runs its declared command or VerifyIT descriptor. A source scorer must
already be installed in the selected image, or supplied as pinned task source.
The runtime does not import the dataset declaration. Common graders live in
VerifyIT; custom grading semantics remain with the source task or package.
Missing goldens skip their positive control, while other available checks run.

| Directory | Contents |
| --- | --- |
| [skyrl/](skyrl/) | MarinSkyRL datasets, including IFEval and code/SQL grader bindings. |
| [tasktrove/](tasktrove/) | TaskTrove release sources and archived task contracts. |
| [nemotron_ultra/](nemotron_ultra/) | Ultra blend components and source grader bindings. |
| [arc/](arc/), [reasoning_gym/](reasoning_gym/), [rewardkit/](rewardkit/) | Shared bindings used by declarations from several families. |
| [`math.py`](math.py) | Original TaskTrove SymPy grader binding for math sources. |

See the [campaign overview](../README.md) for execution and sidecars, the
[image recipes](../images/README.md) for current runtime requirements, and
[GOAL.md](../../../../GOAL.md) for status and proposed cleanup.
