# Tutorial: from one scalar to a shared text-and-science model

## Orientation

A next-token language model takes token embeddings, mixes information with a
transformer, and produces logits for a target token. This experiment keeps that
training loop but changes what one position means. A position can represent a
scientific value such as `future[time=1, cell=0.5]` instead of a word in a
sentence.

This tutorial calls one such scientific-value row a **record**.

The examples below build that idea in four steps:

1. describe a prediction from one scalar to another;
2. inspect how that description becomes transformer inputs and labels;
3. extend the same description to a small advection problem, then request
   repeated refinement;
4. train one transformer on next-token text and scientific records.

The code is a prototype under `experiments/`, not a production Marin API. It
uses a transformer implementation from this repository, but no prior Marin
knowledge is needed.

Generate the checked-in debug reports from the repository root with:

```bash
uv run python -m experiments.probabilistic_dataflow.debug_render
```

The command rewrites [`debug_outputs/`](debug_outputs/README.md). Add `--check`
to verify the checked-in reports without changing them.

## 1. Describe a scalar prediction

Suppose `current=3` is observed and the training label is `future=5`. Both
values are discrete IDs from `0` through `15`. The goal is to learn
`p(future | current)`.

The first layer of the Python description states what depends on what:

- The experiment's `.dsl` module is a small Python interface for describing
  values and their dependencies; users do not write model tensors directly.
- A `FieldType` says how a scientific value is represented. `bins=16` means
  that this scalar is encoded as one of 16 value tokens.
- A `Program` is a named dependency graph.
- Inside a program, `variable` introduces a value that can be supplied as
  input, while `sample` introduces a value that the model must predict.
- `learned_joint(current)` says that the model learns the distribution of the
  sampled value using `current` as input. The name `scalar_transition` labels
  that learned relationship in debug output.

```python
from experiments.probabilistic_dataflow.dsl import (
    FieldType,
    Program,
    learned_joint,
)

measurement = FieldType("measurement", bins=16)

with Program("scalar_forecast") as program:
    current = program.variable("current", measurement)
    future = program.sample(
        "future",
        measurement,
        learned_joint(current, name="scalar_transition"),
    )
```

The second layer states which prediction to make. A `Query` names:

- `given`: values supplied as model inputs;
- `targets`: values the model must produce.

Here, `current` is given and `future` is the target:

```text
given: current
targets: future
```

The checked-in `scalar_forecast_problem()` in [`synthetic.py`](synthetic.py)
constructs this program and its query. The following sections use that runnable
object.

## 2. Read the scalar debug dump

The compiler translates the Python description into a sequence of simpler
snapshots. Each snapshot is called an intermediate representation, or IR. These
IRs are debug output: users do not write them by hand.

The remaining objects bridge from the description to tensors. A
`ConcreteExample` holds realized values for one training case; it includes
`future=5` only so that value can become a training label. `TokenCodec` maps
value IDs and the `<query>` placeholder to vocabulary IDs. `compile_query`
chooses a schedule of model calls, and `lower_to_transformer` converts that
schedule and example into transformer inputs and labels:

```python
from experiments.probabilistic_dataflow.compiler import (
    ConcreteExample,
    TokenCodec,
    compile_query,
    lower_to_transformer,
)
from experiments.probabilistic_dataflow.synthetic import scalar_forecast_problem

problem = scalar_forecast_problem()
program = problem.program

example = ConcreteExample(
    id="scalar-0",
    program_name=program.name,
    values={"current": (3,), "future": (5,)},
)

plan = compile_query(problem.query)
execution = lower_to_transformer(program, plan, example, TokenCodec())
```

The full result is [`debug_outputs/scalar.md`](debug_outputs/scalar.md). Its four
parts answer four different questions.

### Program graph: what depends on what?

The report calls a learned distribution a **factor**. `scalar_transition` is
the factor that connects `current` to `future`:

```text
%0 current : variable measurement[scalar]
    |
    v
%1 future  : sample measurement[scalar], factor=scalar_transition
```

This is the direct equivalent of a PyTorch module's data dependency:
`future` is modeled from `current`.

### Conditional query: what is known now?

In this table, **given** means supplied as model input and **target** means
requested as output:

```text
given:   %0 current
targets: %1 future
```

During training, the concrete example contains the realized target `5` so it can
be used as a label. During prediction, `future` is absent and must be generated.

### Inference plan: how many model calls are needed?

An inference plan is a schedule of transformer calls. A `parallel` call
predicts all of its target coordinates at once from its context. A dependency
would name an earlier call whose generated values are also needed. This scalar
plan has no such dependency:

```text
call 0: parallel(context=current, targets=future, dependencies=none)
```

There is only one target coordinate, so the default plan needs one call.

### Transformer execution: what enters the model?

The final IR contains the records introduced in the orientation. Each record
has one content token plus an embedding that identifies the scientific quantity
represented by that token. The report calls that quantity the record's
**scientific identity**.

| Record role | Scientific identity | Model input | Training label |
| --- | --- | --- | --- |
| known context | `scalar_forecast.current[scalar]` | `value:3` | none |
| requested target | `scalar_forecast.future[scalar]` | `<query>` | `value:5` |

This is ordinary cross-entropy training. The model receives `value:3` and a
`<query>` placeholder. At the target record it produces logits over the same
token vocabulary, and the loss compares those logits with `value:5`. The label
`value:5` is never included in the model input.

Text models usually attach an order-dependent position encoding to each token
and use causal attention, so position 4 can read positions 0 through 4. The
scientific call instead uses:

- full attention, so every record in the example can read every other record;
- a learned scientific-identity embedding, so `current` and `future` remain
  distinguishable;
- rotary position `0` for every record. A rotary position is the
  sequence-position signal used by this transformer. Setting it to zero removes
  the arbitrary row order from the calculation.

The report happens to print `current` before `future`, but that order has no
scientific meaning. Reordering complete records reorders their outputs without
changing the values computed for them.

## 3. Extend the query to an advection field

The advection example represents a quantity moving across a one-dimensional
grid. It observes four initial cell values and twelve forcing values (an
external influence at three times and four cells), then predicts twelve future
values on the same time-by-cell grid.

`MeshAxis` describes named spatial coordinates. `OrderedAxis` describes an
ordered index such as time. Passing these axes to `FieldType` makes one value
token for every coordinate combination. The scalar program then changes only
in its field shapes and inputs:

```python
from experiments.probabilistic_dataflow.dsl import MeshAxis, OrderedAxis

cell = MeshAxis("cell", 4, coordinates=((0.0,), (0.25,), (0.5,), (0.75,)))
time = OrderedAxis("time", 3)

state = FieldType("state", (cell,), bins=16)
forcing_type = FieldType("forcing", (time, cell), bins=16)
trajectory = FieldType("state_trajectory", (time, cell), bins=16)

with Program("synthetic_advection") as program:
    initial = program.variable("initial", state)
    forcing = program.variable("forcing", forcing_type)
    future = program.sample(
        "future",
        trajectory,
        learned_joint(initial, forcing, name="advection_transition"),
    )
```

Its query uses these roles:

```text
given: initial, forcing
targets: future
```

The complete `advection_problem()` in [`synthetic.py`](synthetic.py) constructs
this program and its query.

The change in model inputs is mechanical:

| | Scalar | Advection |
| --- | ---: | ---: |
| Known records | 1 | 4 initial + 12 forcing |
| Target records | 1 | 12 future |
| Total records | 2 | 28 |

Each field element becomes one record. For example, the checked-in report
contains:

```text
input:  <query>
where:  synthetic_advection.future[time=1,cell=(0.5,)]
label:  value:15
```

The scientific-identity embedding tells the transformer which field, time, and
mesh coordinate the query refers to. The current prototype learns a separate
embedding for each complete identity; it does not yet derive geometry from the
numeric coordinate `0.5`.

The default plan still makes one call with full attention and zero rotary
positions. `learned_joint` requested one joint distribution over all twelve
future values. A **marginal** here is the distribution for one coordinate by
itself. The current plan emits twelve such distributions in parallel, so the
report includes this diagnostic:

```text
factor synthetic_advection:future:2 is approximated as 12 parallel token marginals
```

Sampling those distributions independently cannot represent correlations among
the generated values, even though every query record reads the same observed
context. The diagnostic makes that approximation explicit.

See [`debug_outputs/advection.md`](debug_outputs/advection.md) for all 28
records.

## 4. Request repeated refinement

The high-level program says what distribution is wanted. An explicit inference
plan says how to approximate it with model calls. `ParallelQuery` requests one
simultaneous proposal for all targets. `Refine` wraps that proposal in repeated
calls that reconsider low-confidence values. Here, `steps=3` means one proposal
plus two refinement calls, and `resample_fraction=0.25` requests replacement of
the least-confident quarter of the future field:

```python
from experiments.probabilistic_dataflow.compiler import ParallelQuery, Refine, compile_query
from experiments.probabilistic_dataflow.synthetic import advection_problem

problem = advection_problem()

refinement_plan = compile_query(
    problem.query,
    Refine(
        proposal=ParallelQuery(problem.targets),
        steps=3,
        resample_fraction=0.25,
    ),
)
```

```text
call 0: propose future from initial and forcing
call 1: refine future using call 0's proposal
call 2: refine future using call 1's proposal
```

The dependency between calls carries generated future values forward. It does
not impose a left-to-right order on the twelve cells inside a call.

This part is only a compiler demonstration. The generated debug execution uses
the example's true `future` values as refinement context. Code that performs
sampling would need to substitute the preceding call's proposal, and the tiny
training function in the next section does not train the refinement calls. The
report states this limitation directly above the alternate execution tables.

## 5. Train one transformer on text and science

Grug is this repository's transformer backbone. `CrossDomainTransformer` adds
the scientific-identity embedding to one Grug model so that the same parameters
can process text and scientific records. The cross-domain smoke test is a tiny
end-to-end training check with one text loss and one advection loss:

```python
from experiments.probabilistic_dataflow.training import train_cross_domain_smoke

result = train_cross_domain_smoke(steps=80, examples_per_task=8, seed=0)
```

Both tasks update the same token embedding table, transformer blocks, and output
projection. The scientific task also updates a scientific-identity embedding
table that contributes zero to text inputs. Each batch supplies positions,
attention, and labels appropriate to its task.

| | Synthetic text | Synthetic advection |
| --- | --- | --- |
| Input unit | word-like token | scientific record |
| Position signal | rotary positions `0..6` | scientific identity plus rotary position `0` |
| Attention | causal | full within one example |
| Label | next token | value at the same `<query>` record |

The first text example in
[`debug_outputs/cross-domain.md`](debug_outputs/cross-domain.md) contains the
familiar shifted labels:

```text
input <text:the>   -> label <text:ocean>
input <text:ocean> -> label <text:field>
```

The scientific batch uses the layout from the previous sections:

```text
input value:13 at initial[cell=(0.0,)] -> no loss
input <query> at future[time=0,cell=(0.0,)] -> label value:5
```

Text and science are evaluated as separate batches because they require
different attention masks. Their two losses are averaged before one optimizer
update, so both tasks change the shared parameters.

With seed 0, the 80-step smoke run produced:

| Metric | Initial | Final |
| --- | ---: | ---: |
| combined loss | 4.1909 | 0.2022 |
| text training accuracy | 0% | 75% |
| scientific training accuracy | 0% | 100% |

This is a memorization and compatibility check on tiny synthetic datasets. It
does not test held-out scientific prediction, language quality, transfer between
text and science, or the refinement runtime.
