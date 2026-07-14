# Tutorial: from one scalar to a shared text-and-science model

## Orientation

A language model normally treats each input position as a token in a sentence.
This prototype can instead treat one position as a scientific value such as
`future[time=1, cell=0.5]`. We call that position a **record**.

The transformer is still an ordinary dense model. The DSL describes which
scientific records to construct, which records may attend to each other, and
which records have training labels.

The examples build that idea in five steps:

1. predict one scalar from another;
2. read the compiler's debug output;
3. extend the same program to an indexed advection field;
4. request lower-level control over factorization and refinement;
5. train one transformer on scientific records and ordinary text.

The code is an experiment, not a production Marin API. Generate the reports
used below from the repository root with:

```bash
uv run python -m experiments.probabilistic_dataflow.debug_render
```

## 1. Predict one scalar

Suppose the current measurement is the integer `3`, and a training example says
the future measurement is `5`. Both values are discrete IDs from `0` through
`15`. We want to model `p(future | current)`.

```python
from experiments.probabilistic_dataflow.dsl import (
    AttentionPattern,
    Budget,
    DocumentSpec,
    FieldType,
    InferenceProgram,
    PositionMode,
)

measurement = FieldType("measurement", bins=16)
scientific_document = DocumentSpec(
    attention=AttentionPattern.FULL,
    positions=PositionMode.SCIENTIFIC,
)

program = InferenceProgram(
    "scalar_forecast",
    budget=Budget(model_calls=1, generated_tokens=1),
)
current = program.input_value("current", measurement)
future = program.generate(
    "future",
    measurement,
    context=(current,),
    document=scientific_document,
    factor_name="scalar_transition",
)
program.finish(future)
```

`FieldType` describes the kind of value. Here `bins=16` means one token chosen
from 16 possible value tokens. `input_value` introduces a value supplied to a
model call. `generate` does two things together:

- it defines `future` as a scientific value modeled from `current`;
- it adds a transformer call that will generate `future` from that context.

`finish(future)` names the program output and checks the call and token budgets.
There is no separate query or lowering-strategy object.

The `DocumentSpec` is explicit because attention and position have semantic
consequences. This factor has no meaningful left-to-right order, so it uses full
attention and scientific identities rather than sequence positions.

## 2. Read the scalar debug dump

A concrete training example supplies realized values:

```python
from experiments.probabilistic_dataflow.compiler import TokenCodec, lower_to_transformer
from experiments.probabilistic_dataflow.synthetic import scalar_forecast_example

example = scalar_forecast_example(program)  # current=3, future=5
execution = lower_to_transformer(program, example, TokenCodec())
```

`future=5` is present in the example so it can become a cross-entropy label. It
is not fed to the model. The full rendering is
[`debug_outputs/scalar.md`](debug_outputs/scalar.md).

### Inference Program Values

The first graph shows scientific values and dependencies:

```text
%0 current : input measurement[scalar]
    |
    v
%1 future  : sample measurement[scalar], factor=scalar_transition
```

The `FlowInfo` column carries provenance, split keys, and random ancestors for
later analysis. None of those fields changes the document layout in this
example.

### Inference Plan IR

The second graph is the model-call schedule recorded by `generate`:

```text
call 0: generate future from current
attention: full_segment
positions: scientific
```

The plan is a mechanical, validated view of the staged Python program. It is
useful to compiler and runtime code, but users do not author it separately.

### Transformer Execution IR

The final section shows the exact records sent to the transformer:

| Role | Scientific identity | Model input | Training label |
| --- | --- | --- | --- |
| context | `scalar_forecast.current[scalar]` | `value:3` | none |
| target | `scalar_forecast.future[scalar]` | `<query>` | `value:5` |

At the target record the model sees `<query>` plus the embedding identifying
`future[scalar]`. It predicts logits there, and cross-entropy compares those
logits with `value:5`. The target value is only the label.

Every record in this document has rotary position `0`, so RoPE contributes no
serialization-order signal. The scientific identity embedding distinguishes
`current` from `future`. Full attention lets both records exchange information.
Printing `current` first is a packing choice, not part of the scientific model.

## 3. Extend the program to advection

The advection example predicts a field on four spatial cells for three future
times. It has four initial values, twelve forcing values, and twelve target
values.

```python
from experiments.probabilistic_dataflow.dsl import MeshAxis, OrderedAxis

cell = MeshAxis("cell", 4, coordinates=((0.0,), (0.25,), (0.5,), (0.75,)))
time = OrderedAxis("time", 3)

state = FieldType("state", (cell,), bins=16)
forcing_type = FieldType("forcing", (time, cell), bins=16)
trajectory = FieldType("state_trajectory", (time, cell), bins=16)

program = InferenceProgram(
    "synthetic_advection",
    budget=Budget(model_calls=1, generated_tokens=12),
)
initial = program.input_value("initial", state)
forcing = program.input_value("forcing", forcing_type)
future = program.generate(
    "future",
    trajectory,
    context=(initial, forcing),
    document=scientific_document,
    factor_name="advection_transition",
)
program.finish(future)
```

The DSL did not gain a spatial attention primitive. Named axes expand each
field into records with meaningful identities. For example:

```text
synthetic_advection.future[time=1,cell=(0.5,)]
```

The record counts changed mechanically:

| | Scalar | Advection |
| --- | ---: | ---: |
| Context records | 1 | 4 initial + 12 forcing |
| Target records | 1 | 12 future |
| Total records | 2 | 28 |

See [`debug_outputs/advection.md`](debug_outputs/advection.md). Its plan notes
that one twelve-token factor is approximated by twelve parallel token
marginals. All query records see the same context, but independently sampling
their logits cannot represent correlations among generated coordinates. The
compiler reports that approximation instead of silently calling it the original
joint distribution.

## 4. Drop down for control over model calls

Because the staged program is already an inference program, dropping down means
writing more calls and passing generated values between them.

### Preserve a scientific factorization

The structure example first predicts contacts from a sequence, then predicts
distances from the sequence and the generated contacts:

```python
contacts = program.generate(
    "contacts",
    contacts_type,
    context=(sequence,),
    document=scientific_document,
    factor_name="contact_map",
)
distances = program.generate(
    "distances",
    distance_type,
    context=(sequence, contacts),
    document=scientific_document,
    factor_name="distance_given_contacts",
)
program.finish(contacts, distances)
```

The second call consumes a value produced by the first, so the plan in
[`debug_outputs/structure.md`](debug_outputs/structure.md) contains `call 0 ->
call 1`. This is the factorization
`p(contacts | sequence) p(distances | sequence, contacts)`. It is not replaced
with one joint call.

### Add fixed-step refinement

`refine` adds another call for an already generated value:

```python
future = program.generate(
    "future",
    trajectory,
    context=(initial, forcing),
    document=scientific_document,
    factor_name="advection_transition",
)
program.refine(
    future,
    context=(initial, forcing),
    document=scientific_document,
    resample_fraction=0.25,
)
program.refine(
    future,
    context=(initial, forcing),
    document=scientific_document,
    resample_fraction=0.25,
)
program.finish(future)
```

The calls form `proposal -> refinement 1 -> refinement 2`. Generated values
flow between calls; there is still no arbitrary left-to-right order among the
twelve target coordinates within one call.

The debug execution uses the example's true future values to illustrate record
layout. A real inference runtime must substitute the proposal generated by the
preceding call. Sampling and refinement training are outside this spike.

For a genuinely sequential task, choose
`DocumentSpec(attention=CAUSAL, positions=SEQUENCE)`. That uses ordinary rotary
indices and a causal mask. The choice is per call, so it does not require a
different transformer architecture.

## 5. Train one transformer on text and science

The cross-domain demo trains one Grug model on causal synthetic text and
full-attention scientific records:

```python
from experiments.probabilistic_dataflow.training import train_cross_domain_smoke

result = train_cross_domain_smoke(steps=80, examples_per_task=8, seed=0)
```

| | Synthetic text | Synthetic advection |
| --- | --- | --- |
| Input unit | word-like token | scientific record |
| Position signal | rotary index `0..S-1` | scientific identity; rotary position `0` |
| Attention | causal | full within one example |
| Label | next token | value aligned with the `<query>` record |

Both tasks use the same token embeddings, transformer blocks, and output
projection. Scientific records additionally use a scientific-identity embedding
table, which contributes zero to text inputs. The tasks are evaluated as
separate dense batches because their masks differ, then their losses are
averaged before one optimizer update.

[`debug_outputs/cross-domain.md`](debug_outputs/cross-domain.md) places the two
document types side by side. It shows shifted text labels such as:

```text
input <text:the> -> label <text:ocean>
```

and aligned scientific labels such as:

```text
input <query> at future[time=0,cell=(0.0,)] -> label value:5
```

The tiny run is only a compatibility and memorization check. It does not test
held-out scientific prediction, language quality, cross-task transfer, or a
complete refinement runtime.
