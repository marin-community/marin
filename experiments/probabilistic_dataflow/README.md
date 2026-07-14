# Probabilistic scientific dataflow spike

This experiment tests a scientist-facing Python DSL for building scientific
transformer calls without introducing a new model stack or dataset system. A
single staged `InferenceProgram` describes:

- scientifically typed values and their named axes;
- which values are supplied to each model call;
- which values each call generates;
- dependencies between calls;
- the attention and position policy for each generated document.

There is no separate `Query` object or generic strategy lowerer. Calling
`program.generate(...)` creates both a scientific random value and the model
call that will produce it. Passing that returned value as context to a later
`generate(...)` creates an explicit dependency between the calls.

Start with [`TUTORIAL.md`](TUTORIAL.md) for a guided path from a two-record
scalar prediction through indexed advection, refinement, factorized structure,
and shared text-and-science training.

## Scientist-facing surface

```python
document = DocumentSpec(
    attention=AttentionPattern.FULL,
    positions=PositionMode.SCIENTIFIC,
)
program = InferenceProgram(
    "advection",
    budget=Budget(model_calls=1, generated_tokens=12),
)

initial = program.input_value("initial", state)
forcing = program.input_value("forcing", forcing_type)
future = program.generate(
    "future",
    trajectory,
    context=(initial, forcing),
    document=document,
    factor_name="advection_transition",
)
program.finish(future)
```

The logical values contain no physical token order. Compilation creates one
record per scientific value instance:

```text
record embedding = content token embedding + scientific position embedding
```

Context records carry value tokens. Target records carry a `<query>` token and
an aligned training label; the target value is not a model input. Scientific
documents use zero rotary positions and full attention when those choices match
the scientific factor. Reordering complete records therefore only reorders the
outputs.

Sequential factorization is represented by generated values flowing between
model calls. For example:

```python
contacts = program.generate("contacts", contacts_type, context=(sequence,), ...)
distances = program.generate("distances", distance_type, context=(sequence, contacts), ...)
```

The second call depends on the first because `contacts` is generated context.
A causal mask is used only when the task itself requires sequence order.

## Shared text-and-science model

The model remains a normal dense Grug transformer with RoPE. Execution data
selects the behavior for each call:

| Task | Position signal | Attention | Target alignment |
| --- | --- | --- | --- |
| Synthetic text | rotary token index `0..S-1` | causal | next token |
| Scientific records | scientific descriptor; rotary position `0` | full within segment | same record |

Both paths use the same token embeddings, transformer blocks, and output
projection. Calls with different attention layouts are evaluated in separate
dense batches, but their losses update the same model parameters.

The scientific smoke test packs synthetic advection and contact-map examples.
A second smoke test mixes causal synthetic text with full-attention scientific
records. These are compatibility and memorization checks, not scientific
generalization or language-quality results.

## Run

Inspect compiler behavior without training:

```bash
uv run python -m experiments.probabilistic_dataflow.demo --training-steps 0
```

Run the tiny training demonstrations:

```bash
uv run python -m experiments.probabilistic_dataflow.demo --training-steps 80
```

## Debug renderings

Generate readable Markdown for the staged values, model-call plan, transformer
execution, and document layout:

```bash
uv run python -m experiments.probabilistic_dataflow.debug_render
```

The reports live in [`debug_outputs/`](debug_outputs/README.md):

- [`scalar.md`](debug_outputs/scalar.md) shows one context scalar and one target;
- [`advection.md`](debug_outputs/advection.md) shows an indexed field and refinement calls;
- [`contacts.md`](debug_outputs/contacts.md) shows unordered residue-pair targets;
- [`structure.md`](debug_outputs/structure.md) shows `sequence -> contacts -> distances`;
- [`mixed-packing.md`](debug_outputs/mixed-packing.md) shows packed segment boundaries;
- [`cross-domain.md`](debug_outputs/cross-domain.md) compares text and science calls through one model.

Verify that checked-in reports match the renderer:

```bash
uv run python -m experiments.probabilistic_dataflow.debug_render --check
```

## Implemented

- ordered, set, mesh, categorical, and unordered-pair axes;
- typed discrete fields and canonical pair coordinates;
- external inputs, deterministic map/join/select/reduce nodes, and generated values;
- provenance, split-key, and random-ancestor propagation;
- a staged model-call DAG with full or causal attention and scientific or sequence positions;
- parallel generation and fixed-step refinement;
- factor-dependency preservation and explicit parallel-marginal approximation notes;
- inference-plan and transformer-execution IRs;
- shared value-token embeddings plus scientific position embeddings;
- aligned scientific labels, shifted text labels, and standard cross-entropy for both;
- heterogeneous scientific packing with per-document segment boundaries;
- a numerical scientific-record permutation-equivariance check;
- field RMSE and spectral-error metrics;
- tiny scientific-only and cross-domain Marin Grug training loops.

## Deliberate limits

- Values are already discretized synthetic integers, and text uses a tiny fixed vocabulary.
- An LM generating these Python inference programs is the intended workflow, but is not implemented here.
- Parallel field generation is a product-of-token-marginals approximation, recorded in the plan.
- Refinement call graphs compile, but the smoke trainer only trains proposal calls.
- Scientific positions use one learned embedding per fully qualified coordinate; compositional axis and topology encoders are not implemented.
- Calls with different attention layouts are not packed into the same dense batch.
- Inference sampling, KV-cache execution, adaptive stopping, datasets, simulators, and external effects are out of scope.
