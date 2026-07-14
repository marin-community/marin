# Probabilistic scientific dataflow spike

This experiment tests a scientist-facing Python abstraction for probabilistic programs without building dataset ingestion or a new model stack. Two unrelated synthetic problems compile through the same four IR levels and train through Marin's existing dense Grug transformer:

- a four-cell advective field with ordered time and mesh semantics;
- a permutation-aware contact map over unordered residue pairs.

The default scientific smoke configuration packs 144 aligned targets from eight examples of each problem into eight length-64 rows. On CPU with seed 0, 80 optimization steps reduced loss from 4.2265 to 0.0340 and raised training accuracy from 0% to 100%. A random permutation of all 28 records in an advection call changed restored logits by at most `5.96e-08`. These are compatibility, memorization, and numerical equivariance checks. They do not measure held-out scientific generalization.

A second smoke run trains one parameter set on both causal synthetic text and full-attention scientific records. Text uses ordinary rotary positions and shifted next-token labels; science uses zero rotary positions, scientific descriptors, and aligned labels. With equal task weighting, 80 optimization steps reduced combined loss from 4.1909 to 0.2022. Text accuracy rose from 0% to 75%, and scientific accuracy rose from 0% to 100%. This demonstrates shared-model compatibility, not language quality or cross-task transfer.

Start with [`TUTORIAL.md`](TUTORIAL.md) for a guided path from a two-record scalar query through indexed advection, explicit refinement, and shared text-and-science training.

The spike also includes a two-factor structure program to distinguish

```text
sequence -> contacts -> distances
```

from a single joint prediction. A forced parallel plan for the dependent factors is rejected, while the default and explicit autoregressive planners preserve the dependency. A separate training-only normalization example is rejected when requested as a deployment input.

## Scientist-facing surface

```python
cell = MeshAxis("cell", 4, coordinates=((0.0,), (0.25,), (0.5,), (0.75,)))
state = FieldType("state", (cell,), bins=16)

with Program("advection") as model:
    initial = model.variable("initial", state, source=initial_source)
    forcing = model.variable("forcing", state, source=forcing_source)
    future = model.sample("future", state, learned_joint(initial, forcing))

query = Query(
    program=model,
    given=(initial, forcing),
    targets=(future,),
    environment="deployment",
)
plan = compile_query(query, Refine(ParallelQuery((future,)), steps=3, resample_fraction=0.25))
```

The logical program contains no token order. Lowering creates one record per scientific value instance:

```text
record embedding = content token embedding + scientific position embedding
```

Observed records carry value tokens. Target records carry a query token and an aligned label that is not part of the model input. Scientific calls pass zero runtime rotary positions, making RoPE the identity, and parallel factors use full attention within each packed segment. Reordering complete records therefore reorders their outputs without changing them. Autoregressive factorization is represented by generated values flowing between model calls, not by a causal mask over an arbitrary serialization.

The model itself remains a normal Grug transformer with RoPE. Execution data selects the appropriate behavior per call:

| Task | Input position | Attention | Target alignment |
|---|---|---|---|
| Synthetic text | physical token index `0..S-1` | causal | next token |
| Scientific records | zero rotary position plus scientific descriptor | full within segment | same record |

Both paths use the same token embedding, transformer blocks, and output projection. Calls with incompatible attention layouts are evaluated as separate dense batches in the demo, but their gradients update the same model and optimizer state.

## Run

Compile the demonstrations, show diagnostics and packing metadata, and train the tiny shared model:

```bash
uv run python -m experiments.probabilistic_dataflow.demo --training-steps 80
```

Skip training when inspecting compiler output:

```bash
uv run python -m experiments.probabilistic_dataflow.demo --training-steps 0
```

## Debug renderings

Generate readable Markdown for every IR and the compiler-generated document layout:

```bash
uv run python -m experiments.probabilistic_dataflow.debug_render
```

The generated reports live in [`debug_outputs/`](debug_outputs/README.md):

- [`scalar.md`](debug_outputs/scalar.md) shows the four IRs for one observed scalar and one target scalar;
- [`advection.md`](debug_outputs/advection.md) includes parallel and three-step refinement plans;
- [`contacts.md`](debug_outputs/contacts.md) shows unordered-pair targets;
- [`structure.md`](debug_outputs/structure.md) shows the two-factor autoregressive dependency;
- [`mixed-packing.md`](debug_outputs/mixed-packing.md) shows segment boundaries in dense rows;
- [`cross-domain.md`](debug_outputs/cross-domain.md) compares causal text and full-attention scientific calls through one parameter set.

Each domain report contains Mermaid graphs for graph-shaped IRs and tables for exact fields. Execution documents show the content token, scientific position embedding, rotary position, aligned target, attention treatment, and loss weight for every record. Physical positions are packing-only for scientific calls, while text calls deliberately use them as rotary positions.

Verify that checked-in outputs match the renderer:

```bash
uv run python -m experiments.probabilistic_dataflow.debug_render --check
```

## Implemented

- ordered, set, mesh, categorical, and unordered-pair axes;
- typed discrete fields and canonical pair coordinates;
- variables, deterministic map/join/select/reduce nodes, learned factors, and conditional queries;
- provenance, environment, split-key, and random-ancestor propagation;
- environment-allowlist and split-isolation checks;
- explicit train/deployment query-role differences;
- default parallel scheduling, explicit autoregression, and fixed-step refinement;
- factor-dependency preservation and explicit parallel-marginal approximation notes;
- conditional-query, inference-plan, and transformer-execution IRs over the probabilistic dataflow graph;
- shared value-token embeddings plus separate scientific position embeddings;
- runtime rotary position IDs threaded through Grug attention;
- full segmented attention and zero rotary positions for unordered scientific factor calls;
- causal attention and ordinary rotary positions for synthetic text;
- aligned scientific targets, shifted text targets, and standard cross-entropy for both;
- a numerical record-permutation equivariance check;
- field RMSE and spectral-error terminal metrics;
- tiny scientific-only and cross-domain Marin Grug smoke training loops.

## Deliberate limits

- Values are already discretized synthetic integers, and text uses a tiny fixed vocabulary; dataset schemas, readers, and learned tokenizers are out of scope.
- `learned_joint` records factorization but does not yet provide a structured distribution object.
- Parallel field lowering is an explicit product-of-token-marginals approximation. The diagnostic is retained in the plan rather than silently treating it as the original joint factor.
- Refinement call graphs are represented, but the smoke trainer uses only parallel proposal calls; training refinement requires proposal sampling rather than teacher-forced truth in feedback context.
- Scientific positions currently use one learned embedding per fully qualified field coordinate. Compositional axis, topology, coordinate, and relation encoders are not implemented.
- Time-indexed availability rules are not implemented. A future version should express availability symbolically over named indices and bind those indices when selecting an example.
- No current scientific factor requires a within-call causal mask. Factor dependencies use separate calls; an explicitly sequential factor would need a semantic attention-edge representation.
- Cross-domain calls share parameters but are not packed into the same physical row because their attention layouts differ.
- Inference sampling, KV-cache execution, adaptive stopping, simulators, and external effects are not implemented.
