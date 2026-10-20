# Typed SkyRL recipes

The schema API validates immutable authored recipes. Hydra composes the running configuration, including
group defaults, YAML references such as `${trainer.ckpt_interval}` and the launcher's model, data and output paths.

Schema classes such as `Trainer`, `Generator` and `Data` are generated from the base YAML, config groups
and the type sidecar. A Recipe combines sparse authored contributions. Defaults remain in YAML; an
unset field contributes nothing to `to_skyrl()`.

```python
from marinskyrl.recipe_schema import ContextBudget, RecipePatch, SkyRLRecipe, Trainer

base = SkyRLRecipe(
    context_budget=ContextBudget(
        request_window_tokens=32768,
        max_new_tokens_per_turn=4096,
        max_turns=2,
    ),
)
policy = RecipePatch(trainer=Trainer(train_batch_size=64, policy_mini_batch_size=32))
computed = RecipePatch(trainer=Trainer(ckpt_interval=4))
recipe = SkyRLRecipe.combine(base=base, policy=policy, computed=computed)
recipe = recipe.with_settings(["context_budget.max_turns=4"])
document = recipe.to_skyrl()
```

`combine` permits parts to override the base. Different values at the same leaf, or a parent and its
descendant, conflict between parts. Errors name both parts. Lists are whole leaves and empty mappings
contribute nothing. Part names determine a stable merge order, so permuting keyword arguments preserves
the result and mapping order. Use `merge` for an intended override.

`with_settings` parses dotted paths using the field type, merges settings onto the document, then
validates the complete result. JSON arrays and objects are accepted for structured values; `null` is an
explicit null. Fields whose YAML defaults reference another field reject null unless their YAML default
is null. Open mapping keys keep their insertion order during merge and settings.

Group selections use option names such as `config_groups.algorithm_recipe=grpo`. An unset selection
contributes nothing; an explicit null is invalid. Structured settings beginning with `{`, `[` or `"`
require valid JSON. Quote a string beginning with one of those characters as a JSON string.

Use `SkyRLRecipe.from_document` for parsed JSON or YAML mappings. `RecipePatch.from_document` accepts a
part without a context budget. Python constructors are strict; JSON arrays become immutable tuples.
Public classes are available from `marinskyrl.recipe_schema`. Tensor-parallel and artifact-identity
checks run on `SkyRLRecipe` after its parts have been combined.

Generated author classes omit launch-owned and context-derived keys. Set `context_budget` for token
limits; supply launch-owned values through the launch document. Owner checks also reach keys inside
open mappings. `terminal_bench` holds Harbor configuration as an open mapping. Inference-engine and
other third-party option maps retain their documented pass-through surfaces; SkyRL's reserved engine
keys are checked at the recipe root.

The launcher validates expanded source recipes through `SkyRLRecipe.from_document` before Hydra
composition. It fills omitted allocation node counts, runtime profiles and data kinds
from the composed configuration and checks supplied values. Authored data lists survive when envelope
inputs are empty; differing nonempty recipe and envelope lists are rejected. Role GPU counts must fit
the available hardware. Schema-only fields keep their consumer defaults when absent from the recipe.

Artifact draft sources require `source_identity` in the form `sha256:` followed by 64 lowercase hex
digits. Agreement with the artifact manifest is a launch-time check. Hub revisions and local source
identities use their source-specific launcher rules.

After editing YAML or resolving a generated-file conflict, regenerate from the repository root:

```bash
uv run python scripts/generate_recipe_schema.py
uv run python scripts/generate_recipe_schema.py --check
```

Commit the generated sections and public exports with the source edit. Put finite choices, null-default
structures and other constraints that YAML cannot express in `scripts/recipe_schema_sidecar.py`.
`TYPES` and `OPEN` paths must exist in YAML. `NAMES` paths may also name `UNDECLARED` fields or their
parent mappings. `UNDECLARED` entries represent code-default fields and cannot shadow YAML keys.
Ellipsis keeps those fields unset in an authored document.

Add a migration row to `REMOVED` in `ownership.py` when an authored key changes. Document validation
reports that row before an unknown-key error. `terminal_bench.harbor.max_episodes` reports
"use context_budget.max_turns; Harbor reads max_turns".

Marin imports `marinskyrl.recipe_schema` from the standalone `packages/marinskyrl-recipe` wheel.
Its only runtime dependency is `pydantic>=2.12.2,<3`. The Marin workspace uses a Git subdirectory
requirement at the same commit as its isolated launcher environment. Build this project with
`uv build --wheel packages/marinskyrl-recipe`; the wheel uses the shared schema source directly.
