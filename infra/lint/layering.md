# layering — detector prompt

## AGENTS.md anchor

§ Code Reuse — "Dependency direction: {`iris`, `haliax`} → {`levanter`, `zephyr`} → `marin`. Each layer may only import from layers to its left. Never introduce reverse dependencies."

## What to look for

Flag any import that crosses the layering rule in the wrong direction. The
arrow points from "imports from" to "is imported by," so `iris` and `haliax`
sit at the leaf; `levanter` and `zephyr` may import from them; `marin` may
import from any of the others. A `from marin...` line inside `lib/levanter/`,
`lib/iris/`, `lib/zephyr/`, or `lib/haliax/` is a reverse dependency. A
`from levanter...` or `from zephyr...` line inside `lib/iris/` or
`lib/haliax/` is also a reverse dependency.

Also flag sibling imports inside the middle tier: `lib/levanter/` importing
from `lib/zephyr/` (or vice versa) is *not* the layering rule but is usually
a sign that shared code belongs in the leaf tier (`iris`/`haliax`) instead.

## Anchor examples

- **Reverse leaf-from-trunk import**:
  ```python
  # in lib/iris/src/iris/cluster/something.py
  from marin.processing import tokenize  # ❌ iris cannot depend on marin
  ```
  Why: leafs are reusable libraries; pulling `marin` into `iris` makes the
  leaf un-reusable and creates an import cycle the moment `marin` reaches
  back for `iris`.

- **Levanter importing from marin**:
  ```python
  # in lib/levanter/src/levanter/trainer.py
  from marin.experiments.something import default_config
  ```
  Why: `levanter` is a library; `marin` orchestrates experiments on top of
  it. The dependency must point the other way.

- **Zephyr importing from marin**:
  ```python
  # in lib/zephyr/src/zephyr/pipeline.py
  from marin.datakit.download import ...
  ```
  Why: same as above — `marin` composes `zephyr`, not the reverse.

- **Cross-sibling at the middle tier**:
  ```python
  # in lib/levanter/src/levanter/data/foo.py
  from zephyr.coordinator import Coordinator
  ```
  Why: not strictly a layering violation, but in practice this means a
  helper that both libraries need has been homed in the wrong place.
  Move it to a leaf (`iris`/`haliax`) or to `marin` and depend on it from
  one direction.

- **TYPE_CHECKING guard hiding a reverse import**:
  ```python
  if TYPE_CHECKING:
      from marin.experiments import ExperimentConfig
  ```
  Why: AGENTS.md § Code Style forbids `TYPE_CHECKING` guards generally;
  when the guarded import is *also* a layering violation, the type hint is
  doing double duty as a cycle-hider. Restructure with a Protocol in the
  importing layer.

## False-positive guidance

- **Tests**: a test inside `lib/iris/tests/` importing `marin` to exercise
  an integration path is acceptable when explicitly marked
  `@pytest.mark.integration` and the production code stays clean.
- **Tooling under `infra/`, `scripts/`, `experiments/`**: top-level scripts
  may import from any layer; the rule is about library code under `lib/`.
- **String references that aren't imports**: a docstring or comment
  mentioning `marin` inside `lib/iris/` is fine. Flag actual `import` /
  `from` statements only.
- **Vendored or generated code**: `_pb2.py`, protobuf-generated modules, and
  third-party vendored trees are exempt.

## Suggested confidence floor

High confidence on any `from marin...` or `import marin` inside `lib/iris/`,
`lib/haliax/`, `lib/levanter/`, or `lib/zephyr/`. High confidence on any
`from levanter...` / `from zephyr...` inside `lib/iris/` or `lib/haliax/`.
Lower confidence on middle-tier sibling imports — surface them as a `nit`
asking whether the shared helper belongs lower in the stack.
