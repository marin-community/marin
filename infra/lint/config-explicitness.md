# config-explicitness — detector prompt

## What to look for

Flag code that silently falls back to defaults or magic values instead of making configuration decisions explicit. This includes silent defaults that hide behavior, `default_*` wrappers that obscure mechanisms, environment variables substituting for explicit parameters, magic strings/numbers without top-level constants, and global state instead of explicit dependency injection.

## Anchor examples

- **Silent defaults with surprising behavior** (comment 5 from dlwh): "The local pinned JAX 0.8.0 does not expose `jax.core.Literal`, so this now imports `Literal` from `jax._src.core` directly and uses `isinstance(value, JaxprLiteral)` without fallback logic." — Fallback logic silently swallows errors; explicit is better.

- **Default wrapper pattern** (comment 14 from dlwh): "we're moving away from using 'default_' as a pattern. rename this function" — Functions prefixed `default_*` hide what underlying mechanism they're deferring to.

- **Env var instead of explicit parameter** (comment 46 from rjpower): "Hoisted the magic strings to top-level constants (`IRIS_PYPI_MIRROR_ENV_VAR`, `IRIS_PYPI_MIRROR_OPT_OUT`) per § Naming... (2) the same env var is the per-pool override surface via `_task_env`" — Env vars couple config to runtime environment instead of passing parameters through the call stack.

- **Global state instead of instance config** (comment 54 from rjpower): "No globals pls? These should probably be part of a class instance?" — Module-level variables with mutable state scatter configuration across the codebase.

- **Magic numbers without constants** (comment 27 from dlwh): "weird this gets a constant but the other datasets don't" — Inconsistent use of constants for similar magic values makes behavior less discoverable.

- **Default that masks errors** (comment 30 from rjpower): "Intentional — silently trimming hides correctness bugs (caller wrote a query they thought returned everything, got a truncated answer, made a wrong decision). The exception forces an explicit opt-in via `max_rows` or a `LIMIT` clause." — Silent behavior changes surprise callers; make it explicit.

- **Missing explicit configuration at boundaries** (comment 57 from yonromai): "The deploy config exposes `port`, and both GCP bootstrap health checks and k8s manifests use `cfg.port`, but the image itself always starts `finelog.server.main --port 10001`. Any config that sets a different port will advertise/probe that port while the process listens on 10001" — Configuration not threaded through to where it's actually used.

## False-positive guidance

- **Env vars for operational toggles**: Using env vars for kill switches or emergency circuit breakers (e.g. `IRIS_DEBUG_UV_SYNC` for rollback) is acceptable when it's a one-bit override that needs fast path-to-production.

- **Sensible defaults at creation time**: A function parameter with a clear default value that is explicitly documented in the docstring is fine (e.g., `timeout=30`). The issue is when defaults are *implicit* or *scattered*.

- **Protocol defaults in wire formats**: Protobuf and wire-format default values (e.g., `UNSPECIFIED` meaning `PREFIX` on the wire) are acceptable when the boundary is explicit. The problem is when Python wrappers layer their own contradictory defaults on top.

- **Legitimate fallback chains**: A fallback is acceptable when each level in the chain is explicit and tested. For example, "try local path, then fall back to HF, but make the fallback testable."

- **Test-only defaults**: Memory modes or debug configurations used only in tests can have implicit defaults if they're scoped to test-only code paths and the production path requires explicit configuration.

## Suggested confidence floor

Start at high confidence for violations of the "explicit top-level constants" rule (magic numbers without `CONSTANT` naming) and "env vars where an explicit parameter exists" pattern; reduce confidence for borderline cases like optional kwargs with sensible defaults that are actually documented.
