# config-explicitness — detector prompt

## AGENTS.md anchor

§ Configuration — no `default_*` wrappers; force explicit specification of
critical parameters; centralize defaults; prefer constructor/config
parameters over env vars; composition over inheritance. § Naming — top-level
constants for magic strings/numbers. Companion detector: `defensive` covers
silent fallbacks and swallowed exceptions.

## What to look for

Flag configuration that hides decisions: silent defaults, `default_*` wrapper
functions, env vars used in place of explicit parameters, magic
strings/numbers without top-level constants, and global mutable state where a
class instance would do. Silent *fallbacks* and swallowed exceptions belong
in `defensive`; this detector is about *where the configuration is supplied
from*.

## Anchor examples

- **`default_*` wrapper pattern** (comment 14 from dlwh): "we're moving away
  from using 'default_' as a pattern. rename this function" — Functions
  prefixed `default_*` hide what underlying mechanism they're deferring to.

- **Env var instead of explicit parameter** (comment 46 from rjpower):
  "Hoisted the magic strings to top-level constants
  (`IRIS_PYPI_MIRROR_ENV_VAR`, `IRIS_PYPI_MIRROR_OPT_OUT`)... the same env var
  is the per-pool override surface via `_task_env`" — Env vars couple config
  to runtime environment instead of passing parameters through the call
  stack.

- **Global state instead of instance config** (comment 54 from rjpower): "No
  globals pls? These should probably be part of a class instance?" —
  Module-level variables with mutable state scatter configuration across the
  codebase.

- **Magic numbers without constants** (comment 27 from dlwh): "weird this
  gets a constant but the other datasets don't" — Inconsistent use of
  constants for similar magic values makes behavior less discoverable. The
  fix is a top-level `CONSTANT_NAME = ...` at the module head.

- **Missing explicit configuration at boundaries** (comment 57 from
  yonromai): "The deploy config exposes `port`, and both GCP bootstrap health
  checks and k8s manifests use `cfg.port`, but the image itself always starts
  `finelog.server.main --port 10001`. Any config that sets a different port
  will advertise/probe that port while the process listens on 10001" —
  Configuration not threaded through to where it's actually used.

- **Config divergence across deployments**: A flag set via env var in one
  pool and via config object in another. Pick one surface and route the
  other through it.

## False-positive guidance

- **Env vars for operational toggles**: Using env vars for kill switches or
  emergency circuit breakers (e.g. `IRIS_DEBUG_UV_SYNC` for rollback) is
  acceptable when it's a one-bit override that needs fast path-to-production.

- **Sensible defaults at creation time**: A function parameter with a clear
  default value that is explicitly documented in the docstring is fine
  (e.g., `timeout=30`). The issue is when defaults are *implicit* or
  *scattered*.

- **Protocol defaults in wire formats**: Protobuf and wire-format default
  values (e.g., `UNSPECIFIED` meaning `PREFIX` on the wire) are acceptable
  when the boundary is explicit. The problem is when Python wrappers layer
  their own contradictory defaults on top.

- **Test-only defaults**: Memory modes or debug configurations used only in
  tests can have implicit defaults if they're scoped to test-only code paths
  and the production path requires explicit configuration.

## Suggested confidence floor

Start at high confidence for `default_*` wrapper names, env vars where an
explicit parameter exists in the same module, and magic strings/numbers
repeated more than once without a top-level constant. Lower confidence on
optional kwargs with documented defaults — those are explicit by definition.
