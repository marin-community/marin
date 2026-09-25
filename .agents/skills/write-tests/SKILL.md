---
name: write-tests
description: Add, revise, or review Marin tests for an explicit behavior change, regression, or test-quality request.
---

# Write tests

Read root `TESTING.md` before writing or reviewing tests. For tests under
`lib/*`, also read the nearest module `AGENTS.md` and every testing guide it
references.

Treat those files as the source of truth for behavioral value, test style,
fakes and mocks, timing, numerical tolerances, markers, and commands.

Before adding a standalone scalar or configuration guard test, name the
reported regression or compatibility-critical public contract it protects. If
there is none, test consequential behavior on the valid path or omit the test.
Do not pin an exact dependency version, default, or serialized configuration
unless an external consumer depends on it.
