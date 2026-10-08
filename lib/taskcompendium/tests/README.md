# TaskCompendium tests

Tests cover task serialization and answer formats, source conversions,
in-process and machine grading, output capture, source quality gates, review
evidence and pipeline sidecars. [fixtures/](fixtures/README.md) contains recorded inputs;
`pipeline_stages.py` provides pipeline test support.

From the repository root:

```bash
uv run --package taskcompendium --extra pipeline --group test pytest lib/taskcompendium/tests -q
```

Keep the repository's default marker expression. Tests requiring live providers,
containers or clusters are excluded unless explicitly selected for an integration
run. Use the existing Shellbox test factories and temporary storage at those I/O
boundaries. The [root testing policy](../../../TESTING.md) governs new tests;
the [package contract](../README.md) describes the behavior under test.
