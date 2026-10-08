# TaskCompendium tests

Tests cover task serialization and answer formats, source conversions,
in-process and machine grading, grader controls, row admission, output capture,
source quality gates, review evidence and the pipeline's output views.
[fixtures/](fixtures/README.md) contains recorded inputs. `pipeline_stages.py`
provides the fixture source recipe, converters and grading machines;
`FixtureGradingMachines` runs grader controls on in-memory ShellSim machines that
stand in for a grader image.

From the repository root:

```bash
uv run --package taskcompendium --extra pipeline --group test pytest lib/taskcompendium/tests -q
```

Keep the repository's default marker expression. Tests requiring live providers,
containers or clusters are excluded unless explicitly selected for an integration
run. Use the existing Shellbox test factories and temporary storage at those I/O
boundaries. The [root testing policy](../../../TESTING.md) governs new tests;
the [package contract](../README.md) describes the behavior under test.
