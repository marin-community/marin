# No-Slop Examples

Use these examples to calibrate deletion. They are patterns, not templates to
copy into review comments.

## Tests

### Configuration projection

Delete:

```python
def test_resolve_cluster_name_prefers_cli_name():
    config = IrisClusterConfig(name="from-config")
    assert resolve_cluster_name(config, None, "from-cli") == "from-cli"
```

The test feeds a value through a trivial precedence branch and reads it back.
Keep coverage only if precedence caused a real regression or crosses a public
configuration boundary with nontrivial merge behavior.

### Type checker duplication

Delete:

```python
base = ResourceConfig.with_gpu("H100", count=8)
assert isinstance(base.device, GpuConfig)
```

The constructor and return type already establish the type.

### Incidental prose

Delete:

```python
assert "retrying" in caplog.text
```

Test the retry effect, attempt count at the external boundary, or final state.
Keep exact text only when another program parses it or the API promises it.

### Self-generated golden

Delete a golden generated from the same built-in schema or renderer being
tested. Capture independent production evidence or assert the external behavior
that the golden represents.

## Prose

### Stock contrast

Delete or split:

> The issue is not scheduling, but reconciliation.

Prefer:

> Reconciliation drops assigned tasks after a controller restart. Scheduling
> assigns them correctly.

### Empty framing

Delete:

> The mechanism is worth stating because it differs from the original ideas.

Start with the mechanism.

### Visible structure

Delete:

> The configuration has three parts, none of which affect numerics.

Start the list. Include the numerics claim only with evidence.

### Unsupported benefit

Delete:

> Running all arms on one allocation removes placement variance.

If unmeasured, say nothing or write `The sequential harness has not been
characterized for placement variance.`

### Contradictory result

A PR body that claims `12x faster` while its table shows no 12x comparison is
wrong. Correct the prose or the table before polishing either one.

## Deletion Signals

Reassess the whole design when review removes a consumer, transport, command, or
feature. Delete its helpers, tests, docs, flags, and compatibility paths in the
same pass.
