# Coding Standards

Detailed coding standards for agents and contributors. Referenced from `AGENTS.md`.

## Testing

### Principles

- Prefer integration-style tests that validate externally-observable behavior.
- Always fix tests you broke. Do not relax tolerances or hack around failures.
- Use pytest fixtures and parameterization to avoid duplication.
- **Before writing any test**: search for existing test files in `tests/` covering the module you changed. Extend an existing test file before creating a new one.
- **No mocks** unless testing I/O boundaries (network, filesystem). Test against real behavior.

### Anti-patterns (never do these)

```python
# BAD: tautological — tests the constructor, not behavior
def test_config_has_name():
    cfg = MyConfig(name="foo")
    assert cfg.name == "foo"

# BAD: mocking internals — proves nothing about real behavior
def test_processor():
    with patch("mymodule.helper") as m:
        m.return_value = 42
        assert process() == 42

# BAD: type-checking as test — this is what pyrefly is for
def test_config_is_dataclass():
    assert dataclasses.is_dataclass(MyConfig)

# GOOD: validates externally observable behavior
def test_processor_skips_empty_input():
    result = process(input=[])
    assert result.count == 0 and result.errors == []

# GOOD: regression test for a specific bug
def test_dedup_handles_unicode_normalization():
    docs = [Document(text="café"), Document(text="cafe\u0301")]
    result = exact_dedup(docs)
    assert len(result) == 1
```

## Code Reuse

Before writing any utility function, helper, or data structure:

1. Search the codebase: `grep -r "def <function_concept>" lib/`
2. Check subproject utils: `lib/marin/src/marin/`, `lib/iris/src/iris/`, `lib/levanter/`
3. Check `pyproject.toml` for available third-party packages before adding new ones

If a suitable implementation exists, use it. Do not create parallel implementations.

## Exception Handling

- Let exceptions propagate by default.
- Only catch to add meaningful context and re-raise, or to intentionally alter control flow.
- NEVER swallow exceptions unless specifically requested.

```python
# BAD: swallowing exceptions
try:
    result = compute()
except Exception:
    result = None  # silently hides bugs

# BAD: catching too broadly
try:
    config = load_config(path)
except Exception as e:
    logger.warning(f"Config error: {e}")
    config = default_config()  # masks real problems

# GOOD: adding context and re-raising
try:
    config = load_config(path)
except FileNotFoundError:
    raise FileNotFoundError(f"Config file missing at {path}; did you run setup?") from None

# GOOD: intentional control flow change at a system boundary
try:
    response = client.get(url, timeout=5)
except httpx.TimeoutException:
    return CachedResult(stale=True)  # explicit fallback decision
```

## Comments

Write detailed comments for module/class-level behavior or subtle logic. Do not restate the code:

```python
# BAD: restates the code
# Use in-memory rollout queue
rollout_queue = InMemoryRolloutQueue()

# GOOD: explains non-obvious reasoning
# Each FlightServer instance provides ~1GB/s throughput. With 200Gbps NICs,
# 16 parallel servers should saturate the network.
```
