# test-quality — detector prompt

## AGENTS.md anchor

§ Testing — "Prefer integration-style tests that validate
externally-observable behavior. Do not write tautological tests... No mocks
unless testing I/O boundaries (network, filesystem). No `time.sleep()` in
tests — inject `now=time.time()` or mock time instead. Prefer top-level
`def test_*` with fixtures over test classes."

## What to look for

Flag tests that verify implementation details instead of observable behavior, use mock/fake infrastructure when real implementations are available, make overly specific assertions that require frequent updates, or test pedantic edge cases that don't catch real regressions. Also flag duck-typed test doubles that gradually drift from the real thing, `time.sleep()` calls in test bodies, and unittest-style `class TestX(...)` wrappers around what could be top-level `def test_*` functions.

## Anchor examples

- **Assertion on internals**: "Removed the imported constant assertion. The smoke test now checks the runnable bundle and the step config doc cap directly." — Tests verifying internal implementation state change instead of external behavior.

- **Heavy mocking**: "this is super mock heavy, consider removing" — Tests layered with mocks that obscure what is actually being exercised, obscuring whether the test catches real failures.

- **Fake HTTP server**: "i don't think emulating a server that produces the code we told it to have is that useful" — Tests using fake/canned implementations instead of testing against real services or minimal stubs.

- **Tautological tests**: "some of these are tautological agenty tests, i'd remove" — Tests that validate implementation rather than behavior; they fail when code changes but not when it breaks.

- **Duck-typed test doubles**: "we probably should try to not do these duck-typed test reimplementation that will eventually drift sufficiently enough to cause problems" — Hand-rolled test doubles that diverge from actual implementation over time.

- **Over-specific string matching**: "I might consider removing some of the text matching tests or making them less specific, otherwise we'll need to update them with every tweak of the status" — Tests brittle to implementation changes that aren't behavioral changes.

- **Pedantic edge case**: "up to you but i tend to cull some of the more pedantic agent tests or combine them" — Tests covering edge cases unlikely to catch real failures.

- **`time.sleep()` in test body**: Any literal `time.sleep(...)` inside a test function is a smell — the test is racing the system under test instead of controlling time. AGENTS.md prescribes injecting `now=time.time()` or mocking time. The exception is genuinely time-bound integration tests (e.g. waiting on a TPU bring-up), which should be marked `@pytest.mark.slow` and commented.

- **Test class wrapping standalone functions**: `class TestFoo(unittest.TestCase): def test_x(self): ...` where the class adds nothing beyond grouping. AGENTS.md prefers top-level `def test_*` with pytest fixtures.

## False-positive guidance

- Integration tests that validate externally-observable behavior (output files, API responses, state persistence) against real or minimally-stubbed dependencies are good.
- Fixtures that capture real setup/teardown and are reused across multiple test functions are necessary.
- Mocks at I/O boundaries (network, filesystem, external services) are acceptable when testing error handling or behavior with unavailable services.
- Parametrized tests using `@pytest.mark.parametrize` to test multiple inputs are encouraged, not a problem.
- Tests marked `@pytest.mark.slow` that validate expensive operations against real dependencies (TPU checks, large model loads) are expected.

## Suggested confidence floor

Flag with medium confidence when tests use mocks/fakes for domain logic (computation, serialization, state), verify string/debug output brittlely, or assert on internal API calls; reserve high confidence for tautological tests, duck-typed doubles, or when reviewers explicitly call the test quality into question.
