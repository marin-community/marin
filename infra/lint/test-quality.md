# test-quality — detector prompt

## What to look for

Flag tests that verify implementation details instead of observable behavior, use mock/fake infrastructure when real implementations are available, make overly specific assertions that require frequent updates, or test pedantic edge cases that don't catch real regressions. Also flag duck-typed test doubles that gradually drift from the real thing.

## Anchor examples

- **Assertion on internals**: "Removed the imported constant assertion. The smoke test now checks the runnable bundle and the step config doc cap directly." — Tests verifying internal implementation state change instead of external behavior.

- **Heavy mocking**: "this is super mock heavy, consider removing" — Tests layered with mocks that obscure what is actually being exercised, obscuring whether the test catches real failures.

- **Fake HTTP server**: "i don't think emulating a server that produces the code we told it to have is that useful" — Tests using fake/canned implementations instead of testing against real services or minimal stubs.

- **Tautological tests**: "some of these are tautological agenty tests, i'd remove" — Tests that validate implementation rather than behavior; they fail when code changes but not when it breaks.

- **Duck-typed test doubles**: "we probably should try to not do these duck-typed test reimplementation that will eventually drift sufficiently enough to cause problems" — Hand-rolled test doubles that diverge from actual implementation over time.

- **Over-specific string matching**: "I might consider removing some of the text matching tests or making them less specific, otherwise we'll need to update them with every tweak of the status" — Tests brittle to implementation changes that aren't behavioral changes.

- **Pedantic edge case**: "up to you but i tend to cull some of the more pedantic agent tests or combine them" — Tests covering edge cases unlikely to catch real failures.

## False-positive guidance

- Integration tests that validate externally-observable behavior (output files, API responses, state persistence) against real or minimally-stubbed dependencies are good.
- Fixtures that capture real setup/teardown and are reused across multiple test functions are necessary.
- Mocks at I/O boundaries (network, filesystem, external services) are acceptable when testing error handling or behavior with unavailable services.
- Parametrized tests using `@pytest.mark.parametrize` to test multiple inputs are encouraged, not a problem.
- Tests marked `@pytest.mark.slow` that validate expensive operations against real dependencies (TPU checks, large model loads) are expected.

## Suggested confidence floor

Flag with medium confidence when tests use mocks/fakes for domain logic (computation, serialization, state), verify string/debug output brittlely, or assert on internal API calls; reserve high confidence for tautological tests, duck-typed doubles, or when reviewers explicitly call the test quality into question.
