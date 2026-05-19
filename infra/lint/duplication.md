# duplication — detector prompt

## What to look for

Flag functions, blocks, or logic patterns that appear multiple times in the same file or across closely related call sites and could be consolidated into a helper. This includes copy-paste code blocks, identical error handling boilerplate, repeated test case structures (requesting parametrization), and configuration duplication across similar modules.

## Anchor examples

- **Logic extraction**: "this logic is exactly repeated from above, consider to extract" — Same algorithm appearing twice; should pull into a named helper to avoid maintenance divergence (lib/zephyr/external_sort.py).

- **Helper duplication**: "could we just do ... to not duplicate the logic" — Duplicated setup/processing before calling a shared internal function; can simplify by extracting preconditions (lib/marin/processing/tokenize/tokenize.py).

- **Exception handler reuse**: "felt better than having to duplicate the `cloudpickle` exception handling in two places" — Same error-handling pattern in multiple branches; merits a shared internal helper (lib/zephyr/subprocess_worker.py).

- **Test parametrization**: "can we extract the row equality predicates?" with code snippet — Complex test logic repeated across multiple test cases; recommend pytest parametrize or fixture to avoid copying test bodies (lib/zephyr/readers.py).

- **Document/config duplication**: "this duplicates marin.yaml, let's remove" — Same content appearing in both a reference doc and generated/maintained separately; should link or remove the duplicate (lib/iris/OPS.md).

- **Cross-module extraction**: "lets make a data/text/chat.py and move the new chat-related stuff and shared helpers there" — Shared logic scattered across multiple modules; propose centralizing in a dedicated module (lib/levanter/data/text/datasets.py).

- **Data structure reuse**: "Replaced the duplicated `_SUPPORTED_MULTI_REGIONS` frozenset with `SUPPORTED_MULTI_REGIONS = frozenset(_ZONE_PREFIX_TO_MULTI_REGION.values())` imported from `bootstrap`" — Hardcoded constants duplicated when a canonical source already exists; derive from the single source (iris_pypi_mirror/spec.md).

- **Parallel implementations of the same operation (source clones, not test bodies)**: Two methods in production code that perform the same operation against different wires / wrappers / call shapes. Most often appears as a "legacy translator" sitting next to the new translator, or as `submit_X` and `enqueue_X` that differ only in the placeholder constructor argument. Reviewer language: "this is `foo` copy-pasted with a different transport," "two translators for the same payload."
  ```python
  def reconcile_request_from_plan(plan): ...      # builds new-wire request
  def legacy_translator_request(plan): ...        # builds old-wire request from same `plan.request.desired`
  # or:
  def submit_task(self, spec, ...): ...           # builds TaskAttempt, inserts in _tasks, spawns thread
  def enqueue_attempt(self, attempt, ...): ...    # 30 lines copy-pasted, differs only in spec source
  ```
  Why: Source clones drift; a fix in one is silently absent in the other. The right shape is usually one helper that takes the differing input as a parameter, not two helpers that share a body. This is distinct from test-body duplication (covered above) — source duplication is higher-severity because the divergence shows up in production.

- **Test-doubles re-implementing production logic**: A test fixture or `InProcess*Provider` class that re-implements the dispatch / translation logic of the production code, then asserts the production code matches it. Reviewer language: "this fixture is the production code, copy-pasted," "if I change the production logic this test will be wrong in the same way and still pass."
  ```python
  class InProcessLegacyProvider:
      def reconcile_workers(self, plans):
          # 70 lines mirroring worker_provider._reconcile_one
          ...
  ```
  Why: The point of a test double is to isolate the SUT from a dependency, not to mirror the SUT itself. A double that mirrors production gives false confidence: it passes when production is wrong in the same way. Replace with a thin recording adapter that observes inputs/outputs without re-deriving them.

## False-positive guidance

- **Different preconditions**: If two blocks look similar but handle different error conditions, states, or domains, they may legitimately diverge.
- **Intentional copies**: Experiment code, one-off scripts, and test fixtures often intentionally copy logic for isolation; duplication is acceptable there.
- **Parametrization cost**: If extracting a helper would require many parameters or union types just to cover edge cases, inlining is sometimes clearer.
- **Type system friction**: If a helper would need `Any` or `Protocol` to abstract multiple concrete types, the abstraction cost may exceed the duplication cost.
- **Frequency threshold**: Single duplication (two sites) is sometimes acceptable; flag when a pattern repeats three+ times or when future copies are highly likely.

## Suggested confidence floor

Raise this detector when the same block appears identically or near-identically (≤5% syntax variance) in two or more places within the same file or very close call sites, or when reviewers explicitly call out "extract," "factor," "reuse," or "duplicate" language.
