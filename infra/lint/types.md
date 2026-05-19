# types — detector prompt

## What to look for

Flag code that uses loose or imprecise type declarations where Marin's guidelines recommend stricter alternatives. Common issues: bare `Any` types that could be specific (e.g., `list[Any]` should be `list[SpecificType]`), boolean flags that should be enums, tuple return types that should be dataclasses, and raw dicts/dicts that should be dataclasses. Also flag missed opportunities for `Protocol` decoupling, non-auto enum values, and incorrect `isinstance` checks instead of type assertions.

## Anchor examples

1. **Bare Any where type is known** (PR #5290)
   - Reviewer: "nit: why are we typing it as `list[Any]` if we know it's expected to be `list[logging_pb2.LogEntry]`? This pattern is repeated in other cases on 'non-public' methods."
   - Code shape: `payloads: list[Any]` in a method that always works with a specific proto type
   - Real concern: Loses type safety and makes refactoring harder; type checkers can't catch mismatches

2. **Boolean flags instead of enums** (PR #4761)
   - Reviewer: "nit: I have almost never created a boolean argument without realizing I want to change it to an enum: enum DedupMode = { NONE, EXACT }"
   - Code shape: `dedup: bool` parameter that gates between two distinct behaviors
   - Real concern: Boolean flags fail to scale (more modes later require ugly nested bools); enums are clearer and type-safe

3. **Complex tuple returns instead of dataclass** (PR #5276)
   - Reviewer: "`tuple[dict[str, dict], str, bool]` is fairly complicated return type to read, a dataclass would make it easier to reason"
   - Code shape: Function returning `tuple[dict[str, dict], str, bool]` with multiple fields
   - Real concern: Tuple order is implicit and fragile; dataclass fields are self-documenting and refactoring-safe

4. **Dict/dict where dataclass is appropriate** (PR #5276)
   - Reviewer: "the records should be defined as dataclasses"
   - Code shape: Bare dicts or ad-hoc dict-building code representing structured entities
   - Real concern: No schema validation; type checkers can't validate field access; harder to evolve

5. **Missing Protocol for polymorphic behavior** (PR #5430)
   - Reviewer: "Instead of doing all these `if` branches would it be reasonable to define a protocol and have 2 flavors of the `TreeCache`?"
   - Code shape: Class with multiple branching conditionals on a variant field/flag
   - Real concern: Hard to extend; encourages boolean dispatch over concrete types

6. **Missing `isinstance` type narrowing** (PR #4434)
   - Reviewer: "I think this should be `isinstance`?"
   - Code shape: Type check using non-`isinstance` method (e.g., manual attribute checks)
   - Real concern: Type checkers don't narrow the type; code is less maintainable

7. **Non-auto enum values** (PR #4816)
   - Reviewer: "nit: you can use `enum.auto`"
   - Code shape: `class MyEnum(Enum): A = 1, B = 2, C = 3`
   - Real concern: Manual numbering is fragile and error-prone; `auto()` is clearer intent

8. **NamedTuple vs Tuple instantiation confusion** (PR #5385)
   - Reviewer: "you have to construct tuples and NamedTuples differently... (namedtuples are instantiated Foo(*iter) but tuples have to instantiated tuple(iter)"
   - Code shape: Code that tries to generically construct tuples/namedtuples with the same syntax
   - Real concern: Type errors at runtime; confusion between structural and named tuples

## False-positive guidance

- Genuine `Any` at API boundaries is fine (e.g., `cache[K, V]: dict[str, Any]` in a generic store that intentionally decouples from value types).
- Protocol dispatch via `if isinstance(x, FooProto)` is expected and not a flag for alternate types.
- Return type `tuple[T, ...]` for variable-length sequences or numeric coordinates is reasonable; only flag fixed-field tuples >2 items.
- Dataclasses with `@dataclass(slots=True)` or performance-critical inner loops may reasonably use simpler types; respect author intent if documented.
- Boolean flags in library APIs that are already public/stable should not be retroactively converted (breaking change); prefer to add typed variant.

## Suggested confidence floor

Raise confidence when the comment explicitly calls out a concrete alternative (e.g., "use `enum.auto`", "this should be a dataclass") and the change is localized to one function or small file section.
