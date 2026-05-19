# types — detector prompt

## AGENTS.md anchor

§ Types & Data Structures — `Protocol` for decoupling; avoid `Any` where a
concrete type is known; prefer typed structures over loose dicts. Companion
detector: `api-shape` covers bool→enum and tuple→dataclass.

## What to look for

Flag loose static types: bare `Any` where the concrete type is known, missing
`Protocol` decoupling, non-`auto()` enum values, and missed `isinstance`
narrowing. Interface-shape issues (bool→enum, tuple→dataclass, parameter
unions) live in `api-shape` per the precedence rules in `lint.md`.

## Anchor examples

1. **Bare Any where type is known** (PR #5290)
   - Reviewer: "nit: why are we typing it as `list[Any]` if we know it's
     expected to be `list[logging_pb2.LogEntry]`? This pattern is repeated in
     other cases on 'non-public' methods."
   - Code shape: `payloads: list[Any]` in a method that always works with a
     specific proto type
   - Real concern: Loses type safety and makes refactoring harder; type
     checkers can't catch mismatches

2. **Missing Protocol for polymorphic behavior** (PR #5430)
   - Reviewer: "Instead of doing all these `if` branches would it be
     reasonable to define a protocol and have 2 flavors of the `TreeCache`?"
   - Code shape: Class with multiple branching conditionals on a variant
     field/flag
   - Real concern: Hard to extend; encourages boolean dispatch over concrete
     types

3. **Missing `isinstance` type narrowing** (PR #4434)
   - Reviewer: "I think this should be `isinstance`?"
   - Code shape: Type check using non-`isinstance` method (e.g., manual
     attribute checks)
   - Real concern: Type checkers don't narrow the type; code is less
     maintainable

4. **Non-auto enum values** (PR #4816)
   - Reviewer: "nit: you can use `enum.auto`"
   - Code shape: `class MyEnum(Enum): A = 1; B = 2; C = 3`
   - Real concern: Manual numbering is fragile and error-prone; `auto()` is
     clearer intent (unless the values are wire identifiers that must stay
     stable across versions).

5. **NamedTuple vs Tuple instantiation confusion** (PR #5385)
   - Reviewer: "you have to construct tuples and NamedTuples differently...
     (namedtuples are instantiated `Foo(*iter)` but tuples have to
     instantiated `tuple(iter)`)"
   - Code shape: Code that tries to generically construct tuples/namedtuples
     with the same syntax
   - Real concern: Type errors at runtime; confusion between structural and
     named tuples

6. **Raw dict where dataclass fits** (PR #5276)
   - Reviewer: "the records should be defined as dataclasses"
   - Code shape: Ad-hoc dict-building code representing structured entities
   - Real concern: No schema validation; type checkers can't validate field
     access; harder to evolve. (For *return-type* tuples, see `api-shape`.)

## False-positive guidance

- Genuine `Any` at API boundaries is fine (e.g., `cache[K, V]: dict[str, Any]`
  in a generic store that intentionally decouples from value types).
- Protocol dispatch via `if isinstance(x, FooProto)` is expected and not a
  flag for alternate types.
- Return type `tuple[T, ...]` for variable-length sequences or numeric
  coordinates is reasonable; api-shape flags fixed-field tuples >2 items.
- Enum values that double as wire identifiers (proto enum numbers, serialized
  IDs) must stay stable — do not flag missing `auto()` there.
- Dataclasses with `@dataclass(slots=True)` or performance-critical inner
  loops may reasonably use simpler types; respect author intent if
  documented.

## Suggested confidence floor

Raise confidence when the comment explicitly calls out a concrete alternative
(e.g., "use `enum.auto`", "this should be a Protocol") and the change is
localized to one function or small file section. Lower confidence on `Any`
where the surrounding code legitimately handles multiple unrelated types.
