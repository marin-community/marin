# Pyrefly shape types: issues and questions for the Pyrefly and JAX teams

Marin annotated its raw-array JAX code with Pyrefly shape types, using pyrefly `1.4.0.dev2` and the `pyrefly-shape-extensions`, `pyrefly-jax-stubs` and `pyrefly-einops-stubs` packages from the matching tag (`ef08065bc`). The pass replaced 1,389 jaxtyping shape strings with about 2,330 `jax.Array[[...]]` annotations across 80 files: Pallas and XLA kernels, expert-parallel MoE dispatch, attention, optimizers, and the model templates of a 535B-A23B MoE training run. The result checks clean with 34 suppressions; the work is on the [`pyrefly-shape-types`](https://github.com/marin-community/marin/tree/pyrefly-shape-types) branch.

The type system handled the code well. The cost came from the items below, listed roughly in order of impact. Unless marked otherwise, each repro was confirmed standalone at this pin.

## Bugs

1. **An array with no known shape binds the stubs' rank-0 default.** Stub functions declare `Shape: _Shape = []` so that Python scalars work. A bare `jax.Array` (shape `IntTuple`) does not bind `Shape`, so it falls through to `[]`:

   ```python
   def f(x: jax.Array) -> None:
       reveal_type(jax.nn.softmax(x))  # Array[[]]
       jnp.sum(x, axis=0)              # ERROR axis out of bounds
   ```

   It also produces wrong shapes with no error, because broadcasting against the "rank-0" operand keeps the other operand's shape:

   ```python
   def g[M: IntVar, H: IntVar](x: jax.Array[[M, H]], i: jax.Array[[M]], keep: jax.Array[[M]]) -> None:
       reveal_type(jnp.where(keep[:, None], x[i], 0))  # Array[[M, 1]]; x[i] is unshaped
   ```

   Impact in Marin:
   - About 150 false positives on unannotated code.
   - Most workaround annotations: every op that drops a shape needs an annotated local before the next elementwise op.
   - One real `[M, 1]`-vs-`[M, H]` error went unnoticed until we compared baselines (item 6).

   Could an argument with an unknown shape bind `Shape` to the unknown shape, instead of the default?
2. **The stubs' `jax.Array` and real JAX's `jax._src.basearray.Array` are different classes.** `jax.typing` is not stubbed, so `jax.typing.ArrayLike` names the real class:

   ```python
   def f(x: jax.typing.ArrayLike) -> None: ...
   f(jnp.zeros(3))  # ERROR jax._array.Array[[3]] is not assignable to jax._src.basearray.Array | ...
   ```

   The same split affects outputs of `jax.ffi.ffi_call`, `jax.ops.segment_sum`, and `ArrayImpl` from `make_array_from_*`. `ArrayLike` appears 61 times in Marin's JAX code. That includes haliax's `NamedArray | ArrayLike` dispatch, which cannot take a shaped array today. Could `jax.typing` be stubbed, or the two classes unified?
3. **A whole-shape array and a bare array are not assignable to each other.** For `x: jax.Array[S]` with `S: IntTuple`, passing `x` to a parameter typed `x: jax.Array` fails. Returning a bare `jax.Array` where `jax.Array[S]` is declared also fails. `jax.Array[[B, D]]` works in both directions.
4. **`reshape` stubs.** `x.reshape(n, -1)` keeps a literal `-1` dimension. `x.reshape(tuple_value)` returns a bare `Array`, while `jnp.reshape(x, tuple_value)` is precise.
5. **Broadcasting two different variadic prefixes loses both.** `base[..., :, None] * w`, with `base: [*B1, P]` and `w: [*B2, P, R]`, gives `Array[[*tuple[int, ...], P, R]]`.
6. **`--baseline` matches on (path, error code, column) and ignores the message and the count.** Two baseline entries at one location masked three errors, and the third was new. A changed error at a baselined location also stays hidden. We now diff per-location counts after every `--update-baseline`. Is this intended, and could matching include the message or a count?
7. **Diagnostics.**
   - `[*Batch: IntTuple]` in a type-parameter list is invalid Python, since a `TypeVarTuple` takes no bound. It is an easy mistake because shapes are written `[*Batch, D]`, and the nine parse errors that follow never say what is wrong.
   - `tensor-shapes = true`, which was needed in 1.0, now only produces an "extra keys" warning.

## Stub coverage gaps

Each op below returns an unshaped (`Array`), unknown (`Unknown` or `Any`), or widened result for shaped inputs. Because of item 1, each one then needs an annotated local.

| Op | Result for shaped input | Where Marin hits it |
|---|---|---|
| `jax.lax.top_k(x, k)` with a non-literal `k` | `tuple[Array, Array]` (literal `k` works) | MoE routing, capacity selection |
| `jax.lax.dot_general` | `Array` | cross-entropy logits |
| `jnp.einsum` with `...` in the spec | `Array` | model layers |
| advanced indexing `x[idx]` | `Array` (`jnp.take(x, idx, axis=0)` is precise) | MoE gather and scatter |
| `jax.sharding.reshard`, `lax.with_sharding_constraint` | `Any` or `Array` | every model |
| `lax.psum_scatter`, `lax.all_gather`, `lax.all_to_all` | `Array` | expert-parallel dispatch |
| `jax.vmap(f)(x)` | `Any` | kernels, optimizers |
| `@jax.custom_vjp` on a generic function | dimensions widened to `int` | kernels, MoE |
| `functools.partial` of a generic function | type variables re-instantiated, so `Array[[B, D]]` is reported as not assignable to `Array[[B, D]]` | `shard_map(partial(local_fn, ...))` |
| `jax.ffi.ffi_call` | real-JAX `Array` with no link to the `ShapeDtypeStruct`s | DeepEP transport |
| `jax.nn.one_hot`, `jax.nn.logsumexp` | `Unknown` | losses |
| `jax.lax.bitcast_convert_type` | `Array` | checkpoint serialization |
| `jnp.matmul(..., preferred_element_type=...)`, `precision=` | unexpected keyword | SSD, SOAP |
| `jnp.tensordot(a, b, axes=((0,), (1,)))` | `bad-specialization` | SOAP optimizer |
| `einops.rearrange` / `repeat` on JAX arrays | gradual; the einops shape stubs are torch-only | attention in every model |

## Solver and language limits

- **Arithmetic in parameter annotations.** `x: jax.Array[[2 * H]]` cannot be solved at call sites, and variables are solved left to right, so every type variable needs a bare occurrence. Are inverse solutions for `k * N` or `N // k` planned?
- **Divisibility and simplification.** `2 * (D // 2)` does not simplify to `D`. This shows up when rotary embeddings split a head in half and concatenate the halves. `(S // C) * C` does not simplify to `S` either. Could an assertion such as `assert d % 2 == 0` refine a dimension?
- **Narrowing from runtime checks.** `if hq == hkv: return x`, `if x.ndim == 2`, and `%` checks do not narrow dimension variables. We suppress the early returns that depend on them.
- **Unions of shapes do not bind one variable.** Passing `Array[[G, P]] | Array[[P]]` to a parameter of type `[*Batch, P]` fails. We replaced these unions with a single variadic parameter.
- **Named constants cannot be dimensions.** `HIDDEN: Final = 64` and `type Hidden = Literal[64]` are both rejected in an annotation, so literals get duplicated.
- **Generic aliases with dimension parameters.** A bare PEP 695 alias works. Using a parameterized alias inside another generic signature reports `B is an IntVar and cannot be used as an ordinary type`.
- **Reassigning a parameter.** A parameter keeps its declared shape across reassignment, so `x = x.T` needs a new name. Is this by design?

## Runtime and packaging

- **For JAX: runtime subscripting.** `jax.Array` has no `__class_getitem__`, so any shape annotation Python evaluates eagerly raises `TypeError`. This includes a module without `from __future__ import annotations`, a `cast(jax.Array[[...]], x)`, and a class base list. Importing `shape_extensions` avoids this only because it monkeypatches `jax.Array`, so correctness depends on import order. Could `jax.Array` accept subscripts natively?
- **`import shape_extensions` side effects.** It imports torch when torch is installed, which takes 0.45 s of its 0.66 s import time, and it patches classes. Marin's style rules forbid `TYPE_CHECKING` imports, so every annotated module pays this cost. Could the patches be lazy?
- **Distribution.**
  - The stubs are not on PyPI under their names; an unrelated project owns `pyrefly-jax-stubs`.
  - They pin `pyrefly-shape-extensions==0.0.0`, so we install all four from git at a pinned commit.
  - What is the plan for published, versioned stubs, and for shape types in a stable pyrefly release? Is the `Array[[...]]` syntax stable enough to adopt now?

## Direction

- **dtype.** The stubs model shapes only. Converting from jaxtyping dropped 255 non-float markers: 212 `Int` (expert ids, offsets, group sizes, labels), 42 `Bool` (masks) and 1 `Key`. Wrapping native shapes in jaxtyping markers, as `Int[jax.Array[[T, K]], "..."]`, keeps the dtype as documentation, and Pyrefly still checks the shape. Is dtype modeling (`Array[Shape, DType]`) planned, and which spelling should a codebase use until then?
- **Sharding.** A `shard_map` body sees per-shard shapes, such as the global token count divided by the expert-mesh size, but nothing ties them to the caller's global shapes. Recent Marin failures came from inconsistent or incorrectly inferred sharding specs: [#8467](https://github.com/marin-community/marin/pull/8467), [#8861](https://github.com/marin-community/marin/issues/8861) and [#8911](https://github.com/marin-community/marin/issues/8911). What would sharding in types need from users?
- **Named axes.** Most of Marin's model code uses haliax `NamedArray`, whose axes are identified by name and can be reordered freely. Positional shapes cannot describe it. Is anything planned for name-keyed shapes?
- **Migration style.** Native annotations or `@static_jaxtyping` for a codebase moving off jaxtyping? We chose native because it allows `H * D` and `Int[N]` binding, and because Marin does not run jaxtyping's runtime checker.
