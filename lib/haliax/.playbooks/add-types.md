# Adding NamedArray Type Annotations

This playbook explains how to migrate existing code to use the new type
annotation helpers described in `docs/typing.md`.

1. **Read the documentation**: Familiarise yourself with `docs/typing.md`.
   It describes the `Named[...]` syntax and the dtype aware helpers in
   `haliax.typing`.
2. **Annotate parameters**: Replace plain `NamedArray` annotations with
   `Named[...]` that lists the required axes.  Use ellipses or sets when the
   exact order is flexible.

   ```python
   from haliax import Named

   # old
   def foo(x: NamedArray) -> NamedArray:
       ...

   # new
   def foo(x: Named["batch", "embed"]) -> Named["batch", "embed"]:
       ...
   ```
3. **Annotate dtypes when needed**: If the dtype matters, import symbolic dtypes
   from `haliax.typing` (e.g. `ht.f32`, `ht.i32`).  They can be indexed just
   like `Named`.

   ```python
   import haliax.typing as ht

   def bar(x: ht.f32["batch"]):
       ...
   ```
4. **Runtime validation**: Use `arr.matches_axes(...)` to check that a
   `NamedArray` conforms to the expected axes and dtype at runtime.

   ```python
   if not arr.matches_axes(Named["batch embed ..."]):
       raise ValueError("unexpected axes")
   ```
5. **Update return types**: Functions returning `NamedArray` should annotate
   their return values using the same conventions.

Following these steps will gradually port legacy code to the new
annotation style.
