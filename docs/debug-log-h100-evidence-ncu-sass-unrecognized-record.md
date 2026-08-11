# H100 evidence v18 unrecognized NCU SASS record

The single reviewed v18 H100 task used source
`2779523b2c42218810bffc90c047a7f90d21aa81` and the immutable evidence image
`sha256:2efa83fbf8f2073a4175eef919a9f0c6c2db435d7c1ec9ed79017e1ea0d10cef`.
It failed once, without retry, after the exact line-1 table separator passed and
the parser rejected line 2 as an unrecognized record.

The temporary NCU report and SASS export were not retained. The original
exception contained no line-2 content, so the failure artifact does not infer
its bytes, identity, or surrounding structure.

The parser now reports only the offending line number, UTF-8 byte count, and
SHA-256 under a 2,048-byte serialized-error bound. It deliberately never puts
profiler line text in the exception because the exception crosses into external
job logs and that text can contain source or paths. It does not expose adjacent
records, a file path, environment values, or raw profiler content. Behavioral
tests cover the production profiling boundary, 512- and 513-byte records,
escape-heavy text, Unicode and control characters, NUL rejection, and private
tokens. Parse rejection remains fail closed.

No image build, GPU execution, or relaunch was performed for this diagnostic
change.
