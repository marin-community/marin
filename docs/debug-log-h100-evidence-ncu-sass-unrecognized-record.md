# H100 evidence v18 unrecognized NCU SASS record

The single reviewed v18 H100 task used source
`2779523b2c42218810bffc90c047a7f90d21aa81` and the immutable evidence image
`sha256:2efa83fbf8f2073a4175eef919a9f0c6c2db435d7c1ec9ed79017e1ea0d10cef`.
It failed once, without retry, after the exact line-1 table separator passed and
the parser rejected line 2 as an unrecognized record.

The temporary NCU report and SASS export were not retained. The original
exception contained no line-2 content, so the failure artifact does not infer
its bytes, identity, or surrounding structure.

The parser reports the offending line number, UTF-8 byte count, SHA-256, and an
aggregate-only structural summary under a 2,048-byte serialized-error bound.
The summary contains aggregate whitespace, delimiter, ASCII character-class,
and token-size counts. Fixed booleans report only the public words `Kernel`,
`Name`, `Address`, `Source`, `Section`, and `Function`, plus the closed section,
header, separator, instruction, and profiler-status patterns.

The exception never contains profiler line text because it crosses into
external job logs and that text can contain source or paths. It also omits token
values, per-token hashes, character positions, shape strings, adjacent records,
file paths, and environment values. Behavioral tests cover the production
profiling boundary, 512- and 513-byte records, escape-heavy text, Unicode and
control characters, exact aggregate counts, token-order ambiguity, NUL
rejection, and private tokens. Parse rejection remains fail closed.

The aggregate counts are not a confidentiality guarantee for low-entropy input;
some short rows can be inferred from their count vector. The contract prevents
direct line, token-value, and token-sequence disclosure while retaining the
explicitly reviewed whole-line SHA-256.

No image build, GPU execution, or relaunch was performed for this diagnostic
change.
