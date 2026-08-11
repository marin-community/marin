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

## V19 aggregate observation

The single v19 task used source
`e23df72a667e50a7f7fb1fd13b8c4b75b4cd1d01`. It failed once, without retry,
at the same line-2 parser boundary. The bounded diagnostic reports 107 UTF-8
bytes, 71 spaces, no leading spaces or tabs, 62 trailing spaces, three tokens,
and exact standalone `Kernel` and `Name` words. The longest token is 26 bytes.
There are no colons, commas, pipes, hyphens, controls, or non-ASCII codepoints.
The remote row text remains unavailable.

The parser now admits only the corresponding fixed-width public record:
`Kernel Name`, eight padding spaces, one 26-byte CUDA identifier, and 62
trailing padding spaces. The line is exactly 107 bytes. The identifier uses the
same closed ASCII letter, digit, and underscore grammar as normalized CUDA
names, and the parsed name must still match the independently expected NCU
kernel coverage. The former whitespace-tolerant colon form had no retained
profiler evidence and was removed.

Behavioral tests exercise the parser and the NCU subprocess boundary. They
reject changed width, internal or trailing padding, token order/count,
colon-form rows, identifier lookalikes, and invalid identifier characters.
No image build, GPU execution, or relaunch was performed for this parser
change.

## V20 identity-table close observation

The single v20 task used source
`d45ea8fefb7ad73677136b7e6d53d711a46e34da`. It failed once, without retry,
after the line-1 table separator and line-2 fixed-width kernel identity row
passed. The parser then recognized the exact 107-byte separator at line 3 but
rejected it because the kernel identity table had no reviewed close state.

The parser now requires that exact separator immediately after every accepted
fixed-width kernel identity row. A missing, duplicate, moved, or changed close
fails before the address-source table can begin. The previously reviewed
`Address Source`, per-section separator, and instruction grammar remains
unchanged.

The retained v20 evidence proves no line-4 or later syntax. In particular, the
existing address-source grammar is retained from earlier synthetic fixtures,
not inferred from the v20 traceback. The remote NCU report and SASS export were
not retained. No image build, GPU execution, or relaunch was performed for
this repair.
