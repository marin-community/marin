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

## V21 fixed-column diagnostic

The v21 task used source `d1edf1def2bfb604ca46a9cc8693186a50c309fc`.
After an image-pull backoff recovered without a new attempt, the task executed
and failed once at the unrecognized line-4 record. The retained aggregate says
only that the row is 107 ASCII bytes with six whitespace-delimited tokens, 70
spaces, and standalone public `Address` and `Source` words. It does not prove
the row text, token order, space distribution, or adjacent syntax.

For an unrecognized row of exactly 107 UTF-8 bytes, the diagnostic now applies
the six public table widths `18, 60, 6, 6, 6, 6` separated by the five
single-byte gaps established by the reviewed separator. For each column it
reports the column index, space-trimmed byte length, ASCII character-class
counts, non-ASCII byte count, token count, and exact standalone `Address` and
`Source` booleans. It separately reports whether each gap is one ASCII space.
The parser still rejects the row; the new fields do not expand accepted NCU
syntax.

The diagnostic includes no column or token value, token sequence, per-column
or per-token hash, raw or redacted row shape, adjacent line, path, or
environment value. The previously reviewed whole-line SHA-256 remains. The
compact diagnostic retains the 2,048-byte serialized-error cap and falls back
to a fixed non-content error if that cap is exceeded.

These aggregates are not a confidentiality guarantee. In particular, a
low-entropy column can be identifiable from its byte length and character-class
counts even though its value is not emitted. Behavioral tests cover exact byte
boundaries, every gap, non-ASCII and control bytes, public-word lookalikes,
private-value and ordering nonleakage, the serialized bound, and the production
NCU profiling boundary.

No v22 job, image build, GPU execution, or relaunch was performed for this
source-only diagnostic change.

## V22 line-1 diagnostic

The single v22 task used source
`75921c162d31642c14cfa6101421295c9030ec3a`. It authenticated the launcher and
manifest, reached the NCU SASS parser, and failed once because line 1 was not
the exact reviewed top-level separator. The old exception exposed no line-1
length, hash, structure, or text, and the temporary profiler output was not
retained. The v22 artifact therefore makes no claim about the rejected row.

A line-1 mismatch now uses the same bounded structural diagnostic as any later
unrecognized SASS record. It reports only line number 1, UTF-8 byte count,
whole-line SHA-256, aggregate character/token/public-vocabulary and closed
public-pattern facts, plus the reviewed fixed-column summary only when the row
is exactly 107 bytes. The exact line-1 separator remains the sole accepted
top-level record; this change does not relax parser acceptance.

The exception contains no raw or redacted row, adjacent record, path, or
environment value. The existing 2,048-byte serialized bound and fixed
non-content fallback remain. Tests cover blank, profiler-status, kernel
identity, header, instruction, separator-lookalike, Unicode, control, exact
line-bound, and oversized records, as well as private data through the
production profiler boundary.

No image build, GPU execution, v23 submission, or relaunch was performed for
this source-only diagnostic change.

## V23 line-1 separator-width observation

The single v23 task used source
`c7874a8fa8194772dfbe55827d9c2e8be0a14154`. It authenticated the launcher and
manifest, reached the NCU SASS parser, and failed once at the line-1 separator
gate. The bounded diagnostic reports 108 UTF-8 bytes, 103 hyphens, five ASCII
spaces, six non-whitespace tokens, and a maximum token width of 61 bytes. It
contains no raw line or adjacent record. The aggregate does not independently
prove every group width or their order.

The parser admits only the previously reviewed separator widths
`(18, 60, 6, 6, 6, 6)` and the new closed width tuple
`(18, 61, 6, 6, 6, 6)`, each joined by one ASCII space. The tuple selected at
line 1 fixes the exact separator literal for the remainder of that export, so
mixing the 107- and 108-byte forms fails. The existing fixed-width kernel row,
`Address Source` header, and instruction grammar remain unchanged; a later row
that differs under the 108-byte layout still fails with the bounded aggregate
diagnostic.

Behavioral tests run both accepted forms through the parser and NCU subprocess
boundary. They reject other group counts, widths, orders, gaps, mixed separator
forms, and a widened kernel row. No image build, GPU execution, v24 submission,
or relaunch was performed for this source repair.

## V24 kernel identity-width observation

The single v24 task used source
`c19e1bb8582da83519f746da67de88e0ca55f494`. It authenticated the launcher and
manifest, accepted the reviewed 108-byte line-1 separator, and failed once on
line 2. The bounded diagnostic reports 108 UTF-8 bytes, three non-whitespace
tokens, 72 ASCII spaces, 63 trailing spaces, and the exact public vocabulary
tokens `Kernel` and `Name`. It contains no raw identifier, row, or adjacent
record.

The kernel identity row now derives its total width and trailing padding from
the separator selected at line 1. The 60-byte source column requires a
107-byte row with 62 trailing spaces; the 61-byte source column requires a
108-byte row with 63 trailing spaces. Both retain the exact `Kernel Name`
prefix, eight following spaces, and the reviewed 26-byte CUDA identifier
grammar. Rows for the other width, changed trailing padding, or redistributed
internal padding fail before section admission. The exact selected separator
remains mandatory at both later table closes.

Behavioral tests exercise both layouts through the parser and NCU subprocess
boundary and reject mixed row and separator widths. No image build, GPU
execution, v25 submission, or relaunch was performed for this source repair.

## V25 selected-width header diagnostic

The single v25 task used source
`c31d2f8b16eeef716c70cffb108e560e0bc4be56`. It authenticated the launcher and
manifest, accepted the selected 108-byte line-1 separator, matching 108-byte
kernel identity row, and matching identity-table close, then failed once at
line 4. The bounded diagnostic reports 108 UTF-8 bytes, six non-whitespace
tokens, 71 ASCII spaces, and exact standalone public vocabulary tokens
`Address` and `Source`. It contains no raw line or adjacent record. The
aggregate does not prove token values or order beyond those public booleans,
spacing distribution, or column boundaries.

For an unrecognized row after line 1, the diagnostic now derives its six
fixed-column slices from the selected separator tuple. Both reviewed layouts,
`(18, 60, 6, 6, 6, 6)` and `(18, 61, 6, 6, 6, 6)`, report the same bounded
per-column aggregate fields and one-byte gap-validity booleans. A row whose
width differs from the selected layout receives no fixed-column report. The
exact `Address Source` header remains the only accepted header; fixed-column
classification is diagnostic only.

The diagnostic still contains no column or token value, token sequence,
per-column hash, raw or redacted line, adjacent record, path, or environment
value. Its whole-line SHA-256 and 2,048-byte serialized-error bound remain.
The aggregate fields are not a confidentiality guarantee for low-entropy
columns.

Behavioral tests cover both selected widths, mismatched widths, every gap,
private-value and ordering nonleakage, and the production NCU subprocess
boundary. No image build, GPU execution, v26 submission, or relaunch was
performed for this source-only diagnostic change.
