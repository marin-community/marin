# H100 evidence v17 top-level NCU SASS separator

The single reviewed v17 H100 task used source
`87183412f719f78287568eef9d9feb0cd1bcaf0c` and the immutable evidence image
`sha256:2efa83fbf8f2073a4175eef919a9f0c6c2db435d7c1ec9ed79017e1ea0d10cef`.
It failed once, without retry, while parsing the NCU public SASS export.

The bounded exception proves that line 1 is the exact 107-byte literal
`------------------ ------------------------------------------------------------ ------ ------ ------ ------`.
The temporary report and SASS export were not retained, so this observation
does not prove any later line or whether the export contains per-kernel
separators. The parser now consumes exactly one copy of that literal at line 1
and keeps the previously reviewed per-section header, separator, and
instruction requirements unchanged.

The regression fixture starts with the observed line-1 record. Behavioral
tests reject a missing, moved, duplicated, padded, or arbitrary top-level
separator and independently retain the per-section separator mutations. No
image build, GPU execution, or relaunch was performed for this source repair.
