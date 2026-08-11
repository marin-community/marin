# H100 evidence NCU SASS separator

## Initial status

The reviewed v16 job authenticated its capsule and advanced through the repaired
wide NCU CSV units parser. Its first terminal failure was the public SASS parser
rejecting this exact 107-byte table separator:

```text
------------------ ------------------------------------------------------------ ------ ------ ------ ------
```

The six hyphen groups have widths 18, 60, 6, 6, 6, and 6. The bounded terminal
exception is sealed separately under
`h100_contract_map_evidence_sixteenth_launch_failure_69221a_v0`. The temporary
remote NCU report and SASS file were not durably exported, so this checkpoint
does not claim unavailable surrounding rows.

## Hypothesis

The previous metadata grammar admitted only a single all-dash group. Pinned NCU
renders one `Address Source` header and the observed six-group separator before
the address-bearing SASS records for each kernel section.

## Changes

The parser now requires, per exact kernel section, one `Address Source` header,
one separator with the observed group widths, and at least one allowlisted
address-bearing instruction. The separator is legal only after that header and
before the first instruction. Missing, repeated, misplaced, wrong-width, extra-
group, and dashed-text records fail closed.

Both direct text and the production file boundary are capped at 1 MiB, each
line is capped at 1,024 characters, and parse failures report only a line number
rather than serializing profiler text. The production NCU path uses the bounded
file reader before retaining report and SASS identities.

## Results

A real-format fixture covers two kernel sections. Behavior tests mutate every
separator boundary and exercise the real `_run_ncu_profile` boundary with only
the external profiler calls replaced. Exact kernel coverage, allowlisted SASS
mnemonics, and spill rejection remain mandatory.

## Validation boundary

This checkpoint contains source and local tests only. It does not build an
image, query a GPU, relaunch v16, or claim that later evidence gates pass.
