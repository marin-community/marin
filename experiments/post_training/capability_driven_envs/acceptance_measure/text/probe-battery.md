# The probe battery (`run_probe_battery`) and `embed_file`

## run_probe_battery

Grades a fixed, code-generated battery of candidates against the last draft `run_build` produced, on the real
grader machine, and reports one line per probe: its class, the expectation, the reward and whether the
expectation held. You do not write the probes; they come from the draft's reference answer and the contract you
pass. Arguments (the contract the instruction states):

| field | values | default |
|---|---|---|
| `channel` | `reply`, `file` | `reply` |
| `answer_path` | the absolute path of the answer file (channel `file`) | none |
| `framing` | `exact`, `fenced`, `last`, `text` (a plain-text answer) | `exact` |
| `required_keys` | the top-level keys that make an object the answer (framing `last`) | none |
| `prose_allowed` | true or false | false |
| `transport_noise` | `strip` (BOM, CRLF and trailing whitespace are removed before parsing) or `reject` | `strip` |
| `closed_schema` | true: extra keys or entries are wrong | true |
| `labels` | `exact` or `aliases` | `exact` |
| `integers` | `strict` (6.0 is not an integer) or `integral_float_ok` | `strict` |
| `extra_digits_allowed` | true: more digits than required pass | true |

Expectations: `full` means at least full credit; `<=ceiling` means at most the grader's declared
`rejection_ceiling` (a submission failure counts as 0); `<full` means below full credit.

| probe class | expectation | catches |
|---|---|---|
| reference; keys reordered; trailing zeros added (extra digits allowed); BOM, CRLF, trailing whitespace (noise `strip`); the fence or the prose the framing allows | `full` | false rejects of equivalent forms |
| blank, whitespace only, keyword soup, bare values without keys | `<=ceiling` | graders that credit nothing |
| two candidates (hedged), a wrong copy before the right one | `<=ceiling` | several answers |
| prose around the JSON, a fence (framing `exact`) | `<=ceiling` | text around the JSON |
| a duplicated key with a wrong value first | `<=ceiling` | duplicate keys |
| one extra key (closed schema) | `<=ceiling` | open schema |
| a number as a string, an integer as `6.0` (integers `strict`), NaN | `<=ceiling` | type coercion |
| a label in another case (labels `exact`) | `<=ceiling` | label folding |
| BOM or CRLF (noise `reject`) | `<=ceiling` | the stated noise policy |
| the answer in the other channel only | `<=ceiling` | channel fallback |
| a 200 KB nested reply | `<=ceiling`, graded | crashes on adversarial input |
| a false-accept witness from an earlier round | `<full` | regressions |
| a false-reject witness from an earlier round | `full` | regressions |

A probe whose expectation follows from a contract field you did not state uses the default. If a probe fails
because the contract you stated is not the one you want, change the instruction and the arguments together.

## embed_file

`embed_file(program, file)` rewrites the assignment between `# embed: <file>` and `# end embed` in the program
with the file's exact text, and checks that the literal evaluates back to it. Use it for `grade_kit.py`
(`reference/helpers/grade_kit.py`) and for a `grade.py` you keep as a workspace file:

```python
    # embed: reference/helpers/grade_kit.py
    grade_kit = ""
    # end embed
```
