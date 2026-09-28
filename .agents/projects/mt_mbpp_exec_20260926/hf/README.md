---
license: cc-by-4.0
pretty_name: MT-MBPP (executable)
language:
- en
tags:
- code
- evaluation
configs:
- config_name: prompts
  data_files: data/prompts/*.jsonl
- config_name: signatures
  data_files: data/signatures.jsonl
- config_name: tests
  data_files: data/tests/*.jsonl
- config_name: results
  data_files: results/generations/*/*.jsonl
---

# MT-MBPP (executable)

An execution-based companion to the 17 MT-MBPP components of OLMo 3's OlmoBaseEval Easy suite, which score only bits
per byte. MT-MBPP is MBPP's 500 test problems translated by o4-mini into 17 languages
(`allenai/multilingual_mbpp`); it has no tests, and its prompts describe each task in words without naming the function
a test would call. This dataset adds both.

## Prompts (`data/prompts/<language>.jsonl`)

Each prompt is the native OLMo 3 prompt (three solved examples, then the target task description and an opening code
fence) with one added line after every task description: `Function signature: `...``. Removing those lines recovers the
native prompt exactly, and `reference` is the native gold continuation (the o4-mini solution and the closing fence).

Signatures are copied verbatim from the o4-mini reference solutions: the declaration of the function MBPP's Python
asserts call, up to the start of its body, with whitespace collapsed (8352 matched by name, 145 identified by the closest name or as the one function nothing else calls, 3 set by hand). Haskell uses the type signature. Bash
signatures are usage lines derived from how the reference reads its arguments (`name arg1 arg2`, `items...` for a
list passed as the remaining arguments, `< stdin` for input on standard input).

## Tests (`data/tests/<language>.jsonl`)

Each test is MBPP's Python test cases (same inputs, same expected outputs) written for the target language
against the disclosed signature: `imports` go above the solution and `main`, which exits non-zero on the first failed
case, goes after it. Python uses MBPP's own asserts. The other languages were translated by DeepSeek (`deepseek-flash`,
thinking effort `high`, output cap 16,384 tokens; 11 translations that hit the cap were redone with a 65,536-token
cap). Every test ran in a network-free sandbox (`mt-mbpp-sandbox:2277a08ead83`, one toolchain per language) twice: with the o4-mini
reference solution, which must pass, and with a stub whose tested function returns a fixed wrong value, which must
fail. The 424 tests that failed this check went back to DeepSeek once at effort `max` with the sandbox output;
the model was told not to adapt a test to a reference that disagrees with MBPP's expected outputs and to flag such
references instead.

`valid` marks the 8,166 of 8,500 tests that pass the check (bash 465, c 481, cpp 484, csharp 490, go 481, haskell 453, java 483, javascript 485, matlab 460, php 485, python 500, r 474, ruby 493, rust 481, scala 479, swift 483, typescript 489); only those documents are scored. Excluded tests
keep their reason: reference_wrong 296, reference_fails 38.

The sandbox is somewhat lenient: TypeScript is transpiled without type checking, C# gets a console project's implicit
usings, and a few widely used libraries beyond the standard libraries are installed because reference solutions use
them (Haskell regex-tdfa, split, vector and containers packages; Rust regex and num crates; bc, gawk, jq and rev for
bash; PHP mbstring). Script languages must print a completion marker after the tests, so a solution that exits early
cannot pass.


## Results (`results/`)

Greedy completions (at most 1,024 tokens, cut at the closing fence) from four 1e21-FLOP checkpoints of the data-mixing
paper, each joined with its pass or fail on the valid tests (`results/generations/<mixture>/<language>.jsonl`).
Pass@1 per language and its 17-language mean are in `results/summary.json` and `results/components.csv`; the 95%
intervals resample MBPP problems within each language.

| Mixture | Mean pass@1 | 95% interval |
|---|---|---|
| Proportional | 4.4% | 4.0-4.9 |
| UniMax-8 | 8.9% | 8.3-9.6 |
| Olmix | 12.5% | 11.8-13.2 |
| MARINER | 15.7% | 15.0-16.5 |

## Sources

- `google-research-datasets/mbpp` (full), revision `4bb6404fdc6cacfda99d4ac4205087b89d32030c`, CC-BY-4.0.
- `allenai/multilingual_mbpp`, revision `c86b037e7d20e705f3595b1cbb9db993681ca181`; its card states no license.

Request manifest sha256 `707a2ddb4e9bf36a923660d364b453c2871f06e531dcc3c3e324ecf89c9d0bc5` (the prompts the evaluation froze on 2026-09-26). Built by
`experiments/domain_phase_mix/mt_mbpp_exec/` in the Marin repository, commit `d1efc52fc0d376827fed6b0998d2a59ee241a5ba` with uncommitted changes in that directory.
