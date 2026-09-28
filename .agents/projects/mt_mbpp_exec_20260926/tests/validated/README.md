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
