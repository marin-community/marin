# Canonical instruction checkers

Source: Marin 623134de90723e7f189036da005067f36817b6e7, vendor/evalchemy/eval/chat_benchmarks/IFBench (Allen Institute for AI, 2025). Original Apache-2.0 notices are retained.

Changes: package imports are relative; import-time NLTK downloads and package-directory writes are removed. `repeat:repeat_span`, `ratio:overlap`, `ratio:sentence_words`, `count:keywords_multiple`, `words:words_position` and `custom:sentence_alphabet` follow allenai/IFBench 1c40f0c, and `custom:csv_quotes` fails an empty field instead of raising; the other checkers are unchanged. The dataset declares pinned checker packages and NLTK data in its grading environment. Missing data raises an error; grading never downloads it.

Original source SHA-256:

- `instructions.py`: `fe169405312396cb843aa7dda755d160553a4510f47dc186b812dd471da76302`
- `instructions_util.py`: `b7d60e07bbb2c56e42ee1a4a1b2f7281ced6c7af3af2719d2dca322afd3c20b8`
- `instructions_registry.py`: `58a4797496ee0d8279d4e3837575123fc3052465f2b7c9a4b3fbc01d88343aed`
