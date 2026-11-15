# BrowseComp-Plus source programs

Origin: [texttron/BrowseComp-Plus at 046949032b0328319cc9a02663a759ec601d9402](https://github.com/texttron/BrowseComp-Plus/tree/046949032b0328319cc9a02663a759ec601d9402).
License: MIT, preserved in `LICENSE`.

- `searchers/base.py` and `searchers/bm25_searcher.py` retain the upstream files.
- `decode.py` retains `derive_key`, `decrypt_string`, `transform_decrypt` and the
  public canary constant from `scripts_build_index/decrypt_dataset.py`.
- `prompts.py` retains the search-only query template and answer-equivalence
  grader template from `search_agent/prompts.py`.
- `judge_parser.py` retains `parse_judge_response` from
  `scripts_evaluation/evaluate_with_openai.py`.

The extracted modules add only their required imports and source attribution;
function and template source remains unchanged. `tool.py` outside this directory
adapts the pinned BM25 class and original snippet procedure to the task machine's
JSON transport. `grade.py` supplies the source prompt and parser with the
production judge connection. These wrappers do not substitute answer comparators.
