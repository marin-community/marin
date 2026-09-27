# Executable MT-MBPP

Pass@1 for the 17 MT-MBPP components of OlmoBaseEval Easy. `allenai/multilingual_mbpp` has no tests and its prompts
never name the function a test would call, so the upstream suite scores these components by BPB only. This package
adds the tested function's signature to each prompt, translates MBPP's asserts into each language, and runs the
completions in a network-free Docker sandbox. The run record, plans, and results for the paper's four 1e21
checkpoints are in `.agents/projects/mt_mbpp_exec_20260926/`.

## Pinned inputs

| Input | Pin |
|---|---|
| MBPP (full) | `google-research-datasets/mbpp` at `4bb6404fdc6cacfda99d4ac4205087b89d32030c` (CC-BY-4.0) |
| MT-MBPP | `allenai/multilingual_mbpp` at `c86b037e7d20e705f3595b1cbb9db993681ca181` (no license stated) |
| Source snapshot | `data/hf_sources.json.gz` in the project record: both datasets' prompt and test splits at those revisions |
| Native requests | the 8,500 `mt_mbpp_*` rows of `gs://marin-us-east5/raw/eval-datasets/olmo_base_eval_table9/v2`, saved as `data/native_mt_mbpp_requests.jsonl.gz` |
| Request manifest | SHA-256 `77603128a951f725aa1a31be62236d0a21dd7aac7c7a7ed530df838d134f9007` (`NATIVE_PARITY_VERIFIED` in `evaluate_table9_accuracy.py`) |
| Test translator | DeepSeek `deepseek-flash`, thinking effort `high` (first pass) and `max` (repair) |
| Sandbox image | `mt-mbpp-sandbox:2277a08ead83`, built from `sandbox/Dockerfile`; `sandbox/versions.txt` lists its toolchains |
| Released data | Hugging Face `Calvin-Xu/mt-mbpp-exec` (private) at `4d4f9c4`: prompts, signatures, validated tests, completions and grades |

`sandbox.py` names the image by the first 12 hex digits of the Dockerfile's SHA-256, so an edited Dockerfile needs a
rebuild before any program runs. The Dockerfile pins its base image by digest, apt packages to the Ubuntu snapshot of
2026-09-25, direct downloads by SHA-256, and the five direct Rust crates by exact version. Cargo resolves the
transitive Rust crates at build time; `sandbox/versions.txt` records the ones in the image that graded the results.

## Reproduce

Run from the repository root with `P=.agents/projects/mt_mbpp_exec_20260926`. Step 3 needs Iris and the checkpoints,
step 4 a `DEEPSEEK_API_KEY`, step 5 read access to the plans' result paths in GCS, and steps 4 and 5 a running Docker
engine with the sandbox image.

1. Build the sandbox (native arm64, about 8 GB):

   ```bash
   cd experiments/domain_phase_mix/mt_mbpp_exec/sandbox && docker build -t mt-mbpp-sandbox:$(shasum -a 256 Dockerfile | cut -c1-12) .
   ```

2. Build the signature-disclosing requests and `$P/signatures.jsonl.gz`:

   ```bash
   uv run python -m experiments.domain_phase_mix.mt_mbpp_exec.build_requests --native $P/data/native_mt_mbpp_requests.jsonl.gz --sources $P/data/hf_sources.json.gz --output $P/requests
   ```

3. Generate on TPU. `--prepare` writes a frozen plan and uploads the requests to the region's bucket; the launch
   commands in `$P/launch_*_commands.txt` and `$P/euw4/` submit it with `--submit --mode mt_mbpp`. Decoding is greedy,
   at most 1,024 tokens, and stops at the closing code fence.

   ```bash
   uv run python -m experiments.domain_phase_mix.evaluate_table9_accuracy --prepare --region us-east5 --tpu-type v6e-4 --checkpoint-plan $P/checkpoints_east5.json --requests $P/requests --plan $P/plan_east5.json
   ```

4. Translate and validate the tests. A test is kept only if the o4-mini reference solution passes it and a stub that
   returns a fixed wrong value fails it. `--repair` retranslates the failures at the higher effort.

   ```bash
   uv run python -m experiments.domain_phase_mix.mt_mbpp_exec.translate_tests --signatures $P/signatures.jsonl.gz --sources $P/data/hf_sources.json.gz --output $P/tests/translations.jsonl --log $P/tests/translate.log
   uv run python -m experiments.domain_phase_mix.mt_mbpp_exec.validate_tests --signatures $P/signatures.jsonl.gz --translations $P/tests/translations.jsonl --output $P/tests/validation.jsonl --log $P/tests/validate.log
   ```

5. Grade the generations and summarize pass@1 per language, with a bootstrap over MBPP problems:

   ```bash
   uv run python -m experiments.domain_phase_mix.mt_mbpp_exec.grade_generations --plan $P/plan_east1.json --plan $P/plan_euw4.json --signatures $P/signatures.jsonl.gz --translations $P/tests/translations.jsonl --validation $P/tests/validation.jsonl --output $P/grading --log $P/grading/grade.log
   uv run python -m experiments.domain_phase_mix.mt_mbpp_exec.summarize_results --grading $P/grading --output $P/results
   ```

6. Collect the validated tests and the graded completions, then export the Hugging Face folder:
   `collect_tests --project $P`, `collect_results --project $P`, and `export_hf --project $P --output $P/hf`.
   `export_hf --push` refuses a public repository.

The translations are sampled from an LLM, so a rerun of step 4 produces different tests. To regrade with the paper's
tests, decompress `$P/tests/translations.jsonl.gz` and `$P/tests/validation.jsonl.gz` (step 4's outputs, committed) and
start at step 5. `$P/grading/plan_*/` holds the committed per-problem grades.
