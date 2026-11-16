# POLARIS deterministic math reward

The source is [ChenxinAn-fdu/POLARIS](https://github.com/ChenxinAn-fdu/POLARIS/tree/eda12fe22dd83aca59ec46a67147240dc483f62b)
at `eda12fe22dd83aca59ec46a67147240dc483f62b`, Apache-2.0.
`polaris_math.py` is the unchanged `deepscaler/rewards/math_utils/utils.py`;
`extraction.json` records its byte hash, and `LICENSE` preserves the upstream grant.

In upstream `verl/verl/trainer/main_ppo.py`, `RewardManager` selects
`deepscaler_reward_fn` for the released empty `data_source`. It grades the decoded public prompt and response.
The adapter in `../grade.py` preserves that function's deterministic path:
require `</think>`, take the segment immediately after its first occurrence,
extract the boxed answer and compare using the original mathd or SymPy helpers.
A bare correct box receives zero. This format requirement is part of the native
contract. The source defaults `enable_llm=False`; no judge fallback is enabled.

`POLARIS_GRADER_PACKAGES` in `datasets/environments.py` pins SymPy 1.13.3 and
pylatexenc 2.10. Upstream leaves these versions unspecified. The pipeline resolves
this declaration to a hash-locked environment before conversion. Public problems and hidden
answers remain unchanged, including proof requests and malformed answer keys.
The dataset's `answer` field is a grading key; it supplies no executable passing oracle.

Run the packaged-grader behavior tests with its optional parser dependency:

```bash
uv run --with pylatexenc==2.10 pytest experiments/post_training/task_curation/tests/test_polaris.py
```
