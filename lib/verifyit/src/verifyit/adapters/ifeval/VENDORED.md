# Canonical instruction checkers

Source: EleutherAI/lm-evaluation-harness v0.4.12, lm_eval/tasks/ifeval. Original Apache-2.0 notices are retained.

Changes: package imports are relative; import-time NLTK downloads and package-directory writes are removed. Checker logic is unchanged. Install the instructions extra and provision NLTK data before grading. Missing data raises an error; grading never downloads it.

Original source SHA-256:

- `instructions.py`: `511cc41a53787d818c292d8335c8e98aa833296a0789ac470db56bcd6496345e`
- `instructions_util.py`: `e8c4d9187bac1482d93941fb46469609c3ae78d896195bfcb300a0707d492567`
- `instructions_registry.py`: `9db1c062cdb70a91420d3789a588235a1064c2b9a1a66e09e5fb9df9e7584a6c`
