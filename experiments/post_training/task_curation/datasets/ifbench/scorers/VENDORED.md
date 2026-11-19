# Open-Instruct IFEvalG scorer

The three `open_instruct/IFEvalG/` modules are copied without edits from
[allenai/open-instruct](https://github.com/allenai/open-instruct/tree/28c1816664491493242d453c2c7280ee56e6c20a/open_instruct/IFEvalG)
at `28c1816664491493242d453c2c7280ee56e6c20a` under Apache-2.0. `LICENSE` is the source repository license.
Namespace packages preserve the original absolute imports without importing the complete training stack.

The task-local grade script ports the `IFEvalVerifier` loop from `open_instruct/ground_truth_utils.py`
at that same pin: decode the singleton contract, remove answer/thinking wrappers, instantiate each original
constraint, remove null keyword arguments, and return the arithmetic mean of constraint rewards. Empty replies
score zero. Exceptions remain grader failures. No model-generated answer is presented as an oracle solution.

NLTK data and scorer dependencies come from the catalog's pinned `GRADER_PACKAGES` environment.
