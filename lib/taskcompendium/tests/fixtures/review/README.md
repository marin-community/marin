# Recorded review response

[stack_pytest_long_evidence.json](stack_pytest_long_evidence.json) records a review
of a tokenizer task whose private expected IDs and missing fixture conflicted
with its public instructions. The evidence exercises long review-response parsing
and bounded evidence handling without calling the provider.

This fixture supplies the recorded judgment and its explanation. It supplies no
replacement task or grader. See the [fixture policy](../README.md).
