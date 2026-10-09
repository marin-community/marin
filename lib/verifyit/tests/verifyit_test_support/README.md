# Trusted test scorers

`bounded_grading.py` contains top-level test functions used by bounded-worker and script-callable tests. The worker reimports these functions in a child process, so keep them in this uniquely named package and import them as `verifyit_test_support.bounded_grading`. The package is test-only and is not included in the VerifyIT wheel.
