# Session Context

## User Prompts

### Prompt 1

last PR introduced some test failures in:


```
FAILED lib/zephyr/tests/test_execution.py::test_no_duplicate_results_on_heartbeat_timeout[local] - RuntimeError: current_actor() called outside of an actor context
FAILED lib/zephyr/tests/test_execution.py::test_no_duplicate_results_on_heartbeat_timeout[iris] - RuntimeError: current_actor() called outside of an actor context
FAILED lib/zephyr/tests/test_execution.py::test_no_duplicate_results_on_heartbeat_timeout[ray] - RuntimeError: current_actor(...

