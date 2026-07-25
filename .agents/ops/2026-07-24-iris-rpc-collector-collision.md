# Iris RPC metrics: duplicate collector registration

Restore green Iris E2E coverage after #7598 moved RPC metrics into a custom
Prometheus collector.

## Initial status

`iris-e2e-smoke` fails in `test_checkpoint_restore` when its dedicated
`LocalCluster` starts while the module-scoped smoke cluster is still running.
The second controller raises `ValueError: Duplicated timeseries in
CollectorRegistry` for the `iris_rpc_*` families.

## Hypothesis 1

Controller teardown fails to unregister the collector.

The failure occurs before the dedicated cluster restarts. The module-scoped
smoke controller is intentionally still live, so teardown cannot solve the
initial collision. Telltale's registry is process-global, while the E2E process
can host more than one local controller.

## Changes to make

Represent all native proxies in one process-level Iris RPC collector and
aggregate equal service/method/upstream series. Controller shutdown detaches its
proxy; the collector remains registered like Telltale's counter, gauge, and
histogram objects.

Add a behavior test that attaches two native metric sources and verifies the
public Prometheus families expose their combined counters, gauges, statuses,
and histogram values.

## Results

The two-controller regression test failed before the change with the same
`Duplicated timeseries` exception as CI and passed after aggregation.

The native proxy and authentication suites passed 43 tests against the stable
PyPI wheel. The complete local smoke suite, including
`test_checkpoint_restore`, passed 24 tests with one skipped.

## Future work

- [ ] Automate native release pins after stable publication; tracked in #6456.
