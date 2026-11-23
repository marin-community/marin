import { SPAN_LOOKAHEAD_MS } from './contracts.js';
import { createMeasurements } from './measurements.js';
import { driverPhases, driverSetup } from './phases.js';

export function createRunModel({ deployment, namespace, run, window, rows, configuration, driverRows }) {
  const { start_ms, end_ms, inferenceGpus } = run;
  const logs = driverRows ?? [];
  const configRows = configuration ? [{ name: 'async_run_configuration', body_json: configuration }] : [];
  const setup = driverSetup(logs, {
    includesStartup: start_ms !== null && window.from - SPAN_LOOKAHEAD_MS <= start_ms,
    asyncTraining: configRows.length > 0,
  });
  return {
    deployment, namespace, window,
    run: { ...run, ...setup, ...(driverRows === null ? { driverReadFailed: true } : {}) },
    ...createMeasurements(rows, window, configRows, inferenceGpus, {
      startMs: start_ms, endMs: end_ms, trainingRows: setup.trainingRows, driverPhases: driverPhases(logs),
    }),
  };
}
