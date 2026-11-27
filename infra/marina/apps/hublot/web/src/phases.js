export const phaseTypes = {
  startup: { label: 'Startup', color: '#e8eef5', detail: 'Recorded run start to the initial policy sync.' },
  rollout_wait: { label: 'Batch wait', color: '#faeccf', detail: 'The trainer was waiting for the Buffer to admit a batch.' },
  training: { label: 'Training', color: '#e1f1e8', detail: 'The trainer was running the recorded training step.' },
  weight_sync: { label: 'Weight sync', color: '#eadff5', detail: 'The trainer was transferring updated policy weights to inference.' },
  evaluation: { label: 'Evaluation', color: '#deecfa', detail: 'The trainer timed an evaluation. This can include draining training rollouts.' },
  checkpoint: { label: 'Checkpoint', color: '#f0e3eb', detail: 'The trainer timed a checkpoint save. A confirmed save message includes the final upload.' },
};

const timerMarkers = [
  { method: '_run_pretraining_evaluation', timer: 'eval_before_train', kind: 'evaluation', reason: 'Pre-training evaluation' },
  { method: '_run_step_end_callbacks', timer: 'eval', kind: 'evaluation', reason: 'Periodic evaluation' },
  { method: '_finalize_training', timer: 'eval', kind: 'evaluation', reason: 'Final evaluation' },
  { method: '_finalize_training', timer: 'save_checkpoints', kind: 'checkpoint', reason: 'Final checkpoint' },
].map((marker) => ({
  ...marker,
  pattern: new RegExp('skyrl_train\\.trainer:' + marker.method + ":\\d+ - Finished: '" + marker.timer +
    "', time cost: (\\d+(?:\\.\\d+)?)s"),
}));

const savedCheckpoint = /skyrl_train\.trainer:_finalize_training:\d+ - Saved final checkpoint\./;
const datasetSize = /skyrl_train\.dataset\.dataset:_read_files_and_tokenize:\d+ - Filtered dataset size:\s+(\d+)\s*$/;
const batchPolicy = /\bbatch_policy:\s+(full_batch|rolling)\s*$/;
const evalDisabled = /\beval_interval:\s+-1\s*$/;
export const driverTimestamp = /\b(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}\.\d+) \|/;
export const driverLinePatterns = [...timerMarkers.map(({ pattern }) => pattern), savedCheckpoint,
  datasetSize, batchPolicy, evalDisabled];

export const driverPhaseMarkers = [
  ...timerMarkers.map(({ method, timer }) => ({ caller: method, message: "Finished: '" + timer + "', time cost: " })),
  { caller: '_finalize_training', message: 'Saved final checkpoint.' },
  { caller: 'skyrl_train.dataset.dataset', message: 'Filtered dataset size:' },
  { message: 'batch_policy:' },
  { message: 'eval_interval:' },
];

function loggedTime(row) {
  const match = driverTimestamp.exec(row.line);
  const time = match ? Date.parse(match[1] + 'T' + match[2] + 'Z') : NaN;
  return Math.abs(time - row.epoch_ms) < 5000 ? time : row.epoch_ms;
}

export function driverPhases(rows) {
  if (new Set(rows.map((row) => row.key)).size !== 1) return [];
  const saved = rows.filter((row) => savedCheckpoint.test(row.line));
  return rows.flatMap((row) => {
    for (const { pattern, kind, reason } of timerMarkers) {
      const match = pattern.exec(row.line);
      if (!match) continue;
      const finish = loggedTime(row);
      const start = finish - Math.round(Number(match[1]) * 1000);
      const saveEnd = kind === 'checkpoint' && saved.length === 1 ? loggedTime(saved[0]) : finish;
      const end = Math.max(finish, saveEnd);
      return [{ id: 'log/' + row.seq, kind, start, end, recorded: end, duration: (end - start) / 1000,
        reason, source: 'Driver log timer' }];
    }
    return [];
  });
}

export function driverSetup(rows, { includesStartup = false, asyncTraining = false } = {}) {
  if (new Set(rows.map((row) => row.key)).size !== 1) return {};
  const messages = rows.map((row) => row.line);
  const policies = new Set(messages.flatMap((line) => batchPolicy.exec(line)?.[1] ?? []));
  // SkyRL constructs the training dataset first, then the eval dataset.
  const sizes = messages.flatMap((line) => datasetSize.exec(line)?.[1] ?? []);
  const trainingOnly = includesStartup && asyncTraining && messages.some((line) => evalDisabled.test(line));
  return {
    batchPolicy: policies.size === 1 ? [...policies][0] : null,
    trainingRows: sizes.length === 2 || (trainingOnly && sizes.length === 1) ? Number(sizes[0]) : null,
  };
}

export function startupPhases(startMs, initialSync) {
  if (!initialSync || !Number.isFinite(initialSync.duration) || initialSync.duration <= 0) return [];
  const end = initialSync.recorded;
  const syncStart = end - Math.round(initialSync.duration * 1000);
  const phases = [{ ...initialSync, id: 'initial-sync', start: syncStart, end, kind: 'weight_sync' }];
  if (Number.isFinite(startMs) && startMs < syncStart) {
    phases.unshift({ id: 'startup', kind: 'startup', start: startMs, end: syncStart, recorded: end,
      duration: (syncStart - startMs) / 1000, source: 'Controller start and initial sync' });
  }
  return phases;
}
