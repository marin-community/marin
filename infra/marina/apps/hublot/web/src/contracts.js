// Graph, Signal, and Card are kept together so a handle or chart can be traced to Finelog.
const port = (label, side, offset, child) => ({ label, side, offset, child });

// The four-stage training loop forms a rectangle. The single-neighbor
// stages sit above it, with a separate pair of lanes for each worker handoff.
export const nodes = {
  source: {
    label: 'Task source', category: 'Input', position: [24, 24], size: [174, 100],
    ports: { entries: port('Dataset entries', 'bottom', 50) },
  },
  rollouts: {
    label: 'Rollouts', category: 'Stage', position: [385, 244], size: [199, 145],
    ports: {
      entries: port('Dataset entries', 'left', 24, 'dispatch'),
      lease: port('Lease grant', 'left', 76, 'dispatch'),
      request: port('Model input', 'right', 25, 'workers'),
      output: port('Model output', 'right', 66, 'workers'),
      call: port('Task action', 'top', 18, 'workers'),
      observation: port('Observation / reward', 'top', 40, 'workers'),
      submission: port('Grading submission', 'top', 64, 'workers'),
      score: port('Grading outcome', 'top', 85, 'workers'),
      group: port('Completed group', 'bottom', 50, 'workers'),
    },
  },
  dispatch: {
    label: 'Dispatch', category: 'Rollouts', position: [24, 244], size: [174, 145],
    ports: {
      entries: port('Dataset entries', 'top', 50),
      lease: port('Lease grant', 'bottom', 50),
      task: port('Rollout group job', 'right', 49),
    },
  },
  workers: {
    label: 'Workers', category: 'Rollouts', position: [385, 244], size: [199, 145],
    ports: { task: port('Rollout group job', 'left', 49) },
  },
  inference: {
    label: 'Inference', category: 'Stage', position: [798, 244], size: [183, 112],
    ports: {
      request: port('Model input', 'left', 32.5),
      output: port('Model output', 'left', 86),
      weights: port('Policy weights', 'bottom', 50),
    },
  },
  environment: {
    label: 'Task Environment', category: 'Stage', position: [231, 24], size: [188, 108],
    ports: { call: port('Task action', 'bottom', 16), observation: port('Observation / reward', 'bottom', 49) },
  },
  grading: {
    label: 'Grading', category: 'Stage', position: [641, 24], size: [182, 108],
    ports: { submission: port('Grading submission', 'bottom', 30), score: port('Grading outcome', 'bottom', 69) },
  },
  buffer: {
    label: 'Buffer', category: 'State', position: [385, 501], size: [199, 105],
    ports: {
      group: port('Completed group', 'top', 50),
      batch: port('Admitted training batch', 'right', 50),
      lease: port('Lease grant', 'left', 15),
      reject: port('Rejected group', 'left', 80),
    },
  },
  discarded: {
    label: 'Admission rejections', category: 'Sink', position: [24, 536], size: [194, 98],
    ports: { reject: port('Rejected group', 'right', 50) },
  },
  trainer: {
    label: 'Trainer', category: 'Stage', position: [798, 501], size: [183, 105],
    ports: { batch: port('Admitted training batch', 'left', 50), weights: port('Policy weights', 'top', 50) },
  },
};

// Settings use recorded values for the execution. Each node owns its own configuration fields.
export const nodeSettings = {
  source: { purpose: 'Offers eligible dataset entries to Dispatch.', fields: [['trainingRows', 'Training rows']] },
  rollouts: { purpose: 'Dispatches group jobs and runs the rollouts that form each group.',
    fields: [['batchPolicy', 'Batch scheduling'], ['outstandingLeaseLimit', 'Outstanding lease limit']] },
  dispatch: { purpose: 'Acquires a lease and selects an eligible dataset entry.',
    fields: [['batchPolicy', 'Batch scheduling'], ['outstandingLeaseLimit', 'Outstanding lease limit']] },
  workers: { purpose: 'Runs model and tool calls, then assembles and hands off rollout groups.',
    fields: [['outstandingLeaseLimit', 'Outstanding lease limit']] },
  inference: { purpose: 'Serves model requests from the rollout workers.',
    fields: [['inferenceGpus', 'Configured inference GPUs']] },
  environment: { purpose: 'Executes tool calls or task steps and supplies observations and rewards.' },
  grading: { purpose: 'Scores rollouts and reports training and evaluation outcomes.' },
  buffer: { purpose: 'Holds groups, applies admission rules, and assembles training batches.',
    fields: [['batchTarget', 'Groups per batch'], ['bufferCapacity', 'Max untrained groups'],
      ['maxStaleness', 'Staleness limit (policy steps)'], ['batchPolicy', 'Batch scheduling']] },
  discarded: { purpose: 'Collects groups rejected by batch admission.' },
  trainer: { purpose: 'Consumes training batches, updates the policy, and publishes its weights.',
    fields: [['strategy', 'Training strategy'], ['policyGpus', 'Configured trainer GPUs'],
      ['policyNodes', 'Trainer nodes'], ['policyGpusPerNode', 'Trainer GPUs per node'],
      ['policyTp', 'Tensor parallelism'], ['policyPp', 'Pipeline parallelism'],
      ['policyCp', 'Context parallelism'], ['policyEp', 'Expert parallelism']] },
};

const edge = (from, to, label, payload, { kind = 'work', lane } = {}) => ({ from, to, label, payload, kind, lane });

export const edges = {
  entries: edge(['source', 'entries'], ['rollouts', 'entries'], 'Dataset entries', 'Dataset entry'),
  dispatchJob: edge(['dispatch', 'task'], ['workers', 'task'], 'Rollout group job', 'Rollout group job'),
  promptTokens: edge(['rollouts', 'request'], ['inference', 'request'], 'Model input', 'Prompt and continuation tokens'),
  generation: edge(['inference', 'output'], ['rollouts', 'output'], 'Model output', 'Generated tokens'),
  toolCall: edge(['rollouts', 'call'], ['environment', 'call'], 'Task action', 'Tool call or task action', { lane: 214 }),
  observation: edge(['environment', 'observation'], ['rollouts', 'observation'], 'Observation / reward', 'Observation and any reward returned by the same task step', { lane: 164 }),
  gradeSubmit: edge(['rollouts', 'submission'], ['grading', 'submission'], 'Submission', 'Submission for a distinct grading call', { lane: 164 }),
  gradeResult: edge(['grading', 'score'], ['rollouts', 'score'], 'Grading outcome', 'Score or grading outcome', { lane: 214 }),
  completed: edge(['rollouts', 'group'], ['buffer', 'group'], 'Completed groups', 'Completed rollout group and grading outcome'),
  admitted: edge(['buffer', 'batch'], ['trainer', 'batch'], 'Admitted batch', 'Admitted training batch'),
  rejected: edge(['buffer', 'reject'], ['discarded', 'reject'], 'Rejected groups', 'Groups rejected at batch admission'),
  leases: edge(['buffer', 'lease'], ['rollouts', 'lease'], 'Lease grants', 'Buffer lease grant', { kind: 'coordination' }),
  weights: edge(['trainer', 'weights'], ['inference', 'weights'], 'Weight sync', 'Updated policy weights', { kind: 'coordination' }),
};

export function flowGraph(expanded, separateGrading) {
  const activeNodes = Object.entries(nodes)
    .filter(([id]) => (id !== 'grading' || separateGrading) &&
      (expanded ? id !== 'rollouts' : id !== 'dispatch' && id !== 'workers'))
    .map(([id, node]) => [id, id === 'workers' ? {
      ...node,
      ports: { ...Object.fromEntries(Object.entries(nodes.rollouts.ports).filter(([, p]) => p.child === 'workers')),
        ...node.ports },
    } : node]);
  const endpoint = ([id, handle]) => expanded && id === 'rollouts'
    ? [nodes.rollouts.ports[handle].child, handle]
    : [id, handle];
  const activeEdges = Object.entries(edges)
    .filter(([id]) => (expanded || id !== 'dispatchJob') &&
      (separateGrading || id !== 'gradeSubmit' && id !== 'gradeResult'))
    .map(([id, e]) => ({ ...e, id, from: endpoint(e.from), to: endpoint(e.to), card: id }));
  return { nodes: activeNodes, edges: activeEdges };
}

const metric = (subject, name, label, unit, transform, filter = {}, extra = {}) =>
  ({ subject, name, label, unit, transform, filter, time: 'timestamp_ms', ...extra });
const vllm = (source_temporality) => ({ metric_source: 'vllm', source_temporality });
const requestTiming = (phase, label) => metric('inference', 'request_' + phase + '_time_seconds_sum',
  label, 's', 'cumulativeMean', { ...vllm('cumulative_snapshot'), source_kind: 'histogram' },
  { countName: 'request_' + phase + '_time_seconds_count' });
const scalar = (subject, key, label, unit, extra = {}) =>
  metric(subject, 'training_metric_value', label, unit, 'points', { metric: key }, extra);
const work = (subject, kind, label, unit) =>
  metric(subject, 'work_completed', label, unit, 'points', { work_kind: kind });
const awaitTime = (subject, name, label) => metric(subject, 'rollout_wait_seconds', label, 's', 'points',
  { wait: name, stat: 'sum' }, { distribution: true });
const unrecorded = (subject, label, unit) =>
  metric(subject, null, label, unit, 'unavailable');

export const signals = {
  promptTokens: metric('promptTokens', 'prompt_tokens_total', 'Prompt tokens', 'tokens/s', 'cumulative', vllm('cumulative_snapshot')),
  generation: metric('generation', 'generation_tokens_total', 'Generation tokens', 'tokens/s', 'cumulative', vllm('cumulative_snapshot')),
  running: metric('inference', 'num_requests_running', 'Running', 'requests', 'engineSnapshots', vllm('current_snapshot')),
  waiting: metric('inference', 'num_requests_waiting', 'Waiting', 'requests', 'engineSnapshots', vllm('current_snapshot')),
  waitingCapacity: metric('inference', 'num_requests_waiting_by_reason', 'Waiting for capacity', 'requests', 'engineSnapshots',
    { ...vllm('current_snapshot'), reason: 'capacity' }),
  waitingDeferred: metric('inference', 'num_requests_waiting_by_reason', 'Deferred by scheduler', 'requests', 'engineSnapshots',
    { ...vllm('current_snapshot'), reason: 'deferred' }),
  kvCache: metric('inference', 'kv_cache_usage_perc', 'KV cache used', '%', 'engineSnapshots', vllm('current_snapshot'),
    { average: true, factor: 100 }),
  prefillTime: requestTiming('prefill', 'Mean prefill time'),
  decodeTime: requestTiming('decode', 'Mean decode time'),
  quarantined: scalar('source', 'async/quarantined_prompts', 'Quarantined entries', 'entries'),
  entryStarts: metric('entries', 'rollout_call', 'Recorded group starts', 'entries/s', 'eventRate', {}, { time: 'body_json.started_unix_ms' }),
  entryReleases: metric('buffer', 'rollout_groups', 'Dataset entry IDs released', 'entries/s', 'deltaRate',
    { disposition: 'consumed' }, { kind: 'scatter' }),
  dispatched: metric('dispatchJob', 'rollout_call', 'Recorded starts', 'jobs/s', 'eventRate', {}, { time: 'body_json.started_unix_ms' }),
  activeGroupJobs: metric('workers', 'rollout_call', 'Recorded active group jobs', 'jobs', 'concurrency'),
  groupJobEnds: metric('workers', 'rollout_call', 'Group job finishes', 'jobs/s', 'eventRate', {}, { time: 'body_json.finished_unix_ms', dimension: 'outcome' }),
  groupJobTime: metric('workers', 'rollout_call', 'Group job wall time', 's', 'spanDuration', {},
    { time: 'body_json.finished_unix_ms', distribution: true }),
  modelAwaitCount: metric('workers', 'rollout_waits', 'Model awaits per group job', 'awaits/job', 'points', { wait: 'model_client_await' }),
  rolloutsPerGroup: work('workers', 'rollout', 'Rollouts per recorded generated group', 'rollouts/group'),
  completed: metric('completed', 'rollout_call', 'Successful handoffs', 'groups/s', 'eventRate', { outcome: 'success' }, { time: 'body_json.finished_unix_ms' }),
  handoffTokens: metric('completed', 'rollout_call', 'Response tokens handed off', 'tokens/s', 'deltaRate',
    { outcome: 'success' }, { time: 'body_json.finished_unix_ms', value: 'body_json.response_tokens' }),
  calls: metric('toolCall', 'rollout_waits', 'Environment awaits', 'awaits/s', 'deltaRate', { wait: 'env_await' }, { facet: 'task_family_or_backend' }),
  environmentQueue: awaitTime('environment', 'env_queue', 'Environment queue time per group job'),
  environmentExecution: awaitTime('environment', 'env_exec', 'Environment execution time per group job'),
  environmentResume: awaitTime('environment', 'env_resume', 'Environment resume time per group job'),
  callWait: awaitTime('environment', 'env_await', 'Environment wait time per group job'),
  toolTasks: scalar('environment', 'generate/task_rollout/tool_tasks', 'Rollouts with tool calls', 'rollouts/batch'),
  callFailures: unrecorded('environment', 'Environment errors / timeouts', 'calls'),
  taskReturns: unrecorded('observation', 'Observation returns', 'returns/s'),
  gradingIn: unrecorded('gradeSubmit', 'Grading submissions', 'submissions/s'),
  gradingResults: unrecorded('gradeResult', 'Grading results returned', 'results/s'),
  gradingOut: scalar('grading', 'generate/outcome/graded', 'Graded training rollouts', 'rollouts/batch'),
  evalGraded: scalar('grading', 'eval/all/outcome/graded', 'Graded eval rollouts', 'rollouts/eval'),
  evalAttempted: scalar('grading', 'eval/all/num_attempted', 'Attempted eval rollouts', 'rollouts/eval'),
  evalFailed: scalar('grading', 'eval/all/outcome/submission_failure', 'Eval submissions that failed', 'rollouts/eval'),
  evalGrading: scalar('grading', 'eval/all/timing/grading/mean', 'Mean grading time · eval', 's'),
  evalGradingP50: scalar('grading', 'eval/all/timing/grading/p50', 'Median grading time · eval', 's'),
  evalGradingP95: scalar('grading', 'eval/all/timing/grading/p95', 'P95 grading time · eval', 's'),
  evalGradingFamily: metric('grading', 'training_metric_value', 'Mean grading time by eval suite', 's', 'points', {}, { metricPattern: '^eval/([^/]+)/timing/grading/mean$', facet: 'eval_suite' }),
  rolloutFailure: scalar('workers', 'generate/outcome/rollout_failure', 'Rollout failures', 'rollouts/batch'),
  submissionFailure: scalar('workers', 'generate/outcome/submission_failure', 'Submission failures', 'rollouts/batch'),
  rolloutInfra: scalar('workers', 'generate/outcome/infra_error', 'Infrastructure failures', 'rollouts/batch'),
  queue: metric('buffer', 'rollout_queue_depth', 'Ready queue depth', 'groups', 'points', { queue: 'rollout_buffer' }),
  dwell: metric('buffer', 'rollout_buffer_dwell_seconds', 'Group dwell time', 's', 'points', {}, { distribution: true }),
  admissions: metric('admitted', 'rollout_groups', 'Consumed groups', 'groups/s', 'deltaRate', { disposition: 'consumed' }),
  batchGroups: metric('admitted', 'rollout_groups', 'Groups consumed', 'groups/batch', 'batchTotal', { disposition: 'consumed' }),
  batchTokens: metric('admitted', 'rollout_group_tokens', 'Generated tokens in consumed groups', 'tokens/batch', 'batchTotal', { disposition: 'consumed' }),
  rejected: scalar('rejected', 'async/rejected_count', 'Rejected groups', 'groups/batch'),
  staleRejected: scalar('rejected', 'async/rejected_count/stale', 'Rejected for staleness', 'groups/batch'),
  lagMax: scalar('admitted', 'async/staleness_max', 'Oldest policy lag', 'policy steps'),
  lagMean: scalar('admitted', 'async/staleness_mean', 'Mean policy lag', 'policy steps'),
  rejectedByReason: metric('rejected', 'training_metric_value', 'Rejected groups by reason', 'groups/batch', 'points', {}, { metricPattern: '^async/rejected_count/(.+)$', facet: 'reason' }),
  openLeases: unrecorded('buffer', 'Open leases', 'leases'),
  grants: unrecorded('leases', 'Lease grants', 'grants/s'),
  leaseWait: awaitTime('dispatch', ['lease', 'slot'], 'Wait for a lease'),
  taskWait: awaitTime('dispatch', ['task', 'prompt'], 'Waiting to select an eligible task'),
  handoffWait: awaitTime('completed', 'enqueue', 'Buffer write time per group job'),
  modelAwait: awaitTime('workers', 'model_client_await', 'Sum of model waits per group job'),
  wait: metric('trainer', 'async_phase_window', 'Trainer batch wait', 's', 'duration', { phase: 'rollout_wait' },
    { time: 'body_json.finished_unix_ms', distribution: true }),
  waitTime: metric('trainer', 'async_phase_window', 'Waiting for a batch', 's per 10 s', 'waitTime'),
  training: metric('trainer', 'async_phase_window', 'Training duration', 's', 'duration', { phase: 'training' },
    { time: 'body_json.finished_unix_ms', distribution: true }),
  trainedSequences: work('trainer', 'consumed_sample', 'Sequences used', 'sequences/step'),
  trainedResponseTokens: work('trainer', 'consumed_response_token', 'Response tokens used', 'tokens/step'),
  trainedLossTokens: work('trainer', 'consumed_loss_token', 'Tokens contributing to loss', 'tokens/step'),
  optimizer: metric('trainer', 'work_completed', 'Optimizer completion', 'steps', 'event', { work_kind: 'consumed_sample' }),
  sync: metric('weights', 'weight_sync_completed', 'Weight sync duration', 's', 'duration', {}, { distribution: true }),
  batchTarget: metric('buffer', 'async_run_configuration', 'Configured batch size', 'groups', 'config', {}, { value: 'body_json.mini_batch_size' }),
};

const card = (title, series, help, extra = {}) => ({ title, series, help, ...extra });
const rejectionCard = (title) => card(title, ['rejected', 'rejectedByReason'],
  'Each point counts groups the Buffer rejected while filling one batch. Choose a reason to see its count.',
  { rejectionSummary: true, groupLabel: 'Rejection reason' });
const batchFailures = ['rolloutFailure', 'submissionFailure', 'rolloutInfra'];
export const cards = {
  source: card('Task source', ['quarantined'], 'Each point counts entries that the selector has quarantined.', {
    links: [{ id: 'entries', label: 'View dataset entries' }],
  }),
  rollouts: card('Rollouts', ['activeGroupJobs', 'groupJobTime', 'leaseWait', 'taskWait', 'groupJobEnds', ...batchFailures],
    'Each job runs one rollout group. Active counts use start and end times recorded when jobs finish. Batch failures count individual rollouts.', {
      tabs: { 'rollouts/batch': 'Batch failures' },
      links: [{ id: 'dispatch', label: 'View Dispatch' }, { id: 'workers', label: 'View Workers' }],
    }),
  dispatch: card('Dispatch', ['leaseWait', 'taskWait', 'leaseWaitTime', 'taskWaitTime'],
    'Dispatch waits for a lease, then for an eligible entry in the lease’s family, when specified. A point appears when a wait ends. Waiting per 10 s estimates how much of each interval Dispatch spent waiting.', {
      entryOccupancySummary: true,
      links: [{ id: 'leases', label: 'View lease grants' }, { id: 'entries', label: 'View dataset entries' }],
    }),
  workers: card('Workers', ['activeGroupJobs', 'groupJobTime', 'modelAwait', 'rolloutsPerGroup', 'modelAwaitCount',
    'groupJobEnds', ...batchFailures], 'Each job runs one rollout group. Active counts use jobs with recorded start and end times. Model wait time adds the waits for calls within each job. Those waits can overlap.', {
      cancellationSummary: true,
      tabs: { 'rollouts/batch': 'Batch failures' },
      summaries: { rolloutsPerGroup: 'generated group', modelAwaitCount: 'group job' },
    }),
  inference: card('Inference', ['running', 'waiting', 'prefillTime', 'decodeTime', 'kvCache'],
    'Running and waiting count requests inside vLLM. Each 10-second point adds the latest counts from the engines.', {
      engines: true, tabs: { s: 'Prefill and decode', '%': 'KV cache' },
      helpByUnit: {
        s: 'Prefill processes the input. Decode generates new tokens. Each 10-second point estimates the mean time per finished request. Gaps mark intervals with no finishes or missing samples.',
        '%': 'The KV cache stores attention state. The fleet chart averages the percentage used across engines.',
      },
      links: [{ id: 'promptTokens', label: 'View model input' }, { id: 'generation', label: 'View model output' }],
    }),
  environment: card('Task Environment', ['toolTasks', 'callWait', 'environmentQueue', 'environmentExecution', 'environmentResume', 'callFailures'],
    'A batch point counts rollouts that made at least one tool call. Timing points show totals per group job, including time spent in the queue and running calls.', {
      summaries: { toolTasks: 'batch' }, tabs: { 'rollouts/batch': 'Tool use per batch' },
      missing: 'Some environment timing or failure records are missing in this window.',
    }),
  grading: card('Grading', ['evalGraded', 'evalGrading', 'evalGradingP50', 'evalGradingP95', 'evalGradingFamily', 'gradingOut'],
    'Eval grades count graded rollouts across suites. In Timing, choose a suite to see its mean grading time. Training grades count graded rollouts per batch.', {
    evalSummary: true, groupLabel: 'Eval suite', groupDefault: 'All suites',
    summaries: { evalGraded: 'evaluation', evalGrading: 'evaluation', evalGradingFamily: 'evaluation' },
    tabs: { 'rollouts/batch': 'Training grades' },
  }),
  buffer: card('Buffer', ['queue', 'dwell', 'openLeases'],
    'Queue depth counts groups ready for admission. Dwell time measures how long a group stayed in the Buffer, from when it entered to when it left.',
    {
      links: [{ id: 'admitted', label: 'View admitted groups and policy lag' }],
    }),
  discarded: rejectionCard(nodes.discarded.label),
  rejected: rejectionCard('Rejected groups'),
  trainer: card('Trainer', ['training', 'wait', 'waitTime', 'trainedResponseTokens', 'trainedLossTokens', 'trainedSequences'],
    'The trainer records how long each step spent waiting for a batch and training. Counts come from completed optimizer steps and exclude padding added for data parallelism.', {
      batchWaitZoom: true,
    }),
  entries: card('Dataset entries', ['entryStarts'], 'Starts count group jobs that later recorded an end. Releases count entry IDs freed when their groups leave for training. Reuse still depends on the task family and quarantine.', {
    context: ['entryReleases'], tabs: { 'entries/s': 'Releases and starts' },
    links: [{ id: 'dispatch', label: 'View task selection waits in Dispatch' }],
  }),
  dispatchJob: card('Rollout group job', ['dispatched'], 'This chart counts group jobs at their start times once they record an end.'),
  promptTokens: card('Model input', ['promptTokens', 'promptTokensPerGpu'],
    'vLLM counts prompt and continuation tokens it handles, including tokens served from its prefix cache. The default rate adds input across engines in 10-second intervals.',
    { engines: true, tabs: { 'tokens/s': 'Tokens/s', 'tokens/s/GPU': 'Per GPU' } }),
  generation: card('Model output', ['generation', 'generationPerGpu'],
    'vLLM counts new tokens generated during decode. The default rate adds output across engines in 10-second intervals.',
    { engines: true, tabs: { 'tokens/s': 'Tokens/s', 'tokens/s/GPU': 'Per GPU' } }),
  toolCall: card('Task action', ['calls'], 'This chart counts calls the worker waited for in the task environment. Choose a task family or backend when those labels are available.', {
    gapLabel: 'No call counts', gap: 'Finelog has no environment await counts in this window. Task Environment reports batch tool use when available.',
    related: { id: 'environment', label: 'See batch tool use' },
  }),
  observation: card('Observation / reward', ['taskReturns'], 'The task environment returns an observation and may include a reward.', {
    gapLabel: 'No return records',
    gap: 'Finelog has no individual observation or reward returns in this window. The batch summary counts rollouts that used tools.',
    related: { id: 'environment', label: 'See batch tool use' },
  }),
  gradeSubmit: card('Grading submission', ['gradingIn'], 'Workers send rollouts to a separate grader for scoring.', {
    gapLabel: 'No submission counts',
    gap: 'Finelog has no count of submissions sent to the grader. The eval summaries count attempted rollouts and grading outcomes.',
    related: { id: 'grading', label: 'See eval summaries' },
  }),
  gradeResult: card('Grading outcome', ['gradingResults'], 'The grader returns a score or verdict to the worker.', {
    gapLabel: 'No result events', gap: 'Finelog has no individual grading returns in this window. Grading reports aggregate eval and training outcomes.',
    related: { id: 'grading', label: 'See grading summaries' },
  }),
  completed: card('Completed groups', ['completed', 'handoffTokens', 'handoffWait'],
    'Each successful handoff records one group and its response tokens when the job ends. Buffer write time measures how long the worker waited to enqueue the group.'),
  admitted: card('Admitted batch', ['batchGroups', 'batchTokens', 'admissions', 'lagMax', 'lagMean'],
    'Each batch point counts the groups and generated tokens sent to the trainer. Policy lag is how many updates old those groups were at admission.', {
      tabs: { 'tokens/batch': 'Tokens per batch' }, batchSummary: true, lagCaption: true,
      links: [{ id: 'trainer', label: 'View batch waits in Trainer' }],
    }),
  leases: card('Lease grants', ['grants'], 'A lease reserves a slot for one group. Dispatch shows how long it waited for a lease.', {
    gapLabel: 'No grant events',
    gap: 'Finelog has no lease grant events in this window. Dispatch records how long it waited for a lease.',
    links: [{ id: 'dispatch', label: 'View lease waits in Dispatch' }, { id: 'entries', label: 'See entry releases and starts' }],
  }),
  weights: card('Weight sync', ['sync'], 'Each point shows how long a weight sync took. Select a point to see its policy step and reason when recorded.', { tabs: { s: 'Sync duration' } }),
};

export const MAX_WINDOW_MINUTES = 60;
export const MIN_WINDOW_MS = 10_000;
export const SPAN_LOOKAHEAD_MS = 30 * 60_000;
