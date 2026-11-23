// Graph, Signal, and Card are kept together so a handle or chart can be traced to Finelog.
const port = (label, side, offset, child) => ({ label, side, offset, child });

// The four-stage training loop forms a rectangle. The single-neighbor
// stages sit above it, with a separate pair of lanes for each worker handoff.
export const nodes = {
  source: {
    label: 'Task source', category: 'Input', position: [24, 24], size: [174, 100],
    ports: { prompts: port('Prompt work', 'bottom', 50) },
  },
  rollouts: {
    label: 'Rollouts', category: 'Stage', position: [385, 244], size: [199, 145],
    ports: {
      prompts: port('Prompt work', 'left', 24, 'dispatch'),
      lease: port('Lease grant', 'left', 76, 'dispatch'),
      request: port('Prompt or continuation', 'right', 25, 'workers'),
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
      prompts: port('Prompt work', 'top', 50),
      lease: port('Lease grant', 'bottom', 50),
      task: port('Leased task', 'right', 49),
    },
  },
  workers: {
    label: 'Workers', category: 'Rollouts', position: [385, 244], size: [199, 145],
    ports: { task: port('Leased task', 'left', 49) },
  },
  inference: {
    label: 'Inference', category: 'Stage', position: [798, 244], size: [183, 112],
    ports: {
      request: port('Prompt or continuation', 'left', 32.5),
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
  source: { purpose: 'Offers eligible training prompts to Dispatch.', fields: [['trainingRows', 'Training rows']] },
  rollouts: { purpose: 'Dispatches prompts and runs the rollouts that form each training group.',
    fields: [['batchPolicy', 'Batch scheduling'], ['maxConcurrentRollouts', 'Concurrent rollout limit']] },
  dispatch: { purpose: 'Acquires a lease and finds a prompt that can use it.',
    fields: [['batchPolicy', 'Batch scheduling'], ['maxConcurrentRollouts', 'Concurrent rollout limit']] },
  workers: { purpose: 'Runs model and tool calls, then assembles and hands off rollout groups.',
    fields: [['maxConcurrentRollouts', 'Concurrent rollout limit']] },
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
  prompts: edge(['source', 'prompts'], ['rollouts', 'prompts'], 'Prompt work', 'Prompt work item'),
  dispatchTask: edge(['dispatch', 'task'], ['workers', 'task'], 'Leased task', 'Leased rollout task'),
  promptTokens: edge(['rollouts', 'request'], ['inference', 'request'], 'Prompt / continuation', 'Prompt or continuation tokens'),
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
    .filter(([id]) => (expanded || id !== 'dispatchTask') &&
      (separateGrading || id !== 'gradeSubmit' && id !== 'gradeResult'))
    .map(([id, e]) => ({ ...e, id, from: endpoint(e.from), to: endpoint(e.to), card: id }));
  return { nodes: activeNodes, edges: activeEdges };
}

const metric = (subject, name, label, unit, transform, filter = {}, extra = {}) =>
  ({ subject, name, label, unit, transform, filter, time: 'timestamp_ms', ...extra });
const vllm = (source_temporality) => ({ metric_source: 'vllm', source_temporality });
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
  quarantined: scalar('source', 'async/quarantined_prompts', 'Quarantined prompts', 'prompts'),
  promptItems: metric('prompts', 'rollout_call', 'Recorded prompt starts', 'prompts/s', 'eventRate', {}, { time: 'body_json.started_unix_ms' }),
  promptReleases: metric('buffer', 'rollout_groups', 'Prompt IDs released', 'prompts/s', 'deltaRate',
    { disposition: 'consumed' }, { kind: 'scatter' }),
  dispatched: metric('dispatchTask', 'rollout_call', 'Recorded starts', 'attempts/s', 'eventRate', {}, { time: 'body_json.started_unix_ms' }),
  activeAttempts: metric('workers', 'rollout_call', 'Recorded active attempts', 'attempts', 'concurrency'),
  attemptEnds: metric('workers', 'rollout_call', 'Attempt finishes', 'attempts/s', 'eventRate', {}, { time: 'body_json.finished_unix_ms', dimension: 'outcome' }),
  attemptTime: metric('workers', 'rollout_call', 'Group attempt wall time', 's', 'spanDuration', {},
    { time: 'body_json.finished_unix_ms', distribution: true }),
  modelAwaitCount: metric('workers', 'rollout_waits', 'Model awaits per attempt', 'awaits/attempt', 'points', { wait: 'model_client_await' }),
  rolloutsPerGroup: work('workers', 'rollout', 'Rollouts per recorded generated group', 'rollouts/group'),
  completed: metric('completed', 'rollout_call', 'Successful handoffs', 'groups/s', 'eventRate', { outcome: 'success' }, { time: 'body_json.finished_unix_ms' }),
  handoffTokens: metric('completed', 'rollout_call', 'Response tokens handed off', 'tokens/s', 'deltaRate',
    { outcome: 'success' }, { time: 'body_json.finished_unix_ms', value: 'body_json.response_tokens' }),
  calls: metric('toolCall', 'rollout_waits', 'Environment awaits', 'awaits/s', 'deltaRate', { wait: 'env_await' }, { facet: 'task_family_or_backend' }),
  environmentQueue: awaitTime('environment', 'env_queue', 'Environment queue time per attempt'),
  environmentExecution: awaitTime('environment', 'env_exec', 'Environment execution time per attempt'),
  environmentResume: awaitTime('environment', 'env_resume', 'Environment resume time per attempt'),
  callWait: awaitTime('environment', 'env_await', 'Environment wait time per attempt'),
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
  slotWait: awaitTime('dispatch', 'slot', 'Wait for a lease'),
  promptWait: awaitTime('dispatch', 'prompt', 'Wait for a prompt'),
  handoffWait: awaitTime('completed', 'enqueue', 'Buffer write time per attempt'),
  modelAwait: awaitTime('workers', 'model_client_await', 'Sum of model waits per attempt'),
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
  'These counts come from batch admission. Each point is the number of groups rejected while filling that batch. Choose a reason to see its count.',
  { rejectionSummary: true, groupLabel: 'Rejection reason' });
const batchFailures = ['rolloutFailure', 'submissionFailure', 'rolloutInfra'];
export const cards = {
  source: card('Task source', ['quarantined'], 'The loader reports how many prompts are quarantined at each step.', {
    links: [{ id: 'prompts', label: 'View prompt work' }],
  }),
  rollouts: card('Rollouts', ['activeAttempts', 'attemptTime', 'slotWait', 'promptWait', 'attemptEnds', ...batchFailures],
    'Active attempts are reconstructed from recorded calls. Each attempt can produce several sampled rollouts. Batch failures count individual rollouts.', {
      tabs: { 'rollouts/batch': 'Batch failures' },
      links: [{ id: 'dispatch', label: 'View Dispatch' }, { id: 'workers', label: 'View Workers' }],
    }),
  dispatch: card('Dispatch', ['slotWait', 'promptWait', 'slotWaitTime', 'promptWaitTime'],
    'Dispatch waits for a lease, then an eligible prompt in that lease’s task family. Wait durations arrive when the await ends. Waiting per 10 s estimates when Dispatch was blocked.', {
      promptOccupancySummary: true,
      links: [{ id: 'leases', label: 'View lease grants' }, { id: 'prompts', label: 'View prompt work' }],
    }),
  workers: card('Workers', ['activeAttempts', 'attemptTime', 'modelAwait', 'rolloutsPerGroup', 'modelAwaitCount',
    'attemptEnds', ...batchFailures], 'An attempt processes one group. A group contains sampled rollouts, each of which can make multiple model calls. Active attempts use intervals reconstructed from completed records. Model waits are summed within each attempt and can overlap.', {
      cancellationSummary: true,
      tabs: { 'rollouts/batch': 'Batch failures' },
      summaries: { rolloutsPerGroup: 'generated group', modelAwaitCount: 'attempt' },
    }),
  inference: card('Inference', ['running', 'waiting', 'kvCache'],
    'Request totals sum the last snapshot per engine in each bin. Fleet KV cache usage averages those engine percentages.', {
      engines: true, tabs: { '%': 'KV cache' },
      links: [{ id: 'promptTokens', label: 'View prompt tokens' }, { id: 'generation', label: 'View model output' }],
    }),
  environment: card('Task Environment', ['toolTasks', 'callWait', 'environmentQueue', 'environmentExecution', 'environmentResume', 'callFailures'],
    'A batch point counts rollouts with at least one tool call. Timing samples sum observed waits within each attempt.', {
      summaries: { toolTasks: 'batch' }, tabs: { 'rollouts/batch': 'Tool use per batch' },
      missing: 'Some environment timing or failure records are missing in this window.',
    }),
  grading: card('Grading', ['evalGraded', 'evalGrading', 'evalGradingP50', 'evalGradingP95', 'evalGradingFamily', 'gradingOut'],
    'Eval grades and timing use Marin’s summaries across suites. Training grades come from training-batch summaries.', {
    evalSummary: true, groupLabel: 'Eval suite', groupDefault: 'All suites',
    summaries: { evalGraded: 'evaluation', evalGrading: 'evaluation', evalGradingFamily: 'evaluation' },
    tabs: { 'rollouts/batch': 'Training grades' },
  }),
  buffer: card('Buffer', ['queue', 'dwell', 'openLeases'],
    'Queue depth counts ready groups from any batch. Dwell time is recorded for each departing group with a known commit time.', {
      links: [{ id: 'admitted', label: 'View admitted groups and policy lag' }],
    }),
  discarded: rejectionCard(nodes.discarded.label),
  rejected: rejectionCard('Rejected groups'),
  trainer: card('Trainer', ['training', 'wait', 'waitTime', 'trainedResponseTokens', 'trainedLossTokens', 'trainedSequences'],
    'Training and batch-wait durations come from the trainer’s phases. Work counts are recorded after successful optimizer steps and exclude data-parallel padding.', {
      batchWaitZoom: true,
    }),
  prompts: card('Prompt work', ['promptItems'], 'Starts count attempts with recorded start and end times. The release overlay counts consumed groups that free a prompt ID. Lease family and quarantine rules govern which IDs can be reused.', {
    context: ['promptReleases'], tabs: { 'prompts/s': 'Releases and starts' },
    links: [{ id: 'dispatch', label: 'View prompt waits in Dispatch' }],
  }),
  dispatchTask: card('Leased task', ['dispatched'], 'Starts counted by their recorded start times, for attempts that published an end record.'),
  promptTokens: card('Prompt / continuation', ['promptTokensPerGpu', 'promptTokens'], 'vLLM counts all prompt tokens, including cache hits. Each 10 s bin sums rates across observed engines with full coverage.', { engines: true }),
  generation: card('Model output', ['generationPerGpu', 'generation'], 'vLLM counts generated tokens. Gaps mark missing coverage or counter resets.', { engines: true }),
  toolCall: card('Task action', ['calls'], 'The instrumented environment wrapper counts awaits. That count can be broken down by backend when the record carries a backend label.', {
    gapLabel: 'No call counts', gap: 'Finelog has no environment await counts in this window. Task Environment reports batch tool use when available.',
    related: { id: 'environment', label: 'See batch tool use' },
  }),
  observation: card('Observation / reward', ['taskReturns'], 'A task step can return an observation and a reward together.', {
    gapLabel: 'No return records',
    gap: 'Finelog has no individual observation or reward returns in this window. The batch summary counts rollouts that used tools.',
    related: { id: 'environment', label: 'See batch tool use' },
  }),
  gradeSubmit: card('Grading submission', ['gradingIn'], 'This stream carries work sent to a separate grader.', {
    gapLabel: 'No submission counts',
    gap: 'Finelog has no count of submissions sent to the grader. The eval summaries count attempted rollouts and grading outcomes.',
    related: { id: 'grading', label: 'See eval summaries' },
  }),
  gradeResult: card('Grading outcome', ['gradingResults'], 'A grading result is a score or verdict returned to a rollout worker.', {
    gapLabel: 'No result events', gap: 'Finelog has no individual grading returns in this window. Grading reports aggregate eval and training outcomes.',
    related: { id: 'grading', label: 'See grading summaries' },
  }),
  completed: card('Completed groups', ['completed', 'handoffTokens', 'handoffWait'],
    'The driver records successful handoffs and response tokens when the worker call ends. Buffer write time measures the worker’s await of the writer.'),
  admitted: card('Admitted batch', ['batchGroups', 'batchTokens', 'admissions', 'lagMax', 'lagMean'],
    'Batch volume sums consumed groups and their generated tokens. Policy lag measures the policy age of consumed groups.', {
      tabs: { 'tokens/batch': 'Tokens per batch' }, batchSummary: true, lagCaption: true,
      links: [{ id: 'trainer', label: 'View batch waits in Trainer' }],
    }),
  leases: card('Lease grants', ['grants'], 'A lease gives Dispatch permission to launch a rollout.', {
    gapLabel: 'No grant events',
    gap: 'Finelog has no lease grant events in this window. Dispatch records how long it waited for a lease.',
    links: [{ id: 'dispatch', label: 'View lease waits in Dispatch' }, { id: 'prompts', label: 'See prompt releases and starts' }],
  }),
  weights: card('Weight sync', ['sync'], 'Duration from each recorded weight-sync completion. Click a point for its policy step and reason.', { tabs: { s: 'Sync duration' } }),
};

export const MAX_WINDOW_MINUTES = 60;
export const SPAN_LOOKAHEAD_MS = 30 * 60_000;
