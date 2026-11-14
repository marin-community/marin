/* Copyright The Marin Authors. SPDX-License-Identifier: Apache-2.0 */
const $ = id => document.getElementById(id);
function node(tag, value, className) {
  const item = document.createElement(tag);
  if (value !== undefined) item.textContent = value;
  if (className) item.className = className;
  return item;
}
async function json(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Request failed (${response.status})`);
  return response.json();
}

// Every stored assessment row is listed; its full packet loads when opened.
function taskReviews(sourceId, samples) {
  const section = node('section', undefined, 'task-reviews');
  for (const cohort of ['admitted', 'excluded']) {
    const group = node('section');
    group.append(node('h3', cohort === 'admitted' ? 'Retained task reviews' : 'Excluded task reviews'));
    for (const sample of samples.filter(item => item.cohort === cohort)) {
      const details = node('details', undefined, 'assessment-card'), content = node('div');
      details.dataset.taskId = sample.task_id;
      details.append(node('summary', `${sample.verdict} · ${sample.task_id}`), content);
      details.addEventListener('toggle', async () => {
        if (!details.open || content.childNodes.length) return;
        content.append(node('p', 'Loading full review and packet…'));
        try {
          const data = await json(`api/evidence?${new URLSearchParams({source_id: sourceId, task_id: sample.task_id})}`);
          content.replaceChildren(node('p', data.judgment.explanation), repairRecommendations(data), node('pre', JSON.stringify(data, null, 2)));
        } catch (error) { content.replaceChildren(node('p', error.message, 'error')); }
      });
      group.append(details);
    }
    section.append(group);
  }
  return section;
}

// Packets and archived collections may carry recommendations at different levels.
function repairRecommendations(record) {
  const section = node('section', undefined, 'repair-recommendations');
  function visit(value) {
    if (!value || typeof value !== 'object') return;
    for (const [key, item] of Object.entries(value)) {
      if (key === 'repair_recommendation' && item) {
        section.append(node('h3', 'Repair recommendation'), node('p', typeof item === 'string' ? item : JSON.stringify(item)));
      } else visit(item);
    }
  }
  visit(record);
  return section;
}

function daytonaLabel(daytona) {
  if (daytona.exempt) return '—';
  if (daytona.state === 'na') return 'N/A';
  if (daytona.state === 'unmeasured') return 'Pending census';
  if (daytona.state === 'safe') return 'SAFE';
  return `UNSAFE: ${(daytona.unsafe_reasons || []).join(', ') || 'execution requirement'}`;
}

function currentDataset(row) { return row.current_release; }
function datasetLinks(row) {
  const current = currentDataset(row), links = node('div', undefined, 'dataset-links');
  const sameInput = current && (row.url === current.url ||
    (row.dataset_id === current.repo && row.dataset_revision === current.revision));
  function append(label, url, title) {
    const anchor = node('a', label); anchor.href = url; anchor.title = title;
    anchor.target = '_blank'; anchor.rel = 'noopener'; links.append(anchor);
  }
  if (row.url?.startsWith('https://') && !sameInput) {
    links.append(node('span', 'Upstream:'));
    append('Original dataset ↗', row.url, `Upstream original: ${row.dataset_id || row.url}`);
  }
  if (current) append('Current dataset ↗', current.url, `Pinned training release: ${current.repo} @ ${current.revision}`);
  return links.childNodes.length ? links : null;
}

// Progress is a frozen observation; only the declaration's completed QA can set FULL or Result.
function boundProgress(source, measurement) {
  const progress = measurement.sources.find(item => item.source_id === source.id);
  if (!progress || progress.dataset_id !== source.dataset_id || progress.dataset_revision !== source.dataset_revision ||
      progress.retained_count !== source.within_source_dedup.unique_count) return null;
  return {...progress, as_of: measurement.as_of};
}

const contaminationClasses = {
  'NO ISSUES': 'quality-good',
  REMEDIATED: 'contamination-remediated',
  'MEASURED, EVAL': 'quality-some_issues',
  'REMEDIATION REQUIRED': 'quality-bad contamination-required',
  'NOT MEASURED': 'quality-unassessed',
};
function contaminationChip(status) {
  const chip = node('span', status, `chip contamination-chip ${contaminationClasses[status]}`);
  chip.dataset.contamination = status;
  return chip;
}

const backendNames = {local: 'LOCAL', daytona: 'Daytona', docker: 'Docker', gvisor: 'gVisor', qemu: 'QEMU', shellsim: 'ShellSim', iris: 'Iris'};
const supportClasses = {supported: 'good', partial: 'some_issues', unsupported: 'bad', not_applicable: 'unassessed', role_absent: 'unassessed'};
const backendStatusLabels = {not_applicable: 'N/A', role_absent: 'N/A (no role)'};
function readableRecord(value) {
  if (value instanceof Node) return value;
  if (value === null || value === undefined) return node('span', 'Not measured', 'muted');
  if (typeof value !== 'object') return node('span', typeof value === 'number' ? new Intl.NumberFormat('en-US').format(value) : String(value));
  const list = node('ul', undefined, 'record-list');
  for (const [key, item] of Object.entries(value)) {
    const entry = node('li');
    if (!Array.isArray(value)) entry.append(node('strong', key.replaceAll('_', ' ') + ': '));
    entry.append(readableRecord(item)); list.append(entry);
  }
  return list;
}
function backendLabels(source) {
  const capabilities = source.backend_capabilities;
  if (!capabilities || capabilities.basis === 'unassessed') return [];
  return Object.entries(capabilities.backends).map(([key, fit]) => {
    const reasons = [...new Set(Object.keys(fit.blockers).map(reason => reason.split(':').slice(1).join(':') || reason))];
    if (!reasons.length && Object.keys(fit.validation_errors).length) reasons.push('validation');
    const name = backendNames[key] || key;
    const status = backendStatusLabels[fit.support] || fit.support.toUpperCase();
    return {key, fit, label: fit.support === 'supported' ? name : `${name}: ${status}${reasons.length ? ': ' + reasons.join(', ') : ''}`};
  });
}
function backendLink(sourceId) {
  const anchor = node('a', 'Backends ›', 'evidence-link backends-link');
  anchor.href = `backends.html#${new URLSearchParams({source_id: sourceId})}`;
  anchor.setAttribute('aria-label', `Backend evidence for ${sourceId}`);
  return anchor;
}
function gradingChips(source) {
  const group = node('span', undefined, 'enum-chips grading-chips'), census = source.grading_environment;
  if (!census) { group.append(node('span', 'NOT MEASURED', 'chip quality-unassessed')); return group; }
  const problems = Object.entries(census.status_counts).filter(([status, count]) => status !== 'matched' && count > 0).map(([status]) => status.toUpperCase());
  const mismatch = problems.length > 0;
  for (const [boundary, count] of Object.entries(census.required_counts)) {
    if (!count) continue;
    const label = boundary === 'none' ? 'N/A' : boundary.toUpperCase();
    const chip = node('span', label + (mismatch ? ': ' + problems.join(', ') : ''), `chip quality-${mismatch ? 'some_issues' : 'good'}`);
    chip.title = `${count.toLocaleString('en-US')} tasks require this grading boundary${mismatch ? '; declaration mismatches are recorded in Evidence' : ''}`;
    group.append(chip);
  }
  if (!group.children.length) group.append(node('span', 'NOT MEASURED', 'chip quality-unassessed'));
  return group;
}
function backendDetails(source) {
  const wrap = node('div', undefined, 'table-wrap'); wrap.tabIndex = 0; wrap.setAttribute('aria-label', 'Backend fit by execution role');
  const table = node('table', undefined, 'backend-table'), head = node('thead'), headings = node('tr');
  for (const title of ['Backend', 'Overall', 'Agent', 'Grader', 'Not applicable', 'Blockers', 'Validation errors']) headings.append(node('th', title));
  head.append(headings); table.append(head); const body = node('tbody');
  for (const {key, fit} of backendLabels(source)) {
    const row = node('tr'); row.append(node('td', backendNames[key] || key));
    const status = node('td'); status.append(node('span', backendStatusLabels[fit.support] || fit.support.toUpperCase(), `chip quality-${supportClasses[fit.support]}`)); row.append(status);
    for (const role of ['agent', 'grader']) {
      const cell = node('td');
      const counts = Object.fromEntries(Object.entries(fit.roles[role] || {}).filter(([, count]) => count > 0).map(([state, count]) => [backendStatusLabels[state] || state, count]));
      cell.append(Object.keys(counts).length ? readableRecord(counts) : node('span', 'Not measured', 'muted')); row.append(cell);
    }
    for (const counts of [fit.not_applicable, fit.blockers, fit.validation_errors]) {
      const cell = node('td');
      cell.append(counts === undefined ? node('span', 'Not measured', 'muted') : Object.keys(counts).length ? readableRecord(counts) : node('span', 'None measured')); row.append(cell);
    }
    body.append(row);
  }
  if (!body.children.length) return node('p', source.backend_capabilities?.reason || 'Backend requirements have not been measured.', 'muted');
  table.append(body); wrap.append(table); return wrap;
}
function filtrationChip(filtration) {
  const label = {excluded: 'EXCLUDED', pending: 'IN PROGRESS', pending_oracle: 'PENDING ORACLE', pending_static: 'PENDING STATIC', static_retained: 'NO ORACLE', static_blocked: 'STATIC BLOCKED', eval_only: 'EVAL ONLY', complete: 'COMPLETE'}[filtration.state];
  const chip = node('span', label, `chip quality-${['complete', 'static_retained'].includes(filtration.state) ? 'good' : 'unassessed'}`);
  chip.title = `${filtration.reason} Current source state: ${filtration.source_status}.`;
  return chip;
}
function filtrationRows(source) {
  const lexical = source.lexical_contamination, dedup = source.within_source_dedup, oracle = source.oracle_inventory;
  const output = source.oracle_filtration, admission = source.filtration, counts = admission?.count_evidence;
  const measured = ['NO ISSUES', 'REMEDIATED', 'MEASURED, EVAL'].includes(source.contamination_status);
  const receipt = measured ? lexical?.receipt : null;
  const pendingRemediation = source.contamination_status === 'REMEDIATION REQUIRED';
  const zeroDrops = (source.contamination_status === 'NO ISSUES' && lexical?.contaminated_tasks === 0) || source.contamination_status === 'MEASURED, EVAL';
  return [
    ['Filtration', admission && filtrationChip(admission)],
    ['Current admission state', {excluded: 'Excluded', static_retained: 'Retained under no-oracle policy', static_blocked: 'No tasks remain after typed static exclusions', eval_only: 'Measured only (EVAL)', pending_oracle: 'Pending oracle filtration'}[admission?.state] || 'Pending qualified current-pin receipt'],
    ['Recorded source status', source.status],
    ['Raw converted input measured for retention', measured ? lexical?.measured_tasks : dedup?.raw_count],
    ['Original population before oracle filtration', output?.original_count],
    ['Original oracle-bearing tasks', counts?.oracle_tasks ?? oracle?.oracle_tasks ?? 'Not measured'],
    ['Oracle-failing tasks removed', output ? Object.entries(output.exclusion_counts).filter(([reason]) => reason === 'oracle_failed').reduce((sum, [, count]) => sum + count, 0) : oracle?.oracle_tasks === 0 ? 'N/A: complete census found no shipped oracle solutions' : 'Pending: no complete population-wide oracle-filter receipt is recorded'],
    ['Filtration exclusions by typed reason', output?.exclusion_counts],
    ['Post-oracle filtered population', output?.admitted_count],
    ['Within-source duplicates removed', receipt?.duplicate_dropped_tasks ?? dedup?.within_source_duplicates],
    ['Lexically contaminated tasks removed', pendingRemediation ? `Pending: remediation for ${lexical.contaminated_tasks} measured overlapping input tasks is not verified at the current pin` : receipt?.dropped_tasks ?? (zeroDrops ? 0 : 'Not measured')],
    [output ? 'Current filtered population' : 'Pre-filtration retained converted population', dedup?.unique_count],
    ['Typed static blocker tasks excluded (unique)', counts?.static_blocked_tasks],
    ['Static blocker reason counts (may overlap)', counts?.static_reason_counts],
    ['Measured EVAL examples (not admitted TRAIN)', admission?.state === 'eval_only' ? admission.example_count : null],
    ['Unexecutable on production backend dropped (by typed reason)', counts?.oracle_tasks === 0 ? 'Per-task executability run is not required under the pinned no-oracle policy; static exclusions remain applied' : admission?.state === 'excluded' ? 'No retained tasks; typed exclusions are recorded above' : 'Pending: no pin-bound per-task production-executability receipt is recorded'],
    ['Admitted after all filtration', admission?.admitted_count ?? 'Pending: full-population filtration and production executability are not established'],
    ...['policy', 'oracle_inventory', 'static_audit'].map(key => {
      const evidence = counts?.[key];
      const label = {policy: 'No-oracle count policy ruling', oracle_inventory: 'Current-input oracle census', static_audit: 'Current-input static blocker audit'}[key];
      if (!evidence) return [label, null];
      const span = node('span'); span.append(receiptLink(evidence.uri, label + ' ↗'), ' · SHA-256 ', node('code', evidence.sha256));
      return [label, span];
    }),
    ...['oracle_filter', 'tasks_parquet', 'validation'].map(key => {
      const evidence = output?.[key];
      const label = {oracle_filter: 'Canonical oracle filter receipt', tasks_parquet: 'Read-back private output', validation: 'Filtration validation receipt'}[key];
      if (!evidence) return [label, null];
      const span = node('span'); span.append(receiptLink(evidence.uri, label + ' ↗'), ' · SHA-256 ', node('code', evidence.sha256));
      return [label, span];
    }),
  ];
}

function boundOracleInventory(source, measurement) {
  const record = measurement.sources.find(item => item.source_id === source.id);
  const input = source.actor_census_input;
  if (!record || !input || record.input.measured_tasks !== source.within_source_dedup.unique_count ||
      Object.keys(record.input).length !== Object.keys(input).length || !Object.entries(record.input).every(([key, value]) => input[key] === value)) return null;
  return record;
}
function receiptLink(uri, label) {
  const anchor = node('a', label); anchor.href = /^(s3|gs):/.test(uri) ? `https://filer.marina.oa.dev/#${encodeURIComponent(uri)}` : uri;
  anchor.target = '_blank'; anchor.rel = 'noopener'; anchor.title = uri; return anchor;
}
function remediatedReleaseLink(source) {
  if (source.contamination_status !== 'REMEDIATED' || !source.lexical_contamination?.receipt?.release?.private) return null;
  return receiptLink(source.current_release.url, 'Remediated PRIVATE version ↗');
}
