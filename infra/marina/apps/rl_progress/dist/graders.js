const params = new URLSearchParams(location.search);
const run = params.get('run');
const attempt = params.get('attempt');
const status = document.getElementById('status');
const results = document.getElementById('results');
const percent = new Intl.NumberFormat(undefined, { style: 'percent', maximumFractionDigits: 1 });
const seconds = new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 });

function element(tag, value) {
  const node = document.createElement(tag);
  if (value != null) node.textContent = value;
  return node;
}

const columns = [
  ['Environment', (row) => row.environment],
  ['Task source', (row) => row.data_source],
  ['Attempted', (row) => row.attempted],
  ['Verified', (row) => row.verified],
  ['Coverage', (row) => row.verification_coverage == null ? null : percent.format(row.verification_coverage)],
  ['Success rate', (row) => row.success_rate == null ? null : percent.format(row.success_rate)],
  ['Grader failures', (row) => counts(row.grading_failures)],
  ['Rollout failures', (row) => counts(row.rollout_failures)],
  ['Mean (s)', (row) => latency(row, 'mean')],
  ['p95 (s)', (row) => latency(row, 'p95')],
  ['Max (s)', (row) => latency(row, 'max')],
];

function counts(values) {
  const entries = Object.entries(values ?? {});
  return entries.length ? entries.map(([reason, count]) => `${reason.replaceAll('_', ' ')} (${count})`).join(', ') : '0';
}

function latency(row, key) {
  const value = row.grading_latency_seconds?.[key];
  return value == null ? null : seconds.format(value);
}

const failureCount = (row) => Object.values(row.grading_failures ?? {}).reduce((total, value) => total + value, 0);

function section(title, rows) {
  const panel = element('section');
  panel.className = 'panel';
  panel.append(element('h2', title));
  if (!rows.length) {
    panel.append(element('p', 'No samples were recorded.'));
    return panel;
  }
  const table = element('table');
  const head = table.createTHead().insertRow();
  for (const [label] of columns) {
    const cell = element('th', label);
    cell.scope = 'col';
    head.append(cell);
  }
  const body = table.createTBody();
  const sorted = [...rows].sort((a, b) => failureCount(b) - failureCount(a)
    || (b.grading_latency_seconds?.p95 ?? 0) - (a.grading_latency_seconds?.p95 ?? 0));
  for (const row of sorted) {
    const tr = body.insertRow();
    for (const [, value] of columns) tr.append(element('td', value(row) ?? '—'));
  }
  const container = element('div');
  container.className = 'table-scroll grader-table';
  container.append(table);
  panel.append(container);
  return panel;
}

async function load() {
  if (!run || !attempt) {
    status.textContent = 'Choose a run from RL iteration progress.';
    return;
  }
  const link = document.getElementById('run-details');
  link.href = './#' + new URLSearchParams({ run, attempt });
  link.hidden = false;
  try {
    const query = new URLSearchParams({ run_id: run, attempt_id: attempt });
    const response = await fetch('./api/graders?' + query);
    if (!response.ok) throw new Error(`Could not load grader stats (${response.status}).`);
    const { label, report } = await response.json();
    document.getElementById('run-label').textContent = label;
    if (!report) {
      status.textContent = 'This attempt has no grader report.';
      return;
    }
    status.textContent = 'Success rate uses verified responses. Latency covers attempts that entered grading.';
    for (const [key, rows] of Object.entries(report.evaluation ?? {})) {
      const separator = key.lastIndexOf('/');
      const evaluation = separator < 0 ? 'Evaluation' : key.slice(0, separator);
      const step = key.slice(separator + 1);
      const title = step === 'eval_only' ? evaluation : `${evaluation} · step ${step}`;
      results.append(section(title, rows));
    }
    for (const [execution, rows] of Object.entries(report.training ?? {})) {
      results.append(section(`Training · ${execution}`, rows));
    }
  } catch (error) {
    status.textContent = error.message;
  }
}

load();
