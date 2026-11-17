/* Copyright The Marin Authors. SPDX-License-Identifier: Apache-2.0 */

// The Difficulty coverage page renders the document the rebuild published:
// computed counts, the NEEDS_MEASUREMENT queue and the blocked evidence.
// Nothing here is hand-typed; every number comes from /api/difficulty-coverage.

const SOLVER_TITLES = {small: 'Small', large: 'Large', hosted: 'Hosted', snowball: 'Snowball v0.1'};

const COUNT_LABELS = [
  ['sources', 'Sources in scope'],
  ['measured', 'Measured'],
  ['needs_measurement', 'Needs measurement'],
  ['blocked', 'Blocked'],
  ['solvers_measured', 'Solver measurements'],
  ['solvers_needs_measurement', 'Solver slots open'],
  ['solvers_gap_blocked', 'Solver slots blocked'],
];

function metric(label, value) {
  const chip = document.createElement('div');
  chip.className = 'status-metric';
  chip.setAttribute('role', 'listitem');
  const number = document.createElement('strong');
  number.textContent = value;
  const name = document.createElement('span');
  name.textContent = label;
  chip.append(number, name);
  return chip;
}

function sourceLink(sourceId) {
  const link = document.createElement('a');
  link.href = `source.html?source=${encodeURIComponent(sourceId)}`;
  link.textContent = sourceId;
  return link;
}

function missingSolvers(source) {
  const missing = source.solvers.filter(model => model.status !== 'measured');
  return missing.map(model => SOLVER_TITLES[model.solver] || model.solver).join(', ');
}

function renderCounts(counts) {
  const holder = document.getElementById('coverage-counts');
  for (const [key, label] of COUNT_LABELS) {
    holder.append(metric(label, counts[key] ?? 0));
  }
}

function renderNeeds(sources) {
  const body = document.getElementById('needs-rows');
  for (const source of Object.values(sources).filter(row => row.status === 'needs_measurement')) {
    const row = document.createElement('tr');
    const id = document.createElement('td');
    id.append(sourceLink(source.source_id));
    const measured = document.createElement('td');
    measured.textContent = `${source.measurement_ids.length}/${source.solvers.length}`;
    const missing = document.createElement('td');
    missing.textContent = missingSolvers(source);
    row.append(id, measured, missing);
    body.append(row);
  }
}

function renderBlocked(sources) {
  const body = document.getElementById('blocked-rows');
  for (const source of Object.values(sources).filter(row => row.status === 'blocked')) {
    const row = document.createElement('tr');
    const id = document.createElement('td');
    id.append(sourceLink(source.source_id));
    const gaps = document.createElement('td');
    gaps.textContent = [...new Set(source.gaps)].join(', ');
    const evidence = document.createElement('td');
    for (const reason of source.evidence) {
      const detail = document.createElement('div');
      detail.textContent = reason;
      evidence.append(detail);
    }
    row.append(id, gaps, evidence);
    body.append(row);
  }
}

function renderMeasured(sources) {
  const body = document.getElementById('measured-rows');
  for (const source of Object.values(sources).filter(row => row.status === 'measured')) {
    const row = document.createElement('tr');
    const id = document.createElement('td');
    id.append(sourceLink(source.source_id));
    const rates = document.createElement('td');
    for (const model of source.solvers) {
      const cell = document.createElement('a');
      cell.href = `attempts.html?${new URLSearchParams({source: source.source_id, ...(model.measurement_id ? {measurement: model.measurement_id} : {})})}`;
      cell.className = 'coverage-solver';
      cell.textContent = `${SOLVER_TITLES[model.solver] || model.solver}: ${(100 * model.solved / Math.max(model.n, 1)).toFixed(1)}% (${model.solved}/${model.n})`;
      rates.append(cell);
    }
    row.append(id, rates);
    body.append(row);
  }
}

async function main() {
  try {
    const document = await json('api/difficulty-coverage');
    renderCounts(document.counts);
    renderNeeds(document.sources);
    renderBlocked(document.sources);
    renderMeasured(document.sources);
    document.getElementById('coverage-meta').textContent =
      `Computed over the catalog load ${document.load_id} (snapshot ${document.snapshot_id}).`;
    document.getElementById('coverage-stamp').textContent =
      `Computed ${document.generated_at} · load ${document.load_id}`;
  } catch (error) {
    document.getElementById('coverage-meta').textContent = `Coverage unavailable: ${error.message}`;
  }
}

main();
