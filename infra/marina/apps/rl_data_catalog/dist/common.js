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

const ATLAS_VIEWS = [
  {label: 'Status', href: 'status.html', pages: ['status.html']},
  {label: 'By source', href: './', pages: ['', 'index.html', 'source.html', 'assessment.html', 'review.html', 'defects.html', 'backends.html']},
  {label: 'By curriculum', href: 'curriculum.html', pages: ['curriculum.html']},
  {label: 'Difficulty', href: 'difficulty.html', pages: ['difficulty.html', 'attempts.html', 'rollout.html']},
  {label: 'Getting started', href: 'guide.html', pages: ['guide.html']},
];

// Deferred common.js runs before each page's script, with the document parsed.
for (const navigation of document.querySelectorAll('.atlas-tabs')) {
  const filename = location.pathname.split('/').pop();
  navigation.replaceChildren(...ATLAS_VIEWS.map(view => {
    const link = node('a', view.label);
    link.href = view.href;
    if (view.pages.includes(filename)) link.setAttribute('aria-current', 'page');
    return link;
  }));
}

function qualityExplanationText(explanation) {
  if (!explanation) return '';
  const full = explanation.full, sample = explanation.sample;
  const reasons = {
    no_recorded_full_review: 'No recorded FULL review',
    full_review_other_input: `Recorded FULL review is for another input (${full?.input.dataset_id} @ ${full?.input.dataset_revision})`,
    full_review_incomplete: `Recorded FULL review is incomplete (${number.format(full?.reviewed)} of ${number.format(full?.population)} tasks reviewed)`,
    full_review_not_finalized: 'Recorded FULL review has not established FULL coverage for this input',
    full_review_input_unverified: 'Recorded FULL review input identity is incomplete',
  };
  const parts = [reasons[explanation.code]];
  if (sample?.reviewed != null) {
    const selected = sample.selected == null ? '' : ` of ${number.format(sample.selected)} selected tasks`;
    const population = sample.population == null ? '' : ` (${number.format(sample.population)} retained tasks)`;
    parts.push(`SAMPLE: ${number.format(sample.reviewed)}${selected} reviewed${population}`);
  }
  return parts.filter(Boolean).join('. ');
}
function qualityChip(quality, sourceId, prefix = '', explanation = null) {
  const chip = node(['FULL', 'SAMPLE'].includes(quality) ? 'a' : 'span', `${prefix}${quality}`, `chip level level-${quality.toLowerCase()}`);
  chip.title = qualityExplanationText(explanation);
  if (chip.tagName === 'A') {
    chip.href = `assessment.html?source=${encodeURIComponent(sourceId)}`;
    chip.setAttribute('aria-label', `${prefix}${quality}: browse task reviews`);
  }
  return chip;
}

const QC_DEFECT_LABELS = {
  missing_context: 'Missing context', ambiguity: 'Ambiguity', wrong_reference: 'Wrong reference',
  answer_leakage: 'Answer leakage', malformed: 'Malformed', rubric_mismatch: 'Rubric mismatch',
};
function defectChip(defect, suffix = '') {
  const chip = node('a', `${QC_DEFECT_LABELS[defect] || defect}${suffix}`, 'chip quality-some_issues defect-chip');
  chip.href = `defects.html#${encodeURIComponent(defect)}`;
  chip.dataset.defect = defect;
  return chip;
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

const originLabels = {tasktrove: 'TaskTrove', evalchemy: 'Evalchemy', harbor_hub: 'Harbor Hub', nemotron: 'Nemotron', 'Task Trove': 'TaskTrove'};
function originOf(row) { return originLabels[row.origin] || row.origin; }
function qualifiedSourceName(row) {
  const origin = originOf(row), build = row.build_path || '';
  const normalize = value => value.replaceAll(/\s/g, '').toLowerCase();
  const name = build && normalize(build) !== normalize(origin) ? `${build} ${row.display_name}` : row.display_name;
  return `${origin} · ${name}`;
}
const containmentPercent = new Intl.NumberFormat('en-US', {style: 'percent', maximumFractionDigits: 1});
function containmentNote(source, other, fraction, reverse = null) {
  const ownName = qualifiedSourceName(source), otherName = qualifiedSourceName(other);
  const note = node('p', undefined, 'overlap-note');
  const link = () => { const anchor = node('a', otherName); anchor.href = `source.html#${new URLSearchParams({source_id: other.id})}`; return anchor; };
  note.append(`${containmentPercent.format(fraction)} of ${ownName}'s tasks are in `, link());
  if (reverse != null) note.append(` · ${containmentPercent.format(reverse)} of `, link(), `'s tasks are in ${ownName}`);
  return note;
}
function overlapEvidence(source, labels) {
  const declaration = source.declaration || source, byId = new Map(labels.map(row => [row.id, row]));
  const content = node('div'), group = source.duplicate_group;
  if (group) {
    const canonical = byId.get(group.canonical_source_id);
    for (const member of group.members) {
      if (member.source_id === canonical.id || (declaration.id !== canonical.id && member.source_id !== declaration.id)) continue;
      content.append(containmentNote(byId.get(member.source_id), canonical, member.overlap_with_canonical));
    }
  }
  for (const pair of source.partial_overlaps || []) content.append(containmentNote(declaration, byId.get(pair.other_id), pair.source_fraction, pair.other_fraction));
  return content.children.length ? content : null;
}

// Canonical taxonomy tables served by /api/taxonomy; every page loads them
// before rendering and no page defines a label of its own. The backend table
// is the supported-production-backends enum; backendReason covers the
// auto-derived blockers for unsupported production backends.
const AtlasTaxonomy = {
  status: {}, stage: {}, reason: {}, use: {}, method: {}, backend: {}, backendReason: {},
};
async function loadTaxonomy() {
  const tables = await json('api/taxonomy');
  AtlasTaxonomy.status = tables.status || {};
  AtlasTaxonomy.stage = tables.stage || {};
  AtlasTaxonomy.reason = tables.reason || {};
  AtlasTaxonomy.use = tables.use || {};
  AtlasTaxonomy.method = tables.method || {};
  AtlasTaxonomy.backend = tables.backend || {};
  AtlasTaxonomy.backendReason = tables['backend-reason'] || {};
  return tables;
}
const supportClasses = {supported: 'good', partial: 'some_issues', unsupported: 'bad', role_absent: 'unassessed'};
const backendStatusLabels = {partial: 'Partial', role_absent: 'N/A (no role)', verify: 'Verify (live probe pending)'};
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
function displayBackendFit(fit, taskCount) {
  const roles = {}, affected = {}, blockers = {...fit.blockers};
  for (const role of ['agent', 'grader']) {
    const counts = {supported: 0, partial: 0, unsupported: 0, role_absent: 0};
    for (const [state, count] of Object.entries(fit.roles[role] || {})) counts[['supported', 'partial', 'role_absent'].includes(state) ? state : 'unsupported'] += count;
    const known = Object.values(counts).reduce((sum, count) => sum + count, 0);
    if (known < taskCount) { counts.unsupported += taskCount - known; blockers[`${role}:not_measured`] = taskCount - known; }
    roles[role] = Object.fromEntries(Object.entries(counts).filter(([, count]) => count > 0));
    affected[role] = counts.unsupported + counts.partial;
  }
  for (const [reason, count] of Object.entries(fit.not_applicable || {})) {
    const [role, ...kind] = reason.split(':'); blockers[`${role}:unsupported_environment_kind:${kind.join(':')}`] = count;
  }
  if (fit.support === 'partial' && !Object.keys(blockers).length) blockers.incomplete_requirement_support = fit.task_counts.partial || taskCount;
  const absent = Object.values(roles).every(counts => (counts.role_absent || 0) === taskCount);
  const pending = fit.support === 'verify' || Object.values(roles).some(counts => (counts.verify || 0) > 0);
  const failed = ['partial', 'unsupported', 'not_applicable'].includes(fit.support) || Object.values(affected).some(count => count > 0) || Object.keys(blockers).length || Object.keys(fit.validation_errors).length;
  return {...fit, roles, blockers, affected_roles: affected, support: absent ? 'role_absent' : fit.support === 'partial' ? 'partial' : failed ? 'unsupported' : pending ? 'verify' : 'supported'};
}
// The auto-derived reasons one backend's fit fails: typed demands render as
// their own value; the non-demand blockers map onto the canonical enum. A
// blocker key outside both fails loudly — no free-text reason renders.
function blockerReasons(fit) {
  const reasons = new Set();
  for (const key of Object.keys(fit.blockers || {})) {
    const capability = key.split(':').slice(1).join(':');
    // The derivation emits a canonical reason for every stored blocker and the
    // catalog load refuses a survey with an unmapped one; surveys sealed before
    // that guarantee keep the key-derived mapping below.
    const emitted = (fit.blocker_reasons || {})[key];
    if (emitted && AtlasTaxonomy.backendReason[emitted]) reasons.add(emitted);
    else if (capability === 'required_backend') reasons.add('required-backend');
    // displayBackendFit derives these from the survey's not_applicable environment kinds.
    else if (capability.startsWith('unsupported_environment_kind:')) reasons.add('unsupported-environment-kind');
    else if (AtlasTaxonomy.backendReason[capability]) reasons.add(capability);
    else throw new Error(`Backend blocker "${key}" maps to no typed reason; the survey drifted from the taxonomy.`);
  }
  if (Object.keys(fit.validation_errors || {}).length) reasons.add('validation-error');
  return [...reasons].sort();
}
function backendLabels(source) {
  const capabilities = source.backend_capabilities;
  if (!capabilities || capabilities.basis === 'unassessed') return [];
  // The fit table is the supported-production-backends enum plus the reasons
  // for the unsupported ones; dev-only backends the survey also measured never
  // render, and a backend outside the served taxonomy is drift.
  return Object.entries(capabilities.backends)
    .filter(([key]) => AtlasTaxonomy.backend[key])
    .map(([key, recorded]) => {
      const fit = displayBackendFit(recorded, capabilities.task_count);
      const reasons = blockerReasons(fit);
      const name = AtlasTaxonomy.backend[key].label;
      const status = backendStatusLabels[fit.support] || fit.support.toUpperCase();
      const reasonText = reasons.map(reason => AtlasTaxonomy.backendReason[reason]?.label || reason).join(', ');
      return {key, fit, reasons, label: fit.support === 'supported' ? name : `${name}: ${status}${reasons.length ? ': ' + reasonText : ''}`};
    });
}
function daytonaEvidenceRows(source, cap) {
  const declaration = source.declaration || source, safety = source.daytona;
  const snapshots = safety.snapshots ?? declaration.daytona_snapshot_count;
  const limit = safety.limit ?? declaration.daytona_snapshot_threshold ?? cap;
  const ops = receiptLink('https://github.com/marin-community/marin/blob/main/.agents/ops/daytona.md#snapshot-preflight-and-reclamation', 'How to make it safe: Daytona operations ↗');
  return [
    ['Actor verdict', daytonaLabel(safety)],
    ['Unique actor snapshot identities', snapshots ?? 'Not measured'],
    ['Snapshot cap', limit ?? 'Not recorded'],
    ['Snapshot budget rule', limit == null ? 'Not recorded' : `More than ${limit} unique actor snapshot identities is UNSAFE`],
    ['Actor machine slots', declaration.daytona_machine_slots ?? 'Not measured'],
    ['Actor roles', declaration.daytona_roles],
    ['Failed Daytona execution requirements', safety.unsafe_reasons ?? declaration.daytona_unsafe_reasons],
    ['Requirement details', safety.unsafe_details ?? declaration.daytona_unsafe_details],
    ['Affected actor tasks', safety.unsafe_counts ?? declaration.daytona_unsafe_counts],
    ['Execution failures', declaration.daytona_execution_failures ?? 'Not measured'],
    ['Measurement basis', safety.basis || declaration.daytona_count_basis || safety.reason],
    ['Exemption', declaration.daytona_exemption],
    ['Actor census input and provenance', declaration.actor_census_input],
    ['Actor census receipt', declaration.actor_census_input?.evidence_url && receiptLink(declaration.actor_census_input.evidence_url, 'Actor census receipt ↗')],
    ['Operations', ops],
  ];
}
function daytonaTooltip(source, cap) {
  return daytonaEvidenceRows(source, cap).filter(([, value]) => value != null).map(([label, value]) => {
    const text = value instanceof Node ? `${value.textContent} ${value.href || ''}` : typeof value === 'object' ? JSON.stringify(value) : value;
    return `${label}: ${text}`;
  }).join('\n');
}
function backendLink(source, cap) {
  const anchor = node('a', 'Backends ›', 'evidence-link backends-link');
  anchor.href = `backends.html#${new URLSearchParams({source_id: source.id})}`;
  anchor.title = daytonaTooltip(source, cap);
  anchor.setAttribute('aria-label', `Backend evidence for ${source.id}. Daytona ${daytonaLabel(source.daytona)}.`);
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
  // Header order matches the cells: the reason chips sit beside the overall verdict.
  for (const title of ['Backend', 'Overall', 'Reasons', 'Agent', 'Grader', 'Affected tasks by role', 'Blockers', 'Validation errors']) headings.append(node('th', title));
  head.append(headings); table.append(head); const body = node('tbody');
  for (const {key, fit, reasons} of backendLabels(source)) {
    const row = node('tr'); row.append(node('td', AtlasTaxonomy.backend[key].label));
    const status = node('td'); status.append(node('span', backendStatusLabels[fit.support] || fit.support.toUpperCase(), `chip quality-${supportClasses[fit.support]}`)); row.append(status);
    const reasonCell = node('td', undefined, 'backends-cell');
    if (fit.support === 'supported') reasonCell.append(node('span', '—'));
    else if (reasons.length) {
      for (const reason of reasons) {
        const chip = node('span', AtlasTaxonomy.backendReason[reason]?.label || reason, 'chip');
        chip.title = AtlasTaxonomy.backendReason[reason]?.meaning || '';
        reasonCell.append(chip);
      }
    } else reasonCell.append(node('span', 'Not measured', 'muted'));
    row.append(reasonCell);
    for (const role of ['agent', 'grader']) {
      const cell = node('td');
      const counts = Object.fromEntries(Object.entries(fit.roles[role] || {}).filter(([, count]) => count > 0).map(([state, count]) => [backendStatusLabels[state] || state, count]));
      cell.append(Object.keys(counts).length ? readableRecord(counts) : node('span', 'Not measured', 'muted')); row.append(cell);
    }
    for (const counts of [fit.affected_roles, fit.blockers, fit.validation_errors]) {
      const cell = node('td');
      cell.append(counts === undefined ? node('span', 'Not measured', 'muted') : Object.keys(counts).length ? readableRecord(counts) : node('span', 'None measured')); row.append(cell);
    }
    body.append(row);
  }
  if (!body.children.length) return node('p', `Unsupported: current backend fit is not measured. ${source.backend_capabilities?.reason || ''}`, 'muted');
  table.append(body); wrap.append(table); return wrap;
}
function filtrationChip(filtration) {
  const label = {excluded: 'EXCLUDED', pending: 'IN PROGRESS', pending_oracle: 'PENDING ORACLE', pending_static: 'PENDING STATIC', static_retained: 'NO ORACLE', static_blocked: 'STATIC BLOCKED', complete: 'COMPLETE'}[filtration.state];
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
    ['Current admission state', {complete: 'Admitted after full oracle filtration', excluded: 'Excluded', static_retained: 'Admitted under no-oracle policy', static_blocked: 'No tasks remain after typed static exclusions', pending_oracle: 'Pending oracle filtration'}[admission?.state] || 'Pending qualified current-pin receipt'],
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
    ['Admitted EVAL tasks (excluded from TRAIN selection)', source.intended_use === 'eval' ? admission?.admitted_count : null],
    ['Unexecutable on production backend dropped (by typed reason)', admission?.state === 'complete' ? 'Recorded in the pinned oracle-filter receipt' : counts?.oracle_tasks === 0 ? 'Per-task executability run is not required under the pinned no-oracle policy; static exclusions remain applied' : admission?.state === 'excluded' ? 'No retained tasks; typed exclusions are recorded above' : 'Pending: no pin-bound per-task production-executability receipt is recorded'],
    ['Admitted after all filtration', admission?.admitted_count ?? 'Pending: full-population filtration and production executability are not established'],
    ...['policy', 'actorless_policy', 'oracle_inventory', 'static_audit'].map(key => {
      const evidence = counts?.[key];
      const label = {policy: 'No-oracle count policy ruling', actorless_policy: 'Actorless exemption ruling', oracle_inventory: 'Current-input oracle census', static_audit: 'Current-input static blocker audit'}[key];
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

const graderLabels = {'llm-judge': 'LLM judge', 'exact-match': 'Exact match', 'math-verify': 'Math verify', 'unit-tests': 'Unit tests', script: 'Script', 'reward-model': 'Reward model', none: 'None', mixed: 'Mixed'};
function graderChip(source) {
  const reason = source.derived_contract_pending_reason;
  if (source.grader_type) return node('span', graderLabels[source.grader_type], 'chip quality-good');
  if (reason?.state === 'stale-pin' && reason.last_computed_dataset_revision) {
    const chip = node('span', 'Not computed for this pin', 'chip quality-unassessed stale-pin');
    chip.title = `The derived-contract census last ran at ${reason.last_computed_dataset_revision}; the current pin is re-pinned and awaits its census.`;
    return chip;
  }
  return node('span', 'Pending', 'chip quality-unassessed');
}

const PRIVATE_DATA_TIER = 'private_always';
function contentWithheld(data) {
  return Boolean(data?.private || data?.content_withheld || data?.data_restriction === PRIVATE_DATA_TIER
    || data?.declaration?.content_withheld || data?.declaration?.data_restriction === PRIVATE_DATA_TIER);
}
function contentWithheldMessage(data) {
  return data?.marker || data?.declaration?.marker || 'Content is not displayed.';
}
