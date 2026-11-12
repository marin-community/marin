/* Copyright The Marin Authors. SPDX-License-Identifier: Apache-2.0 */
// One page per source that gathers every quality-evidence pointer the loaded catalog holds.
const number = new Intl.NumberFormat('en-US');
const FILER = 'https://filer.marina.oa.dev/#';
const resultLabels = {GOOD: 'Good', SOME_ISSUES: 'Some issues', BAD: 'Bad'};
const resultClasses = {GOOD: 'good', SOME_ISSUES: 'some_issues', BAD: 'bad'};

// Object-store URLs get a Filer link beside the raw URL, which stays available for CLI use.
function pointer(uri, label) {
  const span = node('span', undefined, 'pointer');
  if (/^https:\/\//.test(uri)) {
    const anchor = node('a', label || uri); anchor.href = uri; anchor.target = '_blank'; anchor.rel = 'noopener'; span.append(anchor);
  } else if (/^(s3|gs):\/\//.test(uri)) {
    span.append(node('code', uri), ' ');
    const anchor = node('a', 'Open in Filer ↗'); anchor.href = FILER + encodeURIComponent(uri); anchor.target = '_blank'; anchor.rel = 'noopener'; span.append(anchor);
  } else span.append(node('span', uri));
  return span;
}
// Free-text evidence fields often embed URLs; link each one in place.
function linkedText(text) {
  const span = node('span');
  for (const part of String(text).split(/((?:https|s3|gs):\/\/[^\s,;)]+)/)) span.append(/^(https|s3|gs):\/\//.test(part) ? pointer(part) : part);
  return span;
}
function card(title, rows, intro) {
  const section = node('section', undefined, 'assessment-card');
  section.append(node('h2', title));
  if (intro) section.append(node('p', intro, 'card-intro'));
  const present = rows.filter(([, item]) => item !== null && item !== undefined && item !== '');
  if (!present.length) { section.append(node('p', 'Nothing recorded in this catalog.', 'muted')); return section; }
  for (const [label, item] of present) {
    const field = node('dl', undefined, 'field'), value = node('dd');
    if (item instanceof Node) value.append(item);
    else if (typeof item === 'object') value.append(node('code', JSON.stringify(item)));
    else value.append(linkedText(item));
    field.append(node('dt', label), value); section.append(field);
  }
  return section;
}
function list(items) {
  const element = node('ul', undefined, 'evidence-list');
  for (const item of items) { const li = node('li'); li.append(item); element.append(li); }
  return element;
}
function link(text, href) { const anchor = node('a', text); anchor.href = href; return anchor; }

function render(data, sourceId) {
  const d = data.declaration, curation = data.curation, controls = curation?.verification, assessment = data.assessment;
  document.title = `${d.display_name} · Evidence · RL Data Atlas`;
  $('title').textContent = d.display_name;
  $('message').textContent = `${d.origin} · ${d.id} · catalog snapshot ${data.catalog.snapshot_id}`;
  const result = data.result;
  $('labels').replaceChildren(
    node('span', `Quality ${data.quality}`, `chip level level-${data.quality.toLowerCase()}`),
    node('span', result ? `Result ${resultLabels[result]}` : 'Result not assessed', `chip quality-${result ? resultClasses[result] : 'unassessed'}`),
    node('span', data.daytona.state === 'na' ? 'Daytona N/A' : data.daytona.state === 'unmeasured' ? 'Daytona not measured' : `Daytona ${data.daytona.state}`, 'chip quality-unassessed'),
    node('span', data.updated_at ? `Last updated ${new Date(data.updated_at).toLocaleDateString()}` : 'Not processed by a published campaign', 'chip quality-unassessed'),
  );
  const sections = [];
  sections.push(card('Oversight evidence', [
    ['Declared level', d.oversight], ['Campaign', d.oversight_evidence?.campaign], ['Recipe revision', d.oversight_evidence?.recipe_revision],
    ['Verification', d.oversight_evidence?.verification], ['Reviewed', d.oversight_evidence?.reviewed], ['Admitted', d.oversight_evidence?.admitted],
    ['Oracle validation', d.oracle_validation],
  ], 'The review the admission rests on, as the declaration records it.'));
  sections.push(card('Grader controls and admission', [
    ['Last curation', curation && `${curation.mode} pass, ${curation.status}`], ['Control status', controls?.status],
    ['Control sample', controls && controls.sample_count != null ? `${number.format(controls.sample_count)} of ${number.format(controls.eligible_count)} eligible tasks` : null],
    ['Control counts', controls?.counts], ['Admission counts', curation?.admission_counts],
    ['Source revision', curation?.source_revision], ['Recipe revision', curation?.recipe_revision],
    ['Latest attempt', data.latest_attempt.key && `${data.latest_attempt.status}: ${data.latest_attempt.key}`], ['Attempt note', data.latest_attempt.reason],
  ], 'Golden answers must score 1 and empty or wrong answers 0 before tasks are admitted.'));
  const admitted = data.cohorts.admitted, excluded = data.cohorts.excluded;
  sections.push(card('Sampled LLM-judge review', assessment ? [
    ['Review page', link(`Open the ${number.format(assessment.sample_items)} frozen sample items and their packets →`, `assessment.html?source=${encodeURIComponent(sourceId)}`)],
    ['Admitted cohort', admitted && `${admitted.grade}: ${admitted.grade_reason} (${number.format(admitted.reviewed)} reviewed of ${number.format(admitted.population)})`],
    ['Excluded cohort', excluded && `${excluded.grade}: ${excluded.grade_reason}`],
    ['Judge model', `${assessment.model.model} @ ${assessment.model.revision}`], ['Scope', assessment.scope],
    ['Judge traces', assessment.traces.length ? list(assessment.traces.map(trace => { const span = node('span'); span.append(`${trace.name}: `, pointer(trace.uri)); return span; })) : null],
    ['Curation record', assessment.curation_uri && pointer(assessment.curation_uri)],
    ['Read an archived file', node('code', `uv run python -c "import sys; from rigging.filesystem.s3_compat import configure_coreweave_s3; configure_coreweave_s3(); from finestore.reader import ReadView; sys.stdout.buffer.write(ReadView('${data.catalog.root}').resolve(sys.argv[1]))" FINESTORE_URI`)],
    ['Rubric', (() => { const details = node('details'); details.append(node('summary', 'Show the rubric'), node('pre', assessment.rubric)); return details; })()],
  ] : [], 'A GLM judge grades a seeded sample of admitted and excluded tasks; it does not change admission.'));
  sections.push(card('Independent QC', d.independent_qc ? [
    ['Verdict', d.independent_qc.verdict], ['Method', d.independent_qc.method], ['Reviewed', d.independent_qc.reviewed],
    ['Admitted', d.independent_qc.admitted], ['Evidence', d.independent_qc.evidence_url && pointer(d.independent_qc.evidence_url, 'Release manifest ↗')],
  ] : [], 'Quality control the upstream release recorded itself. It counts as QUICK.'));
  sections.push(card('Releases and measurements', [
    ['Dataset', d.url && pointer(d.url, `${d.dataset_id || d.url} ↗`)], ['Dataset revision', d.dataset_revision],
    ['Pipeline', d.pipeline && `${d.pipeline.name} @ ${d.pipeline.version}`],
    ['Examples', data.coverage?.example_count != null ? number.format(data.coverage.example_count) : null], ['Use', data.coverage?.intended_use],
    ['Top curriculum area', data.coverage?.top_area && `${data.coverage.top_area.name} (${number.format(data.coverage.top_area.instances)} placed tasks)`],
    ['Coverage basis', data.coverage?.basis], ['Coverage gap', data.coverage?.unavailable_reason],
    ['Daytona', data.daytona.state === 'na' || data.daytona.state === 'unmeasured' ? data.daytona.reason : `${data.daytona.state}: ${number.format(data.daytona.snapshots)} unique snapshots, limit ${data.daytona.limit}`],
    ['Daytona basis', data.daytona.basis],
  ]));
  sections.push(card('Diagnostics and archived reviews', [
    ['Diagnostics', data.diagnostics.length ? list(data.diagnostics.map(file => { const span = node('span'); span.append(`${file.name}: `, pointer(file.uri)); return span; })) : null],
    ['Archived reviews', data.archived_reviews.length ? list(data.archived_reviews.map(review => link(`${new Date(review.updated_at).toLocaleDateString()} · ${review.review_count} reviews`, `archive.html?id=${encodeURIComponent(review.id)}`))) : null],
    ['Catalog', pointer(data.catalog.root)], ['Notes', d.notes],
  ]));
  $('sections').replaceChildren(...sections);
}
const sourceId = new URLSearchParams(location.hash.slice(1)).get('source_id');
if (!sourceId) $('message').textContent = 'Open a source from the Atlas to see its evidence.';
else json(`api/source?${new URLSearchParams({source_id: sourceId})}`).then(data => render(data, sourceId)).catch(error => {
  $('title').textContent = 'Source evidence'; $('message').textContent = error.message; $('message').classList.add('error');
});
