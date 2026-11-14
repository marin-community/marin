/* Copyright The Marin Authors. SPDX-License-Identifier: Apache-2.0 */
// One page per source that gathers every quality-evidence pointer the loaded catalog holds.
const number = new Intl.NumberFormat('en-US');
const FILER = 'https://filer.marina.oa.dev/#';
const resultLabels = {GOOD: 'Good', SOME_ISSUES: 'Some issues', BAD: 'Bad', UNKNOWN: 'Unknown'};
const resultClasses = {GOOD: 'good', SOME_ISSUES: 'some_issues', BAD: 'bad', UNKNOWN: 'unknown'};

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
    else if (typeof item === 'object') value.append(readableRecord(item));
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

async function render(data, sourceId) {
  const d = data.declaration, curation = data.curation, controls = curation?.verification, assessment = data.assessment;
  const current = currentDataset(d);
  document.title = `${d.display_name} · Evidence · RL Data Atlas`;
  $('title').textContent = d.display_name;
  $('message').textContent = `${d.origin} · ${d.id} · catalog snapshot ${data.catalog.snapshot_id}`;
  const result = data.result;
  const qualityBadge = node('span', `Quality ${data.quality}`, `chip level level-${data.quality.toLowerCase()}`);
  if (data.quality_review_status) qualityBadge.title = data.quality_review_status;
  $('labels').replaceChildren(
    qualityBadge,
    node('span', result ? `Result ${resultLabels[result]}${data.result_provisional ? ' · provisional' : ''}` : 'Result not assessed', `chip quality-${result ? resultClasses[result] : 'unassessed'}${result && data.result_provisional ? ' result-provisional' : ''}`),
    contaminationChip(data.contamination_status),
    node('span', `Daytona ${daytonaLabel(data.daytona)}`, 'chip quality-unassessed'),
    node('span', data.updated_at ? `Last updated ${new Date(data.updated_at).toLocaleDateString()}` : 'Catalog load time unavailable', 'chip quality-unassessed'),
  );
  const sections = [];
  const difficulty = card('Difficulty', [
    ['Protocol', data.difficulty.protocol_id],
    ...data.difficulty.models.flatMap(model => [
      [AtlasDifficulty.titles[model.solver], model.status === 'measured' ? `${model.solved}/${model.n} solved` : model.reason],
      [`${AtlasDifficulty.titles[model.solver]} sample`, model.sample_receipt && pointer(model.sample_receipt.uri)],
      [`${AtlasDifficulty.titles[model.solver]} population`, model.sample && `${number.format(model.sample.population_count)} retained tasks; seed ${model.sample.seed}${model.sample.pre_oracle_filter ? '; pre-oracle-filter; resampling required after filtering' : ''}`],
      [`${AtlasDifficulty.titles[model.solver]} model`, model.model && `${model.model.model} @ ${model.model.revision}`],
      [`${AtlasDifficulty.titles[model.solver]} attempts`, model.attempts_receipt && pointer(model.attempts_receipt.uri)],
      [`${AtlasDifficulty.titles[model.solver]} infrastructure`, model.infrastructure_attempts == null ? null : `${model.infrastructure_attempts} failed attempts preserved; ${model.pending_tasks} pending tasks`],
    ]),
    ['Historical observations', data.difficulty.historical.length ? list(data.difficulty.historical.map(model => {
      const span = node('span'); span.append(`${AtlasDifficulty.titles[model.solver]} · ${model.protocol_id} · ${model.release.repo} @ ${model.release.revision} · ${model.status}`);
      if (model.attempts_receipt) span.append(' ', pointer(model.attempts_receipt.uri));
      return span;
    })) : null],
  ]);
  difficulty.insertBefore(AtlasDifficulty.comparison(data.difficulty), difficulty.children[1] || null);
  sections.push(difficulty);
  sections.push(card('Filtration ledger', filtrationRows({...d, contamination_status: data.contamination_status, oracle_inventory: data.oracle_inventory}), 'Retained converted counts do not establish admission. Oracle failures must be removed before an admitted count can be shown.'));
  sections.push(card('Backend fit by role', [['Measured tasks', data.backend_capabilities?.task_count], ['Fit and blockers', backendDetails(data)]], 'Agent and grader support are measured separately. The Daytona safety label covers the actor only.'));
  sections.push(card('Grading environment', [['Required boundary', gradingChips(d)], ['Census', d.grading_environment]], 'SHARED grades in the task environment; SEPARATE uses an independent grading environment. Mismatches remain explicit.'));
  sections.push(card('Oversight evidence', [
    ['Declared level', d.oversight], ['Campaign', d.oversight_evidence?.campaign], ['Recipe revision', d.oversight_evidence?.recipe_revision],
    ['Verification', d.oversight_evidence?.verification], ['Reviewed', d.oversight_evidence?.reviewed],
    ['Sampled grader qualification', d.oracle_validation],
  ], 'Recorded execution qualification. This historical record does not establish current post-filtration admission.'));
  sections.push(card('Recorded grader controls', [
    ['Last curation', curation && `${curation.mode} pass, ${curation.status}`], ['Control status', controls?.status],
    ['Control sample', controls && controls.sample_count != null ? `${number.format(controls.sample_count)} of ${number.format(controls.eligible_count)} eligible tasks` : null],
    ['Control counts', controls?.counts],
    ['Source revision', curation?.source_revision], ['Recipe revision', curation?.recipe_revision],
    ['Latest attempt', data.latest_attempt.key && `${data.latest_attempt.status}: ${data.latest_attempt.key}`], ['Attempt note', data.latest_attempt.reason],
  ], 'Golden answers must score 1 and empty or wrong answers 0 before tasks are admitted.'));
  const admitted = data.cohorts.admitted, excluded = data.cohorts.excluded;
  sections.push(card('Sampled LLM-judge review', assessment ? [
    ['Review page', link(`Open the ${number.format(assessment.sample_items)} frozen sample items and their packets →`, `assessment.html?source=${encodeURIComponent(sourceId)}`)],
    ['Retained cohort', admitted && `${admitted.grade}: ${admitted.grade_reason} (${number.format(admitted.reviewed)} reviewed of ${number.format(admitted.population)})`],
    ['Excluded cohort', excluded && `${excluded.grade}: ${excluded.grade_reason}`],
    ['Judge model', `${assessment.model.model} @ ${assessment.model.revision}`], ['Scope', assessment.scope],
    ['Judge traces', assessment.traces.length ? list(assessment.traces.map(trace => { const span = node('span'); span.append(`${trace.name}: `, pointer(trace.uri)); return span; })) : null],
    ['Curation record', assessment.curation_uri && pointer(assessment.curation_uri)],
    ['Rubric', (() => { const details = node('details'); details.append(node('summary', 'Show the rubric'), node('pre', assessment.rubric)); return details; })()],
  ] : [], 'A GLM judge grades a seeded sample of admitted and excluded tasks; it does not change admission.'));
  sections.push(card('Independent QC', d.independent_qc ? [
    ['Verdict', d.independent_qc.verdict], ['Method', d.independent_qc.method], ['Reviewed', d.independent_qc.reviewed],
    ['Evidence', d.independent_qc.evidence_url && pointer(d.independent_qc.evidence_url, 'Release manifest ↗')],
  ] : [], 'Quality control recorded by the pinned release. Complete content-review coverage is shown separately from execution qualification. QC never rejects a source.'));
  sections.push(card('Converted examples', d.converted_examples ? [
    ['TaskSpecs', number.format(d.converted_examples.count)], ['Input revision', d.converted_examples.source_revision],
    ['Converter revision', d.converted_examples.converter_revision], ['Conversion basis', d.converted_examples.converter_version],
    ['Census receipt', pointer(d.converted_examples.evidence_url)],
  ] : [], 'Measured outputs from the registered converter, before admission gates.'));
  sections.push(card('Releases and measurements', [
    ['Origin', d.origin], ['Build path', d.build_path],
    ['Datasets', datasetLinks(d)], ['Input revision', d.dataset_revision],
    ['Current release pin', current && `${current.repo} @ ${current.revision}`],
    ['Pipeline', d.pipeline && `${d.pipeline.name} @ ${d.pipeline.version}`],
    ['Retained converted population', d.within_source_dedup?.unique_count], ['Use', data.intended_use],
    ['Top curriculum area', data.coverage?.top_area && `${data.coverage.top_area.name} (${number.format(data.coverage.top_area.instances)} placed tasks)`],
    ['Coverage basis', data.coverage?.basis], ['Coverage gap', data.coverage?.unavailable_reason],
    ['Daytona', daytonaLabel(data.daytona) + (data.daytona.state === 'safe' ? `: ${number.format(data.daytona.snapshots)} snapshots` : '')],
    ['Daytona reason', data.daytona.reason], ['Actor unsafe reasons', data.daytona.unsafe_reasons], ['Actor unsafe details', data.daytona.unsafe_details], ['Affected actor tasks', data.daytona.unsafe_counts],
    ['Daytona basis', data.daytona.basis || d.daytona_count_basis],
    ['Daytona machine slots', d.daytona_machine_slots], ['Daytona roles', d.daytona_roles],
    ['Daytona execution failures', d.daytona_execution_failures],
  ]));
  const dedup = d.within_source_dedup;
  sections.push(card('Within-source deduplication', dedup ? [
    ['Raw converted tasks', dedup.raw_count], ['Unique contracts', dedup.unique_count],
    ['Duplicates dropped', dedup.within_source_duplicates],
    ['Input dataset', dedup.source_dataset_id], ['Input pin', dedup.source_revision],
    ['Converter', `${dedup.converter_version} @ ${dedup.converter_revision}`],
    ['Drop manifest', pointer(dedup.drop_manifest_uri)], ['Census receipt', pointer(dedup.evidence_url)],
  ] : []));
  const lexical = d.lexical_contamination;
  sections.push(card('Lexical contamination', lexical ? [
    ['Contamination', contaminationChip(data.contamination_status)],
    ['Measurement method', `${lexical.ngram_length}-gram lexical overlap`],
    ['Remediated private release', remediatedReleaseLink({...d, contamination_status: data.contamination_status})],
    ['Measured tasks', lexical.measured_tasks], ['Flagged tasks', lexical.contaminated_tasks],
    ['Flagged rate', `${(100 * lexical.contaminated_fraction).toFixed(2)}%`],
    ['Matched evaluations', lexical.matched_evals], ['Maximum overlap', lexical.max_overlap],
    ['Mean maximum overlap', lexical.mean_max_overlap], ['95th percentile overlap', lexical.p95_max_overlap],
    ['Measured dataset', `${lexical.input_dataset_id} @ ${lexical.input_dataset_revision}`],
    ['Receipt status', lexical.receipt?.status], ['Retained tasks', lexical.receipt?.retained_tasks],
    ['Lexical exclusions', lexical.receipt?.dropped_tasks], ['Duplicate exclusions', lexical.receipt?.duplicate_dropped_tasks],
    ['Measurement evidence', pointer(lexical.evidence_url)], ['Task flags', pointer(lexical.task_flags_uri)],
    ['Lexical drop sets', list(lexical.drop_set_uris.map(uri => pointer(uri)))],
    ['Combined exclusion manifest', lexical.receipt?.release?.drop_manifest_uri && pointer(lexical.receipt.release.drop_manifest_uri)],
  ] : [], 'Measurements describe the recorded input population; the retained release and its exclusions are shown separately.'));
  const progress = data.content_progress;
  if (progress && (!data.review_completeness || data.review_completeness.required < progress.retained_count || data.result_provisional)) sections.push(card('Provisional content-review progress', [
    ['Status', 'Incomplete; strict final assembly pending'], ['Observed at', progress.as_of],
    ['Valid reviews (lower bound)', progress.valid_reviews_lower_bound], ['Retained contracts', progress.retained_count],
    ['Evidence', pointer(progress.evidence_uri)],
  ], 'This observation does not determine FULL or Result. Partial-batch reviews may be additional.'));
  const qualityResult = d.quality_result;
  if (data.quality_review_status) sections.push(card('Content review status', [
    ['Status', data.quality_review_status], ['Declared oversight', d.oversight],
  ], 'No exact-input Result is available yet. The preview displays QUICK while the review is pending.'));
  sections.push(card('Content review coverage', data.review_completeness ? [
    ['Status', data.result_provisional ? 'Provisional / incomplete' : d.quality_result_finalization === 'accepted_incomplete' ? 'Final with accepted whole-source waiver' : 'Required content review complete'],
    ['Strict coverage complete', String(data.review_completeness.coverage_complete)],
    ['Unreviewed tasks', qualityResult?.unavailable_count],
    ['Finalization waiver and exact unreviewed reasons', d.quality_result_waiver && pointer(d.quality_result_waiver.uri)],
    ['Waiver SHA-256', d.quality_result_waiver?.sha256],
    ['Conclusive reviewed contracts', data.review_completeness?.reviewed], ['Required reviews', data.review_completeness?.required], ['Retained contracts', d.within_source_dedup?.unique_count],
    ['Execution qualification', d.oversight],
    ['Review evidence', d.independent_qc?.evidence_url && pointer(d.independent_qc.evidence_url)],
  ] : [], 'Content QA and execution qualification are separate. A whole-source waiver can finalize FULL while strict coverage remains incomplete; its receipt lists every unreviewed task and exact reason.'));
  sections.push(card('Computed Result', qualityResult ? [
    ['Conclusive reviews', qualityResult.reviewed], ['Tasks with issues', qualityResult.issue_count],
    ['Issue rate', qualityResult.issue_rate == null ? null : `${(100 * qualityResult.issue_rate).toFixed(2)}%`],
    ['Unavailable observations', qualityResult.unavailable_count], ['Uncertain observations', qualityResult.uncertain_count],
    ['Issues', qualityResult.issues], ['Repair recommendation', qualityResult.repair_recommendation],
  ] : [], 'GOOD is at most 10% issues; BAD is at least 50% issues. QC does not reject tasks.'));
  const capabilities = data.backend_capabilities;
  sections.push(card('Backend requirements and overlap', [
    ['Measurement basis', capabilities?.basis], ['Measured tasks', capabilities?.basis === 'converted_spec_census' ? capabilities.task_count : null],
    ['Unassessed reason', capabilities?.basis === 'unassessed' ? capabilities.reason : null],
    ['Role assignments', capabilities?.role_backend_assignments], ['Declared requirements', capabilities?.demands],
    ['Duplicate group', data.duplicate_group],
    ['Full backend matrix', link('Open the backend survey', 'api/backends')],
    ['Overlap reviews', link('Open recorded duplicate opinions', 'api/source-duplicates')],
  ]));
  sections.push(card('Diagnostics and archived reviews', [
    ['Diagnostics', data.diagnostics.length ? list(data.diagnostics.map(file => { const span = node('span'); span.append(`${file.name}: `, pointer(file.uri)); return span; })) : null],
    ['Archived reviews', data.archived_reviews.length ? list(data.archived_reviews.map(review => link(`${new Date(review.updated_at).toLocaleDateString()} · ${review.review_count} reviews`, `archive.html?id=${encodeURIComponent(review.id)}`))) : null],
    ['Catalog', pointer(data.catalog.root)], ['Notes', d.notes],
  ]));
  $('sections').replaceChildren(...sections);
  if (assessment) {
    const section = card('All stored assessment reviews', [], 'Loading complete review records…');
    $('sections').append(section);
    try {
      const report = await json(`api/assessment?${new URLSearchParams({source_id: sourceId})}`);
      const record = node('details');
      record.append(node('summary', 'Complete assessment report'), node('pre', JSON.stringify(report.report, null, 2)));
      section.replaceChildren(node('h2', 'All stored assessment reviews'),
        node('p', `${number.format(report.report.evidence.samples.length)} stored assessment records. Open each review for its verdict and packet. This sampled evidence is separate from complete content-review coverage and execution qualification.`, 'card-intro'),
        repairRecommendations(report.report), taskReviews(sourceId, report.report.evidence.samples), record);
    } catch (error) { section.append(node('p', error.message, 'error')); }
  }
  for (const summary of data.archived_reviews) {
    const section = card(`Archived review collection ${summary.id}`, [], 'Loading complete review records…');
    $('sections').append(section);
    try {
      const review = await json(`api/archive/reviews/${encodeURIComponent(summary.id)}`);
      const records = node('details', undefined, 'archived-collection');
      records.append(node('summary', 'Complete review collection'), repairRecommendations(review.collection), node('pre', JSON.stringify(review.collection, null, 2)));
      section.replaceChildren(node('h2', `Archived review collection ${review.id}`), records);
      const artifacts = list(review.artifacts.map(artifact => link(artifact.path,
        `api/archive/reviews/${encodeURIComponent(review.id)}/artifacts/${artifact.path.split('/').map(encodeURIComponent).join('/')}`)));
      section.append(node('h3', 'Saved reports, traces and receipts'), artifacts);
    } catch (error) { section.append(node('p', error.message, 'error')); }
  }

}
const sourceId = new URLSearchParams(location.hash.slice(1)).get('source_id');
if (!sourceId) $('message').textContent = 'Open a source from the Atlas to see its evidence.';
else Promise.all([json(`api/source?${new URLSearchParams({source_id: sourceId})}`), json('measurements/content-progress.json'), json('measurements/oracle-inventory.json')])
  .then(([data, progress, inventory]) => render({...data, content_progress: boundProgress(data.declaration, progress), oracle_inventory: boundOracleInventory(data.declaration, inventory)}, sourceId)).catch(error => {
  $('title').textContent = 'Source evidence'; $('message').textContent = error.message; $('message').classList.add('error');
});
