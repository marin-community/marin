/* Copyright The Marin Authors. SPDX-License-Identifier: Apache-2.0 */
// One page per source that gathers every quality-evidence pointer the loaded catalog holds.
const number = new Intl.NumberFormat('en-US');
const FILER = 'https://filer.marina.oa.dev/#';
const resultLabels = {GOOD: 'Good', SOME_ISSUES: 'Some issues', BAD: 'Bad', UNKNOWN: 'Unknown'};
const resultClasses = {GOOD: 'good', SOME_ISSUES: 'some_issues', BAD: 'bad', UNKNOWN: 'unknown'};

// Object-store URLs get a Filer link beside the raw URL, which stays available for CLI use.
function licenseEvidence(declaration) {
  if (declaration.license_ban) {
    const ban = declaration.license_ban;
    const line = node('span', `${ban.license_ids.join(', ')} · BANNED`);
    line.append(' · ', pointer(ban.issue_url, 'Decision ↗'));
    return line;
  }
  if (declaration.publication_scope) {
    const grant = declaration.publication_scope;
    const line = node('span', `${grant.license_id.toUpperCase()} · `);
    line.append(pointer(grant.upstream_url, `${grant.upstream_name} ↗`));
    if (grant.license_id === 'cc-by-4.0') {
      line.append(' · ', pointer('https://creativecommons.org/licenses/by/4.0/', 'CC BY 4.0 ↗'));
    }
    return line;
  }
  const receipt = declaration.license_receipt;
  if (!receipt) return 'Unresolved';
  const ids = [...new Set(receipt.evidence.flatMap(item => item.license_ids)
    .map(id => id.trim().toLowerCase()).filter(id => id !== 'missing'))];
  const line = node('span', ids.join(', ') || 'Missing');
  const evidence = receipt.evidence.find(item => item.license_ids.some(id => id.toLowerCase() !== 'missing'))
    || receipt.evidence[0];
  if (evidence) line.append(' · ', pointer(evidence.url, 'Evidence ↗'));
  return line;
}
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

// Each supplementary ledger loads independently; a stalled ledger cannot hide the source.
const SOURCE_DEADLINE_MS = 15000;
const AUXILIARY_DEADLINE_MS = 8000;
async function sourceJson(url, timeoutMs = AUXILIARY_DEADLINE_MS) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, {signal: controller.signal});
    if (!response.ok) throw new Error(`Request failed (${response.status}): ${url}`);
    return await response.json();
  } catch (error) {
    if (controller.signal.aborted) throw new Error(`Request timed out: ${url}`);
    throw error;
  } finally { clearTimeout(timer); }
}
function supplement(title, promise, build) {
  const section = node('section', undefined, 'assessment-card');
  section.append(node('h2', title), node('p', 'Loading supplementary evidence…', 'muted'));
  promise.then(value => section.replaceWith(build(value))).catch(error => {
    section.replaceChildren(node('h2', title), node('p', error.message, 'error'));
  });
  return section;
}
function candidateCards(pending, sourceId) {
  const fragment = document.createDocumentFragment();
  for (const candidate of pending.sources.filter(item => item.source_id === sourceId)) {
    const release = candidate.release, sample = candidate.sample;
    const section = card('Filtered private release, pending qualification', [
      ['Candidate source', candidate.candidate_source_id],
      ['Candidate pin', pointer(release.url, `${release.repo} @ ${release.revision}`)],
      ['Oracle-filtered tasks', `${number.format(candidate.filtered_tasks)} of ${number.format(candidate.original_tasks)}`],
      ['Qualification pending', candidate.qualification_pending.map(name => name.replaceAll('_', ' ')).join(', ')],
      ['SAMPLE selected', sample.selected],
      ['SAMPLE reviewed', sample.reviewed],
      ['SAMPLE unavailable', sample.unavailable],
      ['SAMPLE uncertain', sample.uncertain],
      ['SAMPLE issues', sample.issues],
      ['SAMPLE receipt', pointer(sample.receipt.uri)],
      ['SAMPLE receipt SHA-256', sample.receipt.sha256],
      ['Verification receipt', pointer(candidate.verification.uri)],
      ['Verification SHA-256', candidate.verification.sha256],
      ['Original Harbor trials claimed', candidate.original_harbor_trials_claimed],
    ], 'This candidate is not selected or admitted to Atlas. Its SAMPLE receipt describes only the candidate pin. The catalog release, Examples and Quality remain unchanged.');
    section.classList.add('pending-filtered-release');
    fragment.append(section);
  }
  return fragment;
}

function render(data, sourceId, auxiliary) {
  const d = data.declaration, curation = data.curation, controls = curation?.verification, assessment = data.assessment;
  const current = currentDataset(d);
  document.title = `${d.display_name} · Evidence · RL Data Atlas`;
  $('title').textContent = d.display_name;
  $('message').textContent = `${d.origin} · ${d.id} · catalog snapshot ${data.catalog.snapshot_id}`;
  const result = data.result;
  const qualityBadge = qualityChip(data.quality, sourceId, 'Quality ', data.quality_explanation);
  if (data.quality_review_status) qualityBadge.title = [qualityBadge.title, data.quality_review_status].filter(Boolean).join('. ');
  $('labels').replaceChildren(
    qualityBadge,
    node('span', result ? `Result ${resultLabels[result]}${data.result_provisional ? ' · provisional' : ''}` : 'Result not assessed', `chip quality-${result ? resultClasses[result] : 'unassessed'}${result && data.result_provisional ? ' result-provisional' : ''}`),
    contaminationChip(data.contamination_status),
    node('span', `Daytona ${daytonaLabel(data.daytona)}`, 'chip quality-unassessed'),
    node('span', data.updated_at ? `Last updated ${new Date(data.updated_at).toLocaleDateString()}` : 'Catalog load time unavailable', 'chip quality-unassessed'),
  );
  if (data.sample_pin_stale) {
    $('labels').append(node('span', 'SAMPLE pin-stale', 'chip quality-unassessed'));
  }
  const sections = [];
  if (contentWithheld(data)) {
    $('sections').replaceChildren(card('Content withheld', [], contentWithheldMessage(data)));
    return;
  }
  const difficulty = card('Difficulty', [
    ['Protocol', data.difficulty.protocol_id],
    ...data.difficulty.models.flatMap(model => [
      [AtlasDifficulty.titles[model.solver], model.status === 'measured' ? `${model.solved}/${model.n} solved` : model.reason],
      [`${AtlasDifficulty.titles[model.solver]} sample`, model.sample_receipt && pointer(model.sample_receipt.uri)],
      [`${AtlasDifficulty.titles[model.solver]} population`, model.sample && `${number.format(model.sample.population_count)} retained tasks; seed ${model.sample.seed}${model.sample.pre_oracle_filter ? '; pre-oracle-filter; resampling required after filtering' : ''}`],
      [`${AtlasDifficulty.titles[model.solver]} model`, model.model && `${model.model.model} @ ${model.model.revision}`],
      [`${AtlasDifficulty.titles[model.solver]} attempts`, model.measurement_id && link('Browse task attempts and rollouts →', `attempts.html?${new URLSearchParams({source: sourceId, measurement: model.measurement_id})}`)],
      [`${AtlasDifficulty.titles[model.solver]} infrastructure`, model.infrastructure_attempts == null ? null : `${model.infrastructure_attempts} failed attempts preserved; ${model.pending_tasks} pending tasks; ${model.blocked_tasks ?? 0} blocked tasks`],
    ]),
    ['Historical observations', data.difficulty.historical.length ? list(data.difficulty.historical.map(model => {
      const span = node('span'); span.append(`${AtlasDifficulty.titles[model.solver]} · ${model.protocol_id} · ${model.release.repo} @ ${model.release.revision} · ${model.status}`);
      if (model.attempts_receipt) span.append(' ', pointer(model.attempts_receipt.uri));
      return span;
    })) : null],
  ]);
  difficulty.insertBefore(AtlasDifficulty.comparison(data.difficulty), difficulty.children[1] || null);
  sections.push(difficulty);
  sections.push(supplement('Filtration ledger', auxiliary.inventory, inventory => card('Filtration ledger', filtrationRows({...d, filtration: data.filtration, contamination_status: data.contamination_status, oracle_inventory: boundOracleInventory(d, inventory)}), 'Examples requires complete current-pin filtration and admission proof. A source filtered to zero is excluded. Pre-filtration retained counts are shown separately.')));
  sections.push(supplement('Filtered private releases', auxiliary.pending, pending => candidateCards(pending, sourceId)));
  sections.push(card('Backend fit by role', [['Measured tasks', data.backend_capabilities?.task_count], ['Fit and blockers', backendDetails(data)]], 'Supported requires all present task/roles to be supported. Otherwise the backend is Unsupported; N/A is only for absent roles. Affected task counts are per role and may overlap. The Daytona safety label covers the actor only.'));
  sections.push(card('Grader profile', [['Grader', graderChip(d)], ['Per-type task counts', d.grader_type_counts], ['Judge rubrics', d.judge_rubric_counts], ['Judge tasks with reference', d.judge_reference_count], ['Judge tasks without reference', d.judge_no_reference_count], ['Current-input census receipt', d.derived_contract_receipt && pointer(d.derived_contract_receipt.uri)], ['Census binding', d.derived_contract_receipt?.input], ['Scanned TaskSpec bytes', d.derived_contract_receipt?.scanned_task_specs_sha256]], 'Derived from the measured task contracts. Unbound or changed inputs remain Pending.'));
  for (const variant of data.method_variants || []) {
    sections.push(card(`Dataset variant: ${variant.input.variant_id}`, [
      ['Method', variant.tags.map(tag => tag.toUpperCase()).sort().join(' / ')],
      ['Dataset', variant.input.repository], ['Revision', variant.input.revision],
      ['Content SHA-256', variant.input.data_sha256], ['TaskSpec SHA-256', variant.input.task_specs_sha256],
      ['SFT preparation receipt', variant.sft_receipt_reference && pointer(variant.sft_receipt_reference.uri)],
      ['Training output', variant.sft_receipt && pointer(variant.sft_receipt.output_manifest.uri)],
      ['Conversion parent', variant.conversion?.parent.variant_id],
      ['Conversion receipt', variant.conversion && pointer(variant.conversion.receipt.uri)],
    ]));
  }
  sections.push(card('Grading environment', [['Required boundary', gradingChips(d)], ['Census', d.grading_environment]], 'SHARED grades in the task environment; SEPARATE uses an independent grading environment. Mismatches remain explicit.'));
  if (d.consumer_requirements?.length) {
    sections.push(card('Consumer check', [
      ['Result', d.consumer_findings?.length ? d.consumer_findings.join(', ') : 'Pass'],
      ['Required capabilities', d.consumer_requirements.join(', ')],
      ['Runtime', d.consumer_receipt?.runtime_commit],
      ['Evidence', d.consumer_receipt?.evidence_url && pointer(d.consumer_receipt.evidence_url)],
      ['User ruling', d.consumer_decision?.user_ruling],
    ], 'Consumer findings are advisory. A status change requires a separate user ruling for this source.'));
  }
  // Catalogs frozen before the derived-oversight fields keep the historical prose in
  // oversight_evidence and their receipt pointers at the FULL binding or QC evidence paths.
  const derivedReceipt = d.oversight_evidence?.receipt_uri;
  const bindingReceipt = d.full_content_result_binding?.review_receipt;
  const receiptUri = derivedReceipt ?? bindingReceipt?.uri ?? (data.quality === 'QUICK' ? null : d.independent_qc?.evidence_url);
  const receiptRow = data.quality === 'QUICK'
    ? ['Content review', 'Conversion-only — no qualifying content-review receipt']
    : receiptUri
      ? [derivedReceipt || bindingReceipt ? 'Receipt' : 'QA evidence', pointer(receiptUri)]
      : null;
  // The five historical note fields may live in recipe_qa_history or, in frozen catalogs,
  // in oversight_evidence itself; never render that block's level.
  const historicalNotes = d.recipe_qa_history ?? (d.oversight_evidence && {
    campaign: d.oversight_evidence.campaign, recipe_revision: d.oversight_evidence.recipe_revision,
    verification: d.oversight_evidence.verification, reviewed: d.oversight_evidence.reviewed,
    admitted: d.oversight_evidence.admitted,
  });
  sections.push(card('Oversight evidence', [
    ['Level', qualityChip(data.quality, sourceId, 'Quality ', data.quality_explanation)],
    ['Why not FULL', qualityExplanationText(data.quality_explanation)],
    ['SAMPLE receipt', data.quality_explanation?.sample?.receipt && pointer(data.quality_explanation.sample.receipt)],
    ['Recorded FULL receipt', data.quality_explanation?.full?.receipt && pointer(data.quality_explanation.full.receipt.uri)],
    ['Recorded FULL input', data.quality_explanation?.full?.input],
    ['Different input fields', data.quality_explanation?.full?.different_fields?.join(', ')],
    ['Basis', data.quality === 'FULL' ? 'Whole-source QA for the current input' : data.quality === 'SAMPLE' ? 'Sampled content review for the current input' : 'Current-input conversion only'],
    ...(receiptRow ? [receiptRow] : []),
    ['Reviewed', data.quality === 'QUICK' ? null : d.quality_result?.reviewed ?? d.independent_qc?.reviewed],
    ['Sampled grader qualification', d.oracle_validation],
  ], 'The level is derived from the same QA evidence as the main-table Quality column: a current-pin FULL binding, SAMPLE receipt, or conversion-only QUICK. Catalogs frozen before the derived fields keep their receipt pointer at the FULL binding or recorded QC evidence.'));
  sections.push(card('Historical recipe QA notes', [
    ['Campaign', historicalNotes?.campaign], ['Recipe revision', historicalNotes?.recipe_revision],
    ['Verification', historicalNotes?.verification], ['Reviewed', historicalNotes?.reviewed],
    ['Admitted', historicalNotes?.admitted],
  ], 'Historical recipe-QA prose, kept for the record. It can never assert an oversight level; the current level is the evidence-derived one above.'));
  sections.push(card('Recorded grader controls', [
    ['Last curation', curation && `${curation.mode} pass, ${curation.status}`], ['Control status', controls?.status],
    ['Control sample', controls && controls.sample_count != null ? `${number.format(controls.sample_count)} of ${number.format(controls.eligible_count)} eligible tasks` : null],
    ['Control counts', controls?.counts],
    ['Source revision', curation?.source_revision], ['Recipe revision', curation?.recipe_revision],
    ['Latest attempt', data.latest_attempt.key && `${data.latest_attempt.status}: ${data.latest_attempt.key}`], ['Attempt note', data.latest_attempt.reason],
  ], 'Golden answers must score 1 and empty or wrong answers 0 before tasks are admitted.'));
  const admitted = data.cohorts.admitted, excluded = data.cohorts.excluded;
  sections.push(card('QC task reviews', assessment ? [
    ['Review page', link(`Open the ${number.format(assessment.sample_items)} task reviews →`, `assessment.html?source=${encodeURIComponent(sourceId)}`)],
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
    ['License', licenseEvidence(d)],
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
  sections.push(supplement('Content-review progress', auxiliary.progress, payload => {
    const progress = boundProgress(d, payload);
    if (progress && (!data.review_completeness || data.review_completeness.required < progress.retained_count || data.result_provisional)) return card('Provisional content-review progress', [
      ['Status', 'Incomplete; strict final assembly pending'], ['Observed at', progress.as_of],
      ['Valid reviews (lower bound)', progress.valid_reviews_lower_bound], ['Retained contracts', progress.retained_count],
      ['Evidence', pointer(progress.evidence_uri)],
    ], 'This observation does not determine FULL or Result. Partial-batch reviews may be additional.');
    return document.createDocumentFragment();
  }));
  const qualityResult = d.quality_result;
  if (data.quality_review_status) sections.push(card('Content review status', [
    ['Status', data.quality_review_status], ['Declared oversight', d.oversight],
  ], 'No exact-input Result is available yet. The preview displays QUICK while the review is pending.'));
  sections.push(card('Content review coverage', data.review_completeness ? [
    ['Review scope', data.review_scope ? `${data.review_scope}${data.review_coverage && data.review_coverage.required != null ? ` · ${(data.review_coverage.reviewed ?? 0).toLocaleString()} / ${data.review_coverage.required.toLocaleString()} reviewed` : ''}${data.review_coverage?.finalization === 'accepted_incomplete' ? ' · whole-source waiver for the unreviewed remainder' : ''}` : null],
    ['Status', data.result_provisional ? 'Provisional / incomplete' : d.quality_result_finalization === 'accepted_incomplete' ? 'Final with accepted whole-source waiver' : !data.review_completeness.coverage_complete ? 'Final' : 'Required content review complete'],
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
  const overlap = node('span', 'Loading overlap labels…', 'muted');
  auxiliary.labels.then(labels => overlap.replaceWith(overlapEvidence(data, labels.sources) || node('span', 'No duplicate or partial-overlap notes.'))).catch(error => {
    overlap.textContent = error.message; overlap.className = 'error';
  });
  sections.push(card('Backend requirements and overlap', [
    ['Measurement basis', capabilities?.basis], ['Measured tasks', capabilities?.basis === 'converted_spec_census' ? capabilities.task_count : null],
    ['Unassessed reason', capabilities?.basis === 'unassessed' ? capabilities.reason : null],
    ['Role assignments', capabilities?.role_backend_assignments], ['Declared requirements', capabilities?.demands],
    ['Duplicate group ID', data.duplicate_group?.group_id],
    ['Measured cross-source overlap', overlap],
    ['Full backend matrix', link('Open the backend survey', 'api/backends')],
    ['Overlap reviews', link('Open recorded duplicate opinions', 'api/source-duplicates')],
  ]));
  sections.push(card('Diagnostics and provenance', [
    ['Diagnostics', data.diagnostics.length ? list(data.diagnostics.map(file => { const span = node('span'); span.append(`${file.name}: `, pointer(file.uri)); return span; })) : null],
    ['Catalog', pointer(data.catalog.root)], ['Notes', d.notes],
  ]));
  $('sections').replaceChildren(...sections);
  if (assessment) {
    const section = node('details', undefined, 'assessment-card');
    section.append(node('summary', 'All stored assessment reviews'));
    $('sections').append(section);
    section.addEventListener('toggle', async () => {
      if (!section.open || section.dataset.requested) return;
      section.dataset.requested = 'true';
      const status = node('p', 'Loading complete review records…', 'muted');
      section.append(status);
      try {
        const report = await sourceJson(`api/assessment?${new URLSearchParams({source_id: sourceId})}`, SOURCE_DEADLINE_MS);
        const record = node('details');
        record.append(node('summary', 'Complete assessment report'), node('pre', JSON.stringify(report.report, null, 2)));
        status.replaceWith(node('p', `${number.format(report.report.evidence.total)} stored task reviews. The first page is shown below; open QC task reviews for paging and defect filters.`, 'card-intro'));
        section.append(repairRecommendations(report.report), taskReviews(sourceId, report.report.evidence.items), record);
      } catch (error) { status.textContent = error.message; status.className = 'error'; }
    });
  }
}
if (!window.AtlasSourceBootstrap) {
  // The HTML already contains an error if even the independent bootstrap failed.
  const reload = document.getElementById('source-reload');
  reload.href = location.href;
  reload.addEventListener('click', event => { event.preventDefault(); location.reload(); });
} else window.AtlasSourceBootstrap.run(() => {
  const sourceId = new URLSearchParams(location.hash.slice(1)).get('source_id');
  window.addEventListener('hashchange', () => location.reload());
  if (!sourceId) $('message').textContent = 'Open a source from the Atlas to see its evidence.';
  else {
    $('title').textContent = 'Loading source…';
    $('sections').setAttribute('aria-busy', 'true');
    // Backend fit rows render from the served taxonomy, so it loads with the source.
    Promise.all([loadTaxonomy(), sourceJson(`api/source?${new URLSearchParams({source_id: sourceId})}`, SOURCE_DEADLINE_MS)])
      .then(([, data]) => {
        // Restricted sources never request supplementary content or task reviews.
        const auxiliary = contentWithheld(data) ? {} : {
          progress: sourceJson('measurements/content-progress.json'),
          inventory: sourceJson('measurements/oracle-inventory.json'),
          labels: sourceJson('api/source-labels'),
          pending: sourceJson('measurements/pending-filtered-releases.json'),
        };
        render(data, sourceId, auxiliary);
        $('sections').setAttribute('aria-busy', 'false');
      }).catch(error => {
        window.AtlasSourceBootstrap.fail(error);
      });
  }
});
