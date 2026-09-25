// Benchmark and question views share the embedded inventory; no runtime fetches.
const benchmarkById = new Map(D.benchmarks.map(b => [b.id, b]));
const verificationNames = {
  'numeric-contract': 'Executable check proposed',
  'needs-definition': 'Scientific definition needed',
  'defer-interpretation': 'Open-ended endpoint deferred',
};
const routeHref = path => location.href.split('#')[0] + '#' + path;
const benchmarkHref = id => routeHref('benchmark/' + encodeURIComponent(id));
const questionHref = (benchmark, question) => routeHref('question/' + encodeURIComponent(benchmark) + '/' + encodeURIComponent(question));
const questionTitle = q => q.annotation?.question_summary || q.summary || q.id;
const questionLabels = q => q.annotation ? [...q.annotation.focal_competencies, ...q.annotation.supporting_competencies] : [];
const paragraphs = values => values.map(value => `<p>${esc(value)}</p>`).join('');

function benchmarkForSource(source) {
  return D.benchmarks.find(b => source.task_inventory ? b.inventory === source.task_inventory : b.id === source.name);
}

function renderBenchmarkIndex() {
  const first = benchmarkById.get('BixBench-Verified-50');
  el('benchmark-entry').innerHTML = `<h3>Start with the question-level review</h3>
    <p><a class="page-link" href="${benchmarkHref(first.id)}">BixBench-Verified-50 → 50 questions and a flat competency list</a></p>
    <p>Each benchmark has a page; each inventoried ID question has a permalink. Only BixBench-Verified currently has question-level competency annotations. Other inventories retain their inspection limits. OOD question content is excluded.</p>`;
}

function competencyCounts(benchmark, role) {
  return benchmark.review.competencies.map(c => {
    const focal = benchmark.questions.filter(q => q.annotation?.focal_competencies.includes(c.id)).length;
    const total = benchmark.questions.filter(q => questionLabels(q).includes(c.id)).length;
    return {...c, focal, total, count: role === 'focal' ? focal : total};
  }).sort((a, b) => b.count - a.count || a.name.localeCompare(b.name));
}

function renderBenchmark(benchmark) {
  const reviewed = benchmark.questions.filter(q => q.annotation).length;
  const numeric = benchmark.questions.filter(q => q.annotation?.verification_status === 'numeric-contract').length;
  const unresolved = benchmark.questions.filter(q => q.annotation?.verification_status === 'needs-definition').length;
  const deferred = benchmark.questions.filter(q => q.annotation?.verification_status === 'defer-interpretation').length;
  el('benchmark-page').innerHTML = `<div class="breadcrumbs"><a href="${routeHref('sources')}">Benchmarks</a> / ${esc(benchmark.name)}</div>
    <h2 tabindex="-1">${esc(benchmark.name)}</h2>
    <p><span class="badge">${esc(benchmark.distribution)}</span> <a href="${esc(benchmark.source_url)}">Original source</a>
    ${benchmark.inventory ? ' · ' + link('experiments/post_training/bio_tasks/' + benchmark.inventory, 'Versioned question inventory') : ''}</p>
    <p class="reference-note">Inventory scope: ${esc(benchmark.inspection)}</p>
    ${benchmark.review ? `<p class="notice">Draft competency annotations for review. These counts describe source questions, not generated tasks or validated coverage. Only executable rewards are eligible for generation; no LLM judge.</p>
    <div class="stats benchmark-stats"><div class="stat"><b>${reviewed}</b><span>Questions reviewed</span></div><div class="stat"><b>${benchmark.review.competencies.length}</b><span>Flat draft competencies</span></div><div class="stat"><b>${numeric}</b><span>Executable checks proposed</span></div><div class="stat"><b>${unresolved} / ${deferred}</b><span>Need definitions / deferred</span></div></div>
    <p>“Focal” identifies the requested endpoint; “supporting” identifies a necessary component. A question may have several labels. Related questions can share one biological study and one workflow.</p>
    <details><summary>Annotation method and counting</summary><p>${esc(benchmark.review.method)}</p><p>${esc(benchmark.review.counting)}</p><p>${esc(benchmark.review.reward_policy)}</p>
    <p>${link('experiments/post_training/bio_tasks/' + benchmark.review_file, 'Review the versioned annotations and definitions')}</p></details>
    <h3>How often does each competency appear?</h3>
    <div class="toolbar"><label for="competency-count-mode">Count</label><select id="competency-count-mode"><option value="all">Focal + supporting</option><option value="focal">Focal only</option></select><span class="muted">Click a bar to show its questions. Frequency is not a generation quota.</span></div>
    <div id="competency-bars" class="competency-bars" aria-label="Competencies ranked by source-question count"></div>` :
      `<p class="notice">${benchmark.distribution === 'OOD' ? 'Held out: question content and training competency mappings are excluded.' : 'Question-level competency review has not been done for this benchmark. Inventory patterns are provisional descriptions, not endpoint-specific competency assignments.'}</p>`}
    <h3>Questions ${benchmark.questions.length ? '(' + benchmark.questions.length + ' inventory records)' : ''}</h3>
    ${benchmark.questions.length ? `<div class="toolbar"><label for="question-search">Search</label><input id="question-search" type="search" placeholder="Question ID, operation or competency…">
      ${benchmark.review ? `<label for="question-competency">Competency</label><select id="question-competency"><option value="">All competencies</option>${benchmark.review.competencies.map(c => `<option value="${esc(c.id)}">${esc(c.name)}</option>`).join('')}</select>
      <label for="question-verification">Verification</label><select id="question-verification"><option value="">All dispositions</option>${Object.entries(verificationNames).map(([id, name]) => `<option value="${id}">${name}</option>`).join('')}</select>` : ''}</div>
      <p id="question-count" class="reference-note" aria-live="polite"></p><div id="question-table" class="panel table-scroll"></div>` : '<p>No eligible question inventory is available on this page.</p>'}`;
  if (benchmark.questions.length) {
    el('question-search').oninput = () => renderQuestionTable(benchmark);
    if (benchmark.review) {
      el('question-competency').onchange = () => renderQuestionTable(benchmark);
      el('question-verification').onchange = () => renderQuestionTable(benchmark);
      el('competency-count-mode').onchange = () => { renderCompetencyBars(benchmark); renderQuestionTable(benchmark); };
      renderCompetencyBars(benchmark);
    }
    renderQuestionTable(benchmark);
  }
}

function renderCompetencyBars(benchmark) {
  const rows = competencyCounts(benchmark, el('competency-count-mode').value);
  const max = Math.max(1, ...rows.map(c => c.count));
  el('competency-bars').innerHTML = rows.map(c => `<button class="competency-bar" data-competency="${esc(c.id)}" aria-label="${esc(c.name)}: ${c.count} questions; filter questions">
    <span>${esc(c.name)}</span><span class="bar-track"><span style="width:${100 * c.count / max}%"></span></span><b>${c.count}</b>
    <small>${c.focal} focal · ${c.total} total</small></button>`).join('');
  el('competency-bars').querySelectorAll('button').forEach(button => {
    button.onclick = () => {
      el('question-competency').value = button.dataset.competency;
      renderQuestionTable(benchmark);
      el('question-competency').focus();
    };
  });
}

function renderQuestionTable(benchmark) {
  const query = el('question-search').value.toLowerCase().trim();
  const competency = benchmark.review ? el('question-competency').value : '';
  const verification = benchmark.review ? el('question-verification').value : '';
  const names = new Map((benchmark.review?.competencies || []).map(c => [c.id, c.name]));
  const questions = benchmark.questions.filter(q =>
    (!competency || (el('competency-count-mode').value === 'focal' ? q.annotation?.focal_competencies || [] : questionLabels(q)).includes(competency)) &&
    (!verification || q.annotation?.verification_status === verification) &&
    [q.id, questionTitle(q), ...questionLabels(q).map(id => names.get(id))].join(' ').toLowerCase().includes(query));
  el('question-count').textContent = `${questions.length} of ${benchmark.questions.length} records shown`;
  el('question-table').innerHTML = `<table><thead><tr><th>Question</th><th>Focal competencies</th><th>Verification plan</th></tr></thead><tbody>${questions.map(q => `<tr>
    <td><a href="${questionHref(benchmark.id, q.id)}">${esc(q.id)}</a><br>${esc(questionTitle(q))}</td>
    <td>${q.annotation ? q.annotation.focal_competencies.map(id => esc(names.get(id))).join('<br>') : 'Not reviewed'}</td>
    <td>${q.annotation ? esc(verificationNames[q.annotation.verification_status]) : 'Not reviewed'}</td></tr>`).join('')}</tbody></table>`;
}

function renderQuestion(benchmark, question) {
  const annotation = question.annotation;
  const index = benchmark.questions.indexOf(question);
  const previous = benchmark.questions[index - 1], next = benchmark.questions[index + 1];
  const definitions = new Map((benchmark.review?.competencies || []).map(c => [c.id, c]));
  const cards = ids => ids.map(id => {
    const c = definitions.get(id);
    return `<div class="competency-card"><h3>${esc(c.name)}</h3><p>${esc(c.outcome)}</p></div>`;
  }).join('');
  el('question-page').innerHTML = `<div class="breadcrumbs"><a href="${routeHref('sources')}">Benchmarks</a> / <a href="${benchmarkHref(benchmark.id)}">${esc(benchmark.name)}</a> / ${esc(question.id)}</div>
    <h2 tabindex="-1">${esc(question.id)}</h2><p class="lede">${esc(questionTitle(question))}</p>
    <p class="reference-note">${annotation ? 'Paraphrased question summary; competency annotations are a draft.' : 'Inventory pattern only; the exact question endpoint and competencies still need review.'} <a href="${esc(question.source_url)}">Original source metadata</a></p>
    ${annotation ? `<div class="panel"><h3>Focal competencies</h3><div class="competency-cards">${cards(annotation.focal_competencies)}</div>
      <h3>Supporting competencies</h3><div class="competency-cards">${cards(annotation.supporting_competencies) || '<p>No additional supporting labels assigned at this granularity.</p>'}</div></div>
      <div class="panel verification-panel"><span class="badge">${esc(verificationNames[annotation.verification_status])}</span><h3>Executable reward design</h3><p>${esc(annotation.verification)}</p><h3>Decisions and limits</h3><p>${esc(annotation.decisions)}</p>
      <p class="reference-note">This is a proposed verification contract, not a passing implementation. Generated tasks use independent biological data, a reference solution and executable artifact checks. No LLM judge.</p></div>` : '<p class="notice">No question-level competency annotation yet. Shared workflow stages below are source context and may exceed this individual endpoint.</p>'}
    <details><summary>Inventory context and existing task mappings</summary>
      <p>Workflow family: <code>${esc(question.family)}</code>. Formats: ${esc(question.formats.join(', ') || 'Not inventoried')}.</p>
      <p>Inventory mapping status: <strong>${esc(question.status)}</strong>. This status is separate from the new draft competency labels.</p>
      <p>Related authored recipes: ${esc(question.recipes.join(', ') || 'None mapped')}.</p>
      ${question.recipes.length ? `<p><a href="${routeHref('examples')}">Inspect published generated examples</a></p>` : ''}
      <p class="reference-note">The inventory’s shared workflow stages may include upstream operations not required by this question. They are not automatically assigned as competencies.</p>${paragraphs(question.stages)}
      ${annotation ? `<p>Question fingerprint: <code class="hash-value">${esc(annotation.source_question_sha256)}</code></p>` : ''}</details>
    <div class="question-pagination">${previous ? `<a href="${questionHref(benchmark.id, previous.id)}">← ${esc(previous.id)}</a>` : '<span></span>'}
      <a href="${benchmarkHref(benchmark.id)}">All questions</a>${next ? `<a href="${questionHref(benchmark.id, next.id)}">${esc(next.id)} →</a>` : '<span></span>'}</div>`;
}

function routePage() {
  let parts;
  try {
    parts = location.hash.slice(1).split('/').map(decodeURIComponent);
  } catch {
    showPageError('The page link contains an invalid encoded identifier.');
    return;
  }
  const [kind, benchmarkId, questionId] = parts;
  if (kind === 'benchmark' || kind === 'question') {
    const benchmark = benchmarkById.get(benchmarkId);
    if (!benchmark) { showPageError('This benchmark is not in the current inventory.'); return; }
    if (kind === 'benchmark') {
      renderBenchmark(benchmark);
      tab('benchmark-page');
      document.title = benchmark.name + ' · Biology competencies';
    } else {
      const question = benchmark.questions.find(q => q.id === questionId);
      if (!question) { showPageError('This question is not in the eligible inventory.'); return; }
      renderQuestion(benchmark, question);
      tab('question-page');
      document.title = question.id + ' · ' + benchmark.name;
    }
    document.querySelector('.tab:not(.hidden) h2').focus({preventScroll: true});
    return;
  }
  const section = !kind ? 'coverage' : kind;
  if (!['coverage', 'plan', 'examples', 'sources', 'assets', 'method'].includes(section)) {
    showPageError('This page is not available.');
    return;
  }
  if (section === 'examples') renderExamples();
  tab(section);
  document.title = 'Biology tasks · Competency explorer';
}

function showPageError(message) {
  el('benchmark-page').innerHTML = `<h2>Page unavailable</h2><p>${esc(message)}</p><a href="${routeHref('sources')}">Browse benchmarks</a>`;
  tab('benchmark-page');
}

window.addEventListener('hashchange', routePage);
renderBenchmarkIndex();
routePage();
