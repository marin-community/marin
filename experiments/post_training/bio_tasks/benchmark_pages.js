// Benchmark and question views share embedded, versioned inventories.
const benchmarkById = new Map(D.benchmarks.map(benchmark => [benchmark.id, benchmark]));
const routeHref = path => location.href.split('#')[0] + '#' + path;
const benchmarkHref = id => routeHref('benchmark/' + encodeURIComponent(id));
const questionHref = (benchmark, question) => routeHref('question/' + encodeURIComponent(benchmark) + '/' + encodeURIComponent(question));
const questionTitle = question => question.annotation?.question_summary || question.summary || question.id;
const verifiedQuestions = benchmark => benchmark.questions.filter(question => question.annotation?.competencies.length);
const setAsideQuestions = benchmark => benchmark.questions.filter(question => question.annotation && !question.annotation.competencies.length);

function benchmarkForSource(source) {
  return D.benchmarks.find(benchmark => source.task_inventory ? benchmark.inventory === source.task_inventory : benchmark.id === source.name);
}

function renderBenchmarkIndex() {
  const first = benchmarkById.get('BixBench-Verified-50');
  el('benchmark-entry').innerHTML = `<h3>Question-level review</h3>
    <p><a href="${benchmarkHref(first.id)}">BixBench-Verified-50: questions and competencies</a></p>
    <p>Each eligible benchmark has an inventory page and each inventoried question has a permalink. BixBench-Verified currently has competency annotations. OOD question content remains excluded.</p>`;
}

function competencyCounts(benchmark) {
  return benchmark.review.competencies.map(competency => ({
    ...competency,
    count: verifiedQuestions(benchmark).filter(question => question.annotation.competencies.includes(competency.id)).length,
  })).sort((first, second) => second.count - first.count || first.name.localeCompare(second.name));
}

function renderBenchmark(benchmark) {
  const reviewed = verifiedQuestions(benchmark);
  const excluded = setAsideQuestions(benchmark);
  el('benchmark-page').innerHTML = `<div class="breadcrumbs"><a href="${routeHref('sources')}">Benchmarks</a> / ${esc(benchmark.name)}</div>
    <h2 tabindex="-1">${esc(benchmark.name)}</h2>
    <p class="page-meta">${esc(benchmark.distribution)} · <a href="${esc(benchmark.source_url)}">Source</a>
    ${benchmark.inventory ? ' · ' + link('experiments/post_training/bio_tasks/' + benchmark.inventory, 'Inventory') : ''}
    ${benchmark.review_file ? ' · ' + link('experiments/post_training/bio_tasks/' + benchmark.review_file, 'Annotations') : ''}</p>
    ${benchmark.review ? `<p>${reviewed.length} questions with proposed executable checks · ${benchmark.review.competencies.length} draft competencies. Click a bar to see its questions.</p>
      <div id="competency-bars" class="competency-bars" aria-label="Competencies ranked by number of questions"></div>` :
      `<p>${benchmark.distribution === 'OOD' ? 'Held out: question content and training mappings are excluded.' :
        'Question-level competency review is pending. Inventory patterns below are provisional.'}</p>`}
    <h3>${benchmark.review ? 'Questions with proposed executable checks' : 'Inventoried questions'} <span class="count-muted">(${benchmark.review ? reviewed.length : benchmark.questions.length})</span></h3>
    ${benchmark.questions.length ? `<div class="toolbar"><label for="question-search">Search questions</label><input id="question-search" type="search" placeholder="ID, question or competency"></div>
      <p id="question-count" class="reference-note" aria-live="polite"></p><div id="question-table" class="table-scroll"></div>` : '<p>No eligible questions are inventoried for this benchmark.</p>'}
    ${excluded.length ? `<details class="set-aside"><summary>${excluded.length} questions set aside for now</summary><p>These need a clearer scientific contract or ask for open-ended interpretation. They have no competency labels or frequency credit.</p>
      <div class="table-scroll"><table><thead><tr><th>Question</th><th>Why set aside</th></tr></thead><tbody>${excluded.map(question =>
        `<tr><td><a href="${questionHref(benchmark.id, question.id)}">${esc(question.id)}</a><br>${esc(questionTitle(question))}</td>
        <td>${esc(question.annotation.decisions)}</td></tr>`).join('')}</tbody></table></div></details>` : ''}`;
  if (benchmark.review) renderCompetencyBars(benchmark);
  if (benchmark.questions.length) {
    el('question-search').oninput = () => renderQuestionTable(benchmark);
    renderQuestionTable(benchmark);
  }
}

function renderCompetencyBars(benchmark) {
  const counts = competencyCounts(benchmark);
  const max = Math.max(1, ...counts.map(competency => competency.count));
  el('competency-bars').innerHTML = counts.map(competency =>
    `<button class="competency-bar" data-competency="${esc(competency.id)}" aria-label="${esc(competency.name)}: ${competency.count} questions; filter questions">
      <span>${esc(competency.name)}</span><span class="bar-track"><span style="width:${100 * competency.count / max}%"></span></span><b>${competency.count}</b></button>`).join('');
  el('competency-bars').querySelectorAll('button').forEach(button => {
    button.onclick = () => {
      const selected = el('competency-bars').dataset.selected;
      el('competency-bars').dataset.selected = selected === button.dataset.competency ? '' : button.dataset.competency;
      el('competency-bars').querySelectorAll('button').forEach(bar =>
        bar.setAttribute('aria-pressed', String(bar.dataset.competency === el('competency-bars').dataset.selected)));
      renderQuestionTable(benchmark);
      el('question-table').scrollIntoView({behavior: 'smooth', block: 'start'});
    };
  });
}

function renderQuestionTable(benchmark) {
  const query = el('question-search').value.toLowerCase().trim();
  const selected = benchmark.review ? el('competency-bars').dataset.selected : '';
  const definitions = new Map((benchmark.review?.competencies || []).map(competency => [competency.id, competency.name]));
  const candidates = benchmark.review ? verifiedQuestions(benchmark) : benchmark.questions;
  const questions = candidates.filter(question =>
    (!selected || question.annotation?.competencies.includes(selected)) &&
    [question.id, questionTitle(question), ...(question.annotation?.competencies || []).map(id => definitions.get(id))]
      .join(' ').toLowerCase().includes(query));
  el('question-count').textContent = `${questions.length} of ${candidates.length} shown`;
  el('question-table').innerHTML = `<table><thead><tr><th>Question</th><th>Competencies</th></tr></thead><tbody>${questions.map(question =>
    `<tr><td><a href="${questionHref(benchmark.id, question.id)}">${esc(question.id)}</a><br>${esc(questionTitle(question))}</td>
    <td>${question.annotation ? question.annotation.competencies.map(id => esc(definitions.get(id))).join('<br>') : 'Not reviewed'}</td></tr>`).join('')}</tbody></table>`;
}

function renderQuestion(benchmark, question) {
  const annotation = question.annotation;
  const included = Boolean(annotation?.competencies.length);
  const definitions = new Map((benchmark.review?.competencies || []).map(competency => [competency.id, competency]));
  const peers = included ? verifiedQuestions(benchmark) : setAsideQuestions(benchmark);
  const index = peers.indexOf(question);
  const previous = peers[index - 1], next = peers[index + 1];
  el('question-page').innerHTML = `<div class="breadcrumbs"><a href="${routeHref('sources')}">Benchmarks</a> / <a href="${benchmarkHref(benchmark.id)}">${esc(benchmark.name)}</a> / ${esc(question.id)}</div>
    <h2 tabindex="-1">${esc(question.id)}</h2>
    ${annotation ? `<div class="source-question"><div class="eyebrow">Original question</div><p>${esc(annotation.source_question)}</p></div>
      ${included ? `<h3>Competencies</h3><div class="competency-cards">${annotation.competencies.map(id => {
        const competency = definitions.get(id);
        return `<div class="competency-card"><h4>${esc(competency.name)}</h4><p>${esc(competency.outcome)}</p></div>`;
      }).join('')}</div>
      <h3>Executable check to design</h3><p>${esc(annotation.verification)}</p><p class="reference-note">${esc(annotation.decisions)}</p>` :
      `<p class="notice">Set aside for now. No competencies are assigned.</p><p>${esc(annotation.decisions)}</p>`}` :
      `<p>${esc(questionTitle(question))}</p><p class="notice">Question-level competency review is pending for this benchmark.</p>`}
    <details><summary>Source and inventory details</summary><p><a href="${esc(question.source_url)}">Source metadata</a>
      ${benchmark.inventory ? ' · ' + link('experiments/post_training/bio_tasks/' + benchmark.inventory, 'Inventory') : ''}</p>
      <p>Workflow family: ${esc(question.family)}. Inventory mapping status: ${esc(question.status)}.</p>
      <p>Related authored recipes: ${esc(question.recipes.join(', ') || 'None mapped')}.</p>
      ${annotation ? `<p>Question SHA-256: <code class="hash-value">${esc(annotation.source_question_sha256)}</code></p>` : ''}</details>
    <div class="question-pagination">${previous ? `<a href="${questionHref(benchmark.id, previous.id)}">← ${esc(previous.id)}</a>` : '<span></span>'}
      <a href="${benchmarkHref(benchmark.id)}">All questions</a>${next ? `<a href="${questionHref(benchmark.id, next.id)}">${esc(next.id)} →</a>` : '<span></span>'}</div>`;
}

function routePage() {
  let parts;
  try {
    parts = location.hash.slice(1).split('/').map(decodeURIComponent);
  } catch {
    showPageError('The page link contains an invalid identifier.');
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
      const question = benchmark.questions.find(item => item.id === questionId);
      if (!question) { showPageError('This question is not in the eligible inventory.'); return; }
      renderQuestion(benchmark, question);
      tab('question-page');
      document.title = question.id + ' · ' + benchmark.name;
    }
    document.body.classList.add('focus-view');
    document.querySelector('.tab:not(.hidden) h2').focus({preventScroll: true});
    return;
  }
  document.body.classList.remove('focus-view');
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
  document.body.classList.add('focus-view');
}

window.addEventListener('hashchange', routePage);
renderBenchmarkIndex();
routePage();
