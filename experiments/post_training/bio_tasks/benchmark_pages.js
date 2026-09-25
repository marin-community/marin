// Two independent facets describe requirements; workflow contracts guide authoring.
const benchmarkById = new Map(D.benchmarks.map(benchmark => [benchmark.id, benchmark]));
const skillById = new Map(D.review_taxonomy.skills.map(skill => [skill.id, skill]));
const applicationById = new Map(D.review_taxonomy.applications.map(application => [application.id, application]));
const routeHref = path => location.href.split('#')[0] + '#' + path;
const benchmarkHref = id => routeHref('benchmark/' + encodeURIComponent(id));
const questionHref = (benchmark, question) => routeHref('question/' + encodeURIComponent(benchmark) + '/' + encodeURIComponent(question));
const includedQuestions = benchmark => benchmark.questions.filter(question => question.annotation?.disposition === 'reframe');
const questionTitle = question => question.annotation?.source_question.replace(/\s+/g, ' ').slice(0, 200) || question.summary || question.id;
const listItems = values => '<ul>' + values.map(value => '<li>' + esc(value) + '</li>').join('') + '</ul>';

function benchmarkForSource(source) {
  return D.benchmarks.find(benchmark => source.task_inventory ? benchmark.inventory === source.task_inventory : benchmark.id === source.name);
}

function renderBenchmarkIndex() {
  el('benchmark-entry').innerHTML = `<p><a href="${routeHref('benchmarks')}">Browse benchmark questions, skills and proposed executable checks</a></p>
    <p><a href="${routeHref('statistics')}">Global category statistics across all inventoried ID releases</a></p><p>Includes straightforward reframings of source tasks scored by an LLM. These are authoring proposals; missing source instructions and questions without a faithful executable framing are listed separately.</p>`;
}

function renderGlobalStatistics(facet = 'skills', categoryId = '') {
  if (!['skills', 'applications'].includes(facet)) { showPageError('Unknown category facet.'); return; }
  const statistics = D.category_statistics;
  const counts = statistics.facets[facet];
  const selected = categoryId ? counts.find(row => row.id === categoryId) : null;
  const definition = categoryId ? D.review_taxonomy[facet].find(row => row.id === categoryId) : null;
  if (categoryId && !selected) { showPageError('Unknown category.'); return; }
  const total = statistics.eligible_records;
  const max = Math.max(1, ...counts.map(row => row.count));
  const pct = count => total ? (100 * count / total).toFixed(1) + '%' : '0%';
  el('benchmark-page').innerHTML = `<div class="breadcrumbs"><a href="${routeHref('benchmarks')}">Benchmarks</a> / <a href="${routeHref('statistics/' + facet)}">Global categories</a>${selected ? ' / ' + esc(selected.name) : ''}</div>
    <h2 tabindex="-1">${selected ? esc(selected.name) : 'Categories across ID benchmarks'}</h2>
    <p>${total.toLocaleString()} candidate verifiable records · ${statistics.inventoried_releases} inventoried ID releases. Each record counts once per assigned category; categories overlap.</p>
    <p class="reference-note">Provisional annotations. Raw counts include overlapping releases and repeated protocols; they measure requirements, not validated tasks or independent scientific demand.</p>
    ${facet === 'applications' ? `<p class="reference-note">${statistics.unresolved_applications} eligible records have no resolved application and remain in the denominator.</p>` : ''}
    ${selected ? `<p><strong>${selected.count.toLocaleString()}</strong> records (${pct(selected.count)} of eligible records) across <strong>${selected.benchmarks}</strong> ID releases.</p>
      <p>${esc(definition.outcome || definition.scope)}</p>${definition.include ? `<p><strong>Include:</strong> ${esc(definition.include)}</p><p><strong>Exclude:</strong> ${esc(definition.exclude)}</p>` : ''}
      <div class="table-scroll"><table><thead><tr><th>Benchmark</th><th>Records with category</th><th>Eligible records</th><th>Within-benchmark share</th></tr></thead><tbody>${selected.sources.map(source => `<tr><td><a href="${benchmarkHref(source.benchmark)}">${esc(source.benchmark)}</a></td><td>${source.count}</td><td>${source.eligible}</td><td>${(100 * source.count / source.eligible).toFixed(1)}%</td></tr>`).join('')}</tbody></table></div>` :
      `<div class="toolbar"><label for="global-facet">Categories</label><select id="global-facet"><option value="skills" ${facet === 'skills' ? 'selected' : ''}>Analytical skills</option><option value="applications" ${facet === 'applications' ? 'selected' : ''}>Biological applications</option></select>
        <label for="global-category-search">Search</label><input id="global-category-search" type="search"></div>
      <p class="reference-note">Ranked by question-record count. Select a category to inspect its benchmark contributions.</p><div id="global-category-table" class="table-scroll"></div>`}
    <details class="set-aside"><summary>${statistics.missing_inventories.length} ID sources without task inventories</summary><p>These sources are absent from the numerical denominator. Their unknown counts are not evidence of zero demand.</p>${listItems(statistics.missing_inventories)}</details>
    <details><summary>Assignment rules and review scope</summary>${listItems(D.review_taxonomy.annotation_policy)}<p>${statistics.operation_reviewed_records.toLocaleString()} records have a source-protocol or output-contract review of operation boundaries. This does not certify individual verifier feasibility. Other assignments remain first-pass drafts.</p></details>
    <p class="reference-note">${link('docs/experiments/bio-benchmark-categories.md', 'Rankings as Markdown')}</p>`;
  if (!selected) {
    el('global-facet').onchange = () => { location.hash = 'statistics/' + el('global-facet').value; };
    const render = () => {
      const query = el('global-category-search').value.toLowerCase().trim();
      el('global-category-table').innerHTML = `<table><thead><tr><th>Category</th><th>Question records</th><th>% of records</th><th>ID releases</th></tr></thead><tbody>${counts.filter(row => row.name.toLowerCase().includes(query)).map(row => `<tr><td><a href="${routeHref('statistics/' + facet + '/' + row.id)}">${esc(row.name)}</a></td><td><div class="count-bar"><span class="bar-track"><span style="width:${100 * row.count / max}%"></span></span><b>${row.count.toLocaleString()}</b></div></td><td>${pct(row.count)}</td><td>${row.benchmarks}</td></tr>`).join('')}</tbody></table>`;
    };
    el('global-category-search').oninput = render;
    render();
  }
}

function renderBenchmarkDirectory() {
  const benchmarks = D.benchmarks.filter(benchmark => benchmark.distribution === 'ID');
  const reviewed = benchmarks.filter(benchmark => benchmark.review);
  const included = reviewed.flatMap(includedQuestions);
  el('benchmark-page').innerHTML = `<div class="breadcrumbs"><a href="${routeHref('statistics')}">Categories</a> / Benchmarks</div>
    <h2 tabindex="-1">Benchmark questions</h2><p><a href="${routeHref('statistics')}">Global category rankings and benchmark contributions →</a></p>
    <p>${reviewed.length} inventoried ID releases · ${included.length.toLocaleString()} proposed verifiable framings. Skill labels are a provisional review aid, not validation credit. Source releases and shared protocols overlap.</p>
    <details><summary>How to use the categories</summary><p>A skill names an analysis operation; an application names its biological setting. Plan connected workflows from missing operations and decisions with executable outputs. Frequency is a signal, not a generation quota.</p>
    <p>Many assignments are rule-assisted drafts. The question page identifies their basis and preserves the original instruction for correction. A proposed verifier does not establish that the whole source question is already solved or validated.</p></details>
    <div class="toolbar"><label for="benchmark-search">Search benchmarks</label><input id="benchmark-search" type="search"></div>
    <div id="benchmark-list" class="table-scroll"></div>
    <details class="set-aside"><summary>OOD benchmarks · excluded</summary><p>Their question content is not used for category assignments.</p><ul>${D.benchmarks.filter(b => b.distribution === 'OOD').map(b => `<li><a href="${benchmarkHref(b.id)}">${esc(b.name)}</a></li>`).join('')}</ul></details>`;
  const render = () => {
    const query = el('benchmark-search').value.toLowerCase().trim();
    el('benchmark-list').innerHTML = `<table><thead><tr><th>Benchmark</th><th>Proposed</th><th>Set aside</th><th>Missing instructions</th></tr></thead><tbody>${benchmarks.filter(b => b.name.toLowerCase().includes(query)).map(b => {
      const rows = b.questions;
      return `<tr><td><a href="${benchmarkHref(b.id)}">${esc(b.name)}</a>${!b.review ? '<br><small>No question inventory available</small>' : ''}</td>
        <td>${includedQuestions(b).length}</td><td>${rows.filter(q => q.annotation?.disposition === 'excluded').length}</td>
        <td>${b.review ? rows.filter(q => q.annotation?.disposition === 'unavailable').length : 'Unknown'}</td></tr>`;
    }).join('')}</tbody></table>`;
  };
  el('benchmark-search').oninput = render;
  render();

}

function facetCounts(questions, facet) {
  return D.review_taxonomy[facet].map(definition => ({
    ...definition,
    count: questions.filter(question => question.annotation[facet].includes(definition.id)).length,
  })).filter(row => row.count).sort((a, b) => b.count - a.count || a.name.localeCompare(b.name));
}

function renderBenchmark(benchmark) {
  const included = includedQuestions(benchmark);
  const excluded = benchmark.questions.filter(question => question.annotation?.disposition === 'excluded');
  const unavailable = benchmark.questions.filter(question => question.annotation?.disposition === 'unavailable');
  el('benchmark-page').innerHTML = `<div class="breadcrumbs"><a href="${routeHref('benchmarks')}">Benchmarks</a> / ${esc(benchmark.name)}</div>
    <h2 tabindex="-1">${esc(benchmark.name)}</h2>
    <p class="page-meta">${esc(benchmark.distribution)} · <a href="${esc(benchmark.source_url)}">Source</a>
    ${benchmark.inventory ? ' · ' + link('experiments/post_training/bio_tasks/' + benchmark.inventory, 'Inventory') : ''}
    ${benchmark.review_file ? ' · ' + link('experiments/post_training/bio_tasks/' + benchmark.review_file, 'Annotations') : ''}</p>
    ${benchmark.review ? `<p>${included.length} proposed verifiable framings out of ${benchmark.questions.length} source records. Draft labels; counts overlap. Click a bar to filter questions.</p>
      <div class="toolbar"><label for="facet-select">Rank by</label><select id="facet-select"><option value="skills">Analytical skill</option><option value="applications">Biological application</option></select></div>
      <div id="competency-bars" class="competency-bars"></div>` :
      `<p>${benchmark.distribution === 'OOD' ? 'Held out: question content and training mappings are excluded.' : 'Question inventory unavailable.'}</p><p>${esc(benchmark.inspection)}</p>`}
    ${benchmark.review ? `<h3>Questions with proposed executable checks <span class="count-muted">(${included.length})</span>
      </h3><div class="toolbar"><label for="question-search">Search</label><input id="question-search" type="search" placeholder="Question, skill or application"></div>
      <p id="question-count" class="reference-note" aria-live="polite"></p><div id="question-table" class="table-scroll"></div>` : ''}
    ${excluded.length ? `<details class="set-aside"><summary>${excluded.length} questions set aside</summary><p>No faithful executable framing established. No skill labels or frequency credit.</p>${asideTable(benchmark, excluded)}</details>` : ''}
    ${unavailable.length ? `<details class="set-aside"><summary>${unavailable.length} original instructions unavailable</summary>${asideTable(benchmark, unavailable)}</details>` : ''}
    ${benchmark.review ? `<details class="set-aside"><summary>Source, review method and counting</summary><p>${esc(benchmark.inspection)}</p><p>${esc(benchmark.review.method)}</p><p>${esc(benchmark.review.counting)}</p><p>Shared instructions are labeled explicitly. A source record can be a question, dataset instance or protocol; it is not necessarily an independent workflow.</p></details>` : ''}`;
  if (benchmark.review) {
    el('facet-select').onchange = () => { renderFacetBars(benchmark); renderQuestionTable(benchmark); };
    el('question-search').oninput = () => renderQuestionTable(benchmark);
    renderFacetBars(benchmark);
    renderQuestionTable(benchmark);
  }
}

function asideTable(benchmark, questions) {
  return `<div class="table-scroll"><table><thead><tr><th>Question</th><th>Reason</th></tr></thead><tbody>${questions.map(question =>
    `<tr><td><a href="${questionHref(benchmark.id, question.id)}">${esc(question.id)}</a><br>${esc(questionTitle(question))}</td><td>${esc(question.annotation.reason)}</td></tr>`).join('')}</tbody></table></div>`;
}

function renderFacetBars(benchmark) {
  const counts = facetCounts(includedQuestions(benchmark), el('facet-select').value);
  const max = Math.max(1, ...counts.map(row => row.count));
  el('competency-bars').dataset.selected = '';
  el('competency-bars').innerHTML = counts.length ? counts.map(row =>
    `<button class="competency-bar" data-facet-id="${esc(row.id)}" aria-pressed="false" aria-label="${esc(row.name)}: ${row.count} questions; filter questions">
      <span>${esc(row.name)}</span><span class="bar-track"><span style="width:${100 * row.count / max}%"></span></span><b>${row.count}</b></button>`).join('') : '<p>No eligible assignments.</p>';
  el('competency-bars').querySelectorAll('button').forEach(button => {
    button.onclick = () => {
      const selected = el('competency-bars').dataset.selected;
      el('competency-bars').dataset.selected = selected === button.dataset.facetId ? '' : button.dataset.facetId;
      el('competency-bars').querySelectorAll('button').forEach(bar => bar.setAttribute('aria-pressed', String(bar.dataset.facetId === el('competency-bars').dataset.selected)));
      renderQuestionTable(benchmark);
    };
  });
}

function renderQuestionTable(benchmark) {
  const query = el('question-search').value.toLowerCase().trim();
  const facet = el('facet-select').value;
  const selected = el('competency-bars').dataset.selected;
  const candidates = includedQuestions(benchmark);
  const questions = candidates.filter(question => {
    const annotation = question.annotation;
    const text = [question.id, annotation.source_question, ...annotation.skills.map(id => skillById.get(id).name), ...annotation.applications.map(id => applicationById.get(id).name)].join(' ').toLowerCase();
    return (!selected || annotation[facet].includes(selected)) && text.includes(query);
  });
  el('question-count').textContent = `${questions.length} of ${candidates.length} shown${selected ? ' · Click the selected bar to clear the filter' : ''}`;
  el('question-table').innerHTML = `<table><thead><tr><th>Question</th><th>Analytical skills</th><th>Applications</th></tr></thead><tbody>${questions.map(question =>
    `<tr><td><a href="${questionHref(benchmark.id, question.id)}">${esc(question.id)}</a><br>${esc(questionTitle(question))}</td>
    <td>${question.annotation.skills.map(id => esc(skillById.get(id).name)).join('<br>')}</td>
    <td>${question.annotation.applications.map(id => esc(applicationById.get(id).name)).join('<br>') || 'Application not resolved'}</td></tr>`).join('')}</tbody></table>`;
}

function renderQuestion(benchmark, question) {
  const a = question.annotation;
  const included = a?.disposition === 'reframe';
  const peers = benchmark.questions.filter(q => q.annotation?.disposition === a?.disposition);
  const index = peers.indexOf(question);
  const previous = peers[index - 1], next = peers[index + 1];
  const shared = a?.source_question_kind.startsWith('shared-');
  const stem = a?.source_question_kind.includes('stem;');
  el('question-page').innerHTML = `<div class="breadcrumbs"><a href="${routeHref('benchmarks')}">Benchmarks</a> / <a href="${benchmarkHref(benchmark.id)}">${esc(benchmark.name)}</a></div>
    <h2 tabindex="-1" title="${esc(question.id)}">${esc(question.id)}</h2>
    ${a?.source_question ? `<div class="source-question"><div class="eyebrow">${shared ? 'Original shared protocol / instruction template' : stem ? 'Original question stem · answer options not retained' : 'Original question'}</div><p>${esc(a.source_question)}</p></div>` : '<p>Original instruction unavailable in the inspected release.</p>'}
    ${included ? `<h3>Proposed verifiable framing</h3><p>${esc(a.reframed_question)}</p>
      ${a.scope_changes.length ? `<div class="scope-note"><h4>Scope of the adaptation</h4>${listItems(a.scope_changes)}</div>` : ''}
      <h3>Analytical skills</h3><p class="reference-note">${esc(a.annotation_basis)} · Required operations, not demonstrated competence.</p>
      <div class="competency-cards">${a.skills.map(id => {const skill = skillById.get(id);return `<div class="competency-card"><h4><a href="${routeHref('statistics/skills/' + id)}">${esc(skill.name)}</a></h4><p>${esc(skill.outcome)}</p></div>`;}).join('')}</div>
      ${a.category_review ? `<details><summary>Why these categories</summary><p>${esc(a.category_review.note)}</p><p class="reference-note">Review scope: ${esc(a.category_review.scope)}. Category review does not certify the proposed verifier.</p></details>` : ''}
      <p><strong>Biological applications:</strong> ${a.applications.map(id => esc(applicationById.get(id).name)).join(' · ') || 'Application not resolved'}${a.context_tags.length ? '<br><strong>Context:</strong> ' + esc(a.context_tags.join(' · ')) : ''}</p>
      <h3>Outputs to check</h3>${listItems(a.outputs)}<h3>Executable reward</h3><p>${esc(a.verification)}</p>
      ${a.decisions.length ? `<details><summary>Scientific decisions to specify</summary>${listItems(a.decisions)}</details>` : ''}
      <details><summary>What authoring still requires</summary>${listItems(a.authoring_requirements)}<p>This page proposes a contract. It does not certify a runnable task or passing oracle.</p></details>` :
      `<p class="notice">${a?.disposition === 'unavailable' ? 'Source access gap' : 'Set aside'} · No skill labels assigned.</p><p>${esc(a?.reason || benchmark.inspection)}</p>`}
    <details class="set-aside"><summary>Source and inventory details</summary><p><a href="${esc(a?.source_url || question.source_url)}">Original source</a>
      ${benchmark.review_file ? ' · ' + link('experiments/post_training/bio_tasks/' + benchmark.review_file, 'Editable annotations') : ''}</p>
      <p>Workflow: ${esc(a?.workflow || question.family)}.</p>
      ${a?.source_group ? '<p>Source group: ' + esc(a.source_group) + '.</p>' : ''}
      ${a?.source_question_sha256 ? `<p>Instruction SHA-256: <code class="hash-value">${esc(a.source_question_sha256)}</code></p>` : ''}</details>
    <div class="question-pagination">${previous ? `<a href="${questionHref(benchmark.id, previous.id)}">← Previous</a>` : '<span></span>'}
      <a href="${benchmarkHref(benchmark.id)}">All questions</a>${next ? `<a href="${questionHref(benchmark.id, next.id)}">Next →</a>` : '<span></span>'}</div>`;
}

function renderBenchmarkRoute(parts) {
  const [kind, benchmarkId, questionId] = parts;
  if (kind === 'statistics') {
    renderGlobalStatistics(benchmarkId || 'skills', questionId || '');
    tab('benchmark-page');
    document.title = 'Global categories · Biology benchmarks';
  } else if (kind === 'benchmarks') {
    renderBenchmarkDirectory();
    tab('benchmark-page');
    document.title = 'Benchmark questions · Biology benchmarks';
  } else {
    const benchmark = benchmarkById.get(benchmarkId);
    if (!benchmark) { showPageError('This benchmark is not in the current inventory.'); return; }
    if (kind === 'benchmark') {
      renderBenchmark(benchmark);
      tab('benchmark-page');
      document.title = benchmark.name + ' · Biology benchmarks';
    } else {
      const question = benchmark.questions.find(item => item.id === questionId);
      if (!question) { showPageError('This question is not in the eligible inventory.'); return; }
      renderQuestion(benchmark, question);
      tab('question-page');
      document.title = question.id + ' · ' + benchmark.name;
    }
  }
  document.body.classList.add('focus-view');
  document.querySelector('.tab:not(.hidden) h2').focus({preventScroll: true});
  window.scrollTo(0, 0);
}

function showPageError(message) {
  el('benchmark-page').innerHTML = `<h2>Page unavailable</h2><p>${esc(message)}</p><a href="${routeHref('benchmarks')}">Browse benchmarks</a>`;
  tab('benchmark-page');
  document.body.classList.add('focus-view');
}
