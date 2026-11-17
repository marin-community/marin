/* Copyright The Marin Authors. SPDX-License-Identifier: Apache-2.0 */
(async () => {
  try {
    const data = await json('api/qc/defects');
    $('message').textContent = `${data.sources.toLocaleString()} sources with loaded task reviews. A task can have several defects; counts overlap.`;
    for (const defect of data.defects) {
      const section = node('section', undefined, 'assessment-card');
      section.id = defect.id;
      section.append(node('h2', `${QC_DEFECT_LABELS[defect.id]} · ${defect.count.toLocaleString()} reviews`),
        node('code', defect.id), node('p', defect.description));
      const sources = node('ul');
      for (const source of defect.sources) {
        const item = node('li'), anchor = node('a', `${source.name} · ${source.count.toLocaleString()}`);
        anchor.href = `assessment.html?${new URLSearchParams({source: source.id, defect: defect.id})}`;
        item.append(anchor); sources.append(item);
      }
      section.append(sources); $('content').append(section);
    }
    if (location.hash) document.getElementById(decodeURIComponent(location.hash.slice(1)))?.scrollIntoView();
  } catch (error) { $('message').textContent = error.message; $('message').className = 'error'; }
})();
