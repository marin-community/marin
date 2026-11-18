/* Copyright The Marin Authors. SPDX-License-Identifier: Apache-2.0 */
(async () => {
  const parameters = new URLSearchParams(location.search);
  const sourceId = parameters.get('source'), task = parameters.get('task');
  try {
    const source = await json(`api/source?${new URLSearchParams({source_id: sourceId})}`);
    const back = node('a', '← Task reviews');
    back.href = `assessment.html?source=${encodeURIComponent(sourceId)}`;
    $('content').append(back);
    if (source.data_restriction === 'private_always' || source.declaration.data_restriction === 'private_always') {
      $('message').textContent = 'Private data';
      return;
    }
    const data = await json(`api/evidence?${new URLSearchParams({source_id: sourceId, task_id: task})}`);
    if (data.private || data.data_restriction === 'private_always') {
      $('message').textContent = 'Private data';
      return;
    }
    $('title').textContent = `Raw QC review · ${task}`;
    $('message').textContent = `${source.declaration.display_name} · ${data.judgment.verdict}`;
    const labels = node('div', undefined, 'evidence-labels');
    labels.append(...(data.judgment.defects || []).map(defect => defectChip(defect)));
    $('content').append(labels);
    if (data.raw_review?.unavailable) $('content').append(node('p', `${data.raw_review.provenance?.mapping_status === 'not_mapped' ? 'Raw review not mapped' : 'Raw messages unavailable'}: ${data.raw_review.unavailable}`, 'muted'));
    const fields = [
      ['Stored task judgment', data.raw_review?.typed_observation ?? data.judgment],
      [data.raw_review ? 'Reviewer output' : 'Stored normalized judgment', data.raw_review?.output ?? data.judgment],
      ['Evidence the reviewer saw', data.raw_review?.input ?? data.packet],
      ['Provenance', data.raw_review?.provenance ?? null],
    ];
    for (const [label, value] of fields) {
      const section = node('section', undefined, 'assessment-card');
      section.append(node('h2', label), value == null
        ? node('p', 'Not present in the loaded review evidence.', 'muted')
        : node('pre', JSON.stringify(value, null, 2)));
      $('content').append(section);
    }
  } catch (error) { $('message').textContent = error.message; $('message').className = 'error'; }
})();
