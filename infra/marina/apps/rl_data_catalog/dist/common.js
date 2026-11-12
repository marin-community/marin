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
          content.replaceChildren(node('p', data.judgment.explanation), node('pre', JSON.stringify(data, null, 2)));
        } catch (error) { content.replaceChildren(node('p', error.message, 'error')); }
      });
      group.append(details);
    }
    section.append(group);
  }
  return section;
}
