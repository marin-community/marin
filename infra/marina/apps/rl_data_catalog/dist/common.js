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
