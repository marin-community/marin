/* Copyright The Marin Authors. SPDX-License-Identifier: Apache-2.0 */
async function render(sourceId) {
  const params = new URLSearchParams({source_id: sourceId});
  const [source, survey] = await Promise.all([json(`api/source?${params}`), json('api/backends')]);
  if (source.catalog.root !== survey.catalog_root || source.catalog.snapshot_id !== survey.snapshot_id) {
    throw new Error('The source and backend survey belong to different catalog snapshots. Reload this page.');
  }
  const declaration = source.declaration, census = source.backend_capabilities;
  document.title = `${declaration.display_name} · Backends · RL Data Atlas`;
  $('title').textContent = declaration.display_name;
  $('message').textContent = `${declaration.id} · catalog snapshot ${source.catalog.snapshot_id}`;
  $('quality-evidence').href = `source.html#${params}`;
  $('fit').replaceChildren(backendDetails(source));
  const inputs = node('ul', undefined, 'record-list');
  for (const [uri, digest] of Object.entries(census?.inputs || {})) {
    const item = node('li');
    item.append(/^(https|s3|gs):/.test(uri) ? receiptLink(uri, uri) : node('code', uri), ' · ', node('code', digest)); inputs.append(item);
  }
  const sections = {
    census: [
      ['Measurement basis', census?.basis ?? 'Not measured'],
      ['Unassessed reason', census?.reason],
      ['Measured tasks', census?.basis === 'converted_spec_census' ? census.task_count : 'Not measured'],
      ['Current dataset', declaration.current_release && receiptLink(declaration.current_release.url, `${declaration.current_release.repo} @ ${declaration.current_release.revision}`)],
      ['Converter version', declaration.pipeline?.version ?? 'Not recorded'],
      ['Task inputs and SHA-256', inputs.children.length ? inputs : 'Not measured'],
      ['Role assignments', census?.role_backend_assignments],
      ['Declared requirements by role', census?.demands],
      ['Resource ranges by role', census?.resource_ranges],
    ],
    provenance: [
      ['Survey ID', survey.survey_id], ['Measured at', survey.generated_at],
      ['Code revision', survey.git_revision], ['Requirement schema SHA-256', survey.schema_sha256],
      ['Catalog root', survey.catalog_root], ['Catalog snapshot', survey.snapshot_id],
      ['Runtime defaults', survey.runtime_defaults], ['Deployment observations', survey.deployment_limits],
      ['Verified LOCAL disk pool', survey.local_disk_pool ?? 'Not recorded'],
    ],
  };
  for (const [id, rows] of Object.entries(sections)) {
    $(id).replaceChildren(...rows.filter(([, value]) => value !== null && value !== undefined).map(([label, value]) => {
      const field = node('dl', undefined, 'field'), item = node('dd');
      item.append(readableRecord(value)); field.append(node('dt', label), item); return field;
    }));
  }
}
const sourceId = new URLSearchParams(location.hash.slice(1)).get('source_id');
window.addEventListener('hashchange', () => location.reload());
if (!sourceId) { $('title').textContent = 'Backend evidence'; $('message').textContent = 'Open Backends from a source in the Atlas.'; }
else render(sourceId).catch(error => {
  $('title').textContent = 'Backend evidence'; $('message').textContent = error.message; $('message').classList.add('error');
});
