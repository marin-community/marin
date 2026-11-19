/* Copyright The Marin Authors. SPDX-License-Identifier: Apache-2.0 */
(async () => {
  const parameters = new URLSearchParams(location.search), source = parameters.get('source'), measurement = parameters.get('measurement');
  try {
    const evidence = await json(`api/source?${new URLSearchParams({source_id: source})}`);
    if (contentWithheld(evidence)) {
      $('message').textContent = contentWithheldMessage(evidence); return;
    }
    const data = await json(`api/difficulty-rollout?${new URLSearchParams({source_id: source, measurement_id: measurement, attempt_id: parameters.get('attempt')})}`);
    if (contentWithheld(data)) { $('message').textContent = contentWithheldMessage(data); return; }
    $('title').textContent = `Rollout · ${data.summary.task_id}`;
    $('message').textContent = `${data.summary.solver} · score ${data.summary.score ?? 'unscored'} · ${data.summary.status} · trial ${data.summary.trial_id}`;
    const back = node('a', '← Attempts'); back.href = `attempts.html?${new URLSearchParams({source, measurement})}`;
    $('content').append(back);
    for (const [title, value] of [['Transcript and tool calls', data.detail.conversation], ['Model input', data.detail.model_input],
      ['Verifier output', data.detail.verifier ?? data.detail.verifier_result], ['Exception', data.detail.exception || data.detail.exception_info]]) {
      const section = node('section', undefined, 'assessment-card');
      section.append(node('h2', title), value == null ? node('p', 'Not recorded in this attempt artifact.', 'muted') : node('pre', typeof value === 'string' ? value : JSON.stringify(value, null, 2)));
      $('content').append(section);
    }
    for (const [title, ref] of [['Raw attempt receipt', data.detail.receipt], ['Raw rollout archive', data.detail.archive]]) {
      const section = node('section', undefined, 'assessment-card'), link = node('a', 'Open in Filer ↗');
      link.href = 'https://filer.marina.oa.dev/#' + encodeURIComponent(ref.uri); link.target = '_blank'; link.rel = 'noopener';
      section.append(node('h2', title), link, node('p', ref.uri), node('p', `SHA-256 ${ref.sha256}`)); $('content').append(section);
    }
  } catch (error) { $('message').textContent = error.message; $('message').className = 'error'; }
})();
