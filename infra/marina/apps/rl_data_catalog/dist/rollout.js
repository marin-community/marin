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

    // The transcript sections draw the human view through the shared renderer and keep the
    // original bytes one click away; the JSON sections are already the raw bytes.
    function renderedSection(title, build) {
      const section = node('section', undefined, 'assessment-card');
      section.append(node('h2', title));
      build(section);
      return section;
    }

    function rawDisclosure(section, value) {
      const details = node('details', undefined, 'raw-json');
      details.append(node('summary', 'Raw JSON'));
      details.append(node('pre', typeof value === 'string' ? value : JSON.stringify(value, null, 2), 'raw-json-view'));
      section.append(details);
    }

    function transcriptSection(title, value) {
      return renderedSection(title, (section) => {
        const body = node('div');
        section.append(body);
        if (TranscriptView.normalizeTurns(value).length) {
          TranscriptView.renderTranscript(body, value, {resultMode: 'collapsed'});
        } else {
          body.append(node('p', 'No conversation recorded in this attempt artifact.', 'muted'));
        }
        rawDisclosure(section, value);
      });
    }

    function jsonSection(title, value) {
      return renderedSection(title, (section) => {
        if (value == null) {
          section.append(node('p', 'Not recorded in this attempt artifact.', 'muted'));
          return;
        }
        section.append(node('pre', JSON.stringify(value, null, 2), 'json-view'));
      });
    }

    $('content').append(
      transcriptSection('Transcript and tool calls', data.detail.conversation),
      transcriptSection('Model input', data.detail.model_input),
      jsonSection('Verifier output', data.detail.verifier ?? data.detail.verifier_result),
      jsonSection('Exception', data.detail.exception || data.detail.exception_info),
    );
    for (const [title, ref] of [['Raw attempt receipt', data.detail.receipt], ['Raw rollout archive', data.detail.archive]]) {
      const section = node('section', undefined, 'assessment-card'), link = node('a', 'Open in Filer ↗');
      link.href = 'https://filer.marina.oa.dev/#' + encodeURIComponent(ref.uri); link.target = '_blank'; link.rel = 'noopener';
      section.append(node('h2', title), link, node('p', ref.uri), node('p', `SHA-256 ${ref.sha256}`)); $('content').append(section);
    }
  } catch (error) { $('message').textContent = error.message; $('message').className = 'error'; }
})();
