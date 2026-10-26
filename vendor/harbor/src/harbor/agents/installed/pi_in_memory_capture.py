"""Pi extension that captures model evidence inside Pi's own process."""

EXTENSION_SCRIPT = r"""
import fs from 'node:fs';
import { pathToFileURL } from 'node:url';

const stdoutRetry = new Int32Array(new SharedArrayBuffer(4));
function writeEvidenceLine(line) {
  // FD 3 is a dedicated evidence channel saved from the outer command's
  // stdout before Pi's JSON stream enters its validator pipe. Short atomic
  // writes prevent partial frame lines even when the pipe is under pressure.
  const bytes = Buffer.from(line, 'utf8');
  if (bytes.length >= 4096) throw new Error('Pi evidence line exceeds atomic pipe write limit');
  for (let attempt = 0; attempt < 30000; attempt++) {
    try {
      if (fs.writeSync(3, bytes) !== bytes.length) throw new Error('Partial Pi evidence stdout write');
      return;
    } catch (error) {
      if (error?.code !== 'EAGAIN') throw error;
      Atomics.wait(stdoutRetry, 0, 0, 1);
    }
  }
  throw new Error('Pi evidence stdout remained unwritable');
}
import { startRecorder, recordDiagnostic } from '/tmp/harbor-pi-literal-recorder.mjs';

export default async function (pi) {
  const packageRoot = process.env.HARBOR_PI_PACKAGE_ROOT;
  if (!packageRoot) throw new Error('HARBOR_PI_PACKAGE_ROOT is required');
  const aiPath = `${packageRoot}/node_modules/@earendil-works/pi-ai/dist/api/openai-completions.lazy.js`;
  const { openAICompletionsApi } = await import(pathToFileURL(aiPath).href);
  const recorder = await startRecorder();
  const provider = process.env.HARBOR_PI_PROVIDER;
  if (!provider) throw new Error('HARBOR_PI_PROVIDER is required');
  const native = openAICompletionsApi();
  pi.registerProvider(provider, {
    api: 'openai-completions',
    streamSimple(model, context, options) {
      return native.streamSimple({ ...model, baseUrl: recorder.url }, context, options);
    },
  });
  pi.on('tool_result', (event) => {
    const entry = recorder.records.at(-1);
    const content = entry?.actor === 'teacher'
      ? [{ type: 'text', text: 'Action authored and executed by the teacher. The following is its tool observation:\n' },
         ...event.content]
      : event.content;
    recorder.recordObservation(event, content);
    if (entry?.actor === 'teacher') return { content };
  });
  const critique = Boolean(process.env.HARBOR_PI_CRITIQUE_PLAN);
  if (critique) {
    pi.on('turn_end', async (event) => {
      if (recorder.records.length !== 1 || recorder.critique || recorder.critiqueFailure) return;
      try {
        const feedback = await recorder.requestCritique(event.toolResults);
        pi.sendUserMessage(feedback, { deliverAs: 'steer' });
      } catch (error) {
        recorder.failCritique(error);
        throw error;
      }
    });
    pi.on('tool_call', (event) => {
      if (recorder.records.at(-1)?.actor !== 'revision') return;
      recorder.failCritique('Revision attempted a tool call');
      return { block: true, reason: 'Revision must not execute tools', terminate: true };
    });
  }
  pi.on(critique ? 'agent_settled' : 'agent_end', async () => {
    try {
      // SSE can deliver the final token before the recorder finishes parsing and
      // appending its evidence. Wait for the in-flight HTTP handler, not log files.
      await recorder.waitForIdle();
      const payload = critique
        ? { entries: recorder.records, critique: recorder.critique, failure: recorder.critiqueFailure }
        : process.env.HARBOR_PI_DAGGER_PLAN
          ? { entries: recorder.records, dagger_turns: recorder.daggerTurns }
          : { entries: recorder.records };
      payload.trajectory = recorder.buildTrajectory();
      recordDiagnostic('capture_complete', { turn_count: recorder.records.length });
      // Short numbered lines survive stdout transports that split or reorder
      // large writes. The payload stays in memory and is never a log artifact.
      const encoded = Buffer.from(JSON.stringify(payload), 'utf8').toString('base64');
      const size = 1024;
      const total = Math.ceil(encoded.length / size);
      for (let index = 0; index < total; index++) {
        writeEvidenceLine(`HARBOR_PI_EVIDENCE_V2:${index}/${total}:${encoded.slice(index * size, (index + 1) * size)}\n`);
      }
    } catch (error) {
      // Do not log the error message: it may contain task text or an endpoint URL.
      recordDiagnostic('capture_failed', { turn_count: recorder.records.length,
        missing_prompt_ids: recorder.records.some(entry => !Array.isArray(entry.literal?.prompt_token_ids)),
        missing_completion_ids: recorder.records.some(entry => !entry.literal?.completion_token_ids?.length),
        missing_logprobs: recorder.records.some(entry =>
          entry.literal?.completion_token_ids?.length !== entry.literal?.logprobs?.length),
        missing_assistant: recorder.records.some(entry => !entry.literal?.assistant_message) });
      throw error;
    } finally {
      // Validation failure must not leave the loopback server holding Pi's
      // event loop open until the outer trial timeout deletes the sandbox.
      recorder.server.close();
    }
  });
}
"""
