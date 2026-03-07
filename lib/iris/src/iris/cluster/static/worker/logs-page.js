import { h, render } from 'preact';
import htm from 'htm';
import { workerRpc } from '/static/shared/rpc.js';
import { LogViewer } from '/static/shared/log-viewer.js';
const html = htm.bind(h);

render(html`<${LogViewer}
  rpc=${workerRpc}
  source="/system/process"
  title="Worker Logs"
/>`, document.getElementById('root'));
