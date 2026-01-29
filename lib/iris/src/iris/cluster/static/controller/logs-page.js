import { h, render } from 'preact';
import htm from 'htm';
import { controllerRpc } from '/static/shared/rpc.js';
import { LogsApp } from '/static/shared/log-viewer.js';
const html = htm.bind(h);

const fetchLogs = (prefix, limit) =>
  controllerRpc('GetProcessLogs', { prefix, limit });

render(html`<${LogsApp} fetchLogs=${fetchLogs} />`, document.getElementById('root'));
