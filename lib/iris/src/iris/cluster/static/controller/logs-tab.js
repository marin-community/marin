import { h } from 'preact';
import htm from 'htm';
import { controllerRpc } from '/static/shared/rpc.js';
import { LogViewer } from '/static/shared/log-viewer.js';

const html = htm.bind(h);

export function LogsTab() {
  return html`<${LogViewer}
    rpc=${controllerRpc}
    source="/process"
    title="Controller Logs"
  />`;
}
