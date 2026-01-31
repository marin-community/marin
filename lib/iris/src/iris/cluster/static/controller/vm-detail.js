/**
 * VM detail page — fetches VM info from autoscaler status and bootstrap logs.
 * Extracts vm_id from the URL path: /vm/{vm_id}
 */
import { h, render } from 'preact';
import { useState, useEffect, useRef, useCallback } from 'preact/hooks';
import htm from 'htm';
import { controllerRpc } from '/static/shared/rpc.js';
import { formatVmState } from '/static/shared/utils.js';
import { InfoRow, InfoCard } from '/static/shared/components.js';

const html = htm.bind(h);

const vmId = decodeURIComponent(window.location.pathname.split('/vm/')[1]);

function VmDetailApp() {
  const [vmInfo, setVmInfo] = useState(null);
  const [groupName, setGroupName] = useState('-');
  const [sliceId, setSliceId] = useState('-');
  const [logs, setLogs] = useState('Loading logs...');
  const [error, setError] = useState(null);
  const [logState, setLogState] = useState(null);
  const logsRef = useRef(null);

  const refresh = useCallback(async () => {
    try {
      const [autoscalerResp, logsResp] = await Promise.all([
        controllerRpc('GetAutoscalerStatus'),
        controllerRpc('GetVmLogs', { vmId, tail: 500 }),
      ]);

      const status = autoscalerResp.status || {};
      let found = null;
      let foundGroup = '-';
      let foundSlice = '-';

      for (const group of (status.groups || [])) {
        for (const slice of (group.slices || [])) {
          for (const vm of (slice.vms || [])) {
            if (vm.vmId === vmId) {
              found = vm;
              foundGroup = group.name;
              foundSlice = slice.sliceId;
            }
          }
        }
      }

      setVmInfo(found);
      setGroupName(foundGroup);
      setSliceId(foundSlice);

      if (!found && logsResp.state) {
        setLogState(formatVmState(logsResp.state));
      }

      if (logsResp.logs) {
        setLogs(logsResp.logs);
      } else {
        setLogs('No bootstrap logs available');
      }
      setError(null);
    } catch (e) {
      setError('Failed to load VM details: ' + e.message);
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  useEffect(() => {
    if (logsRef.current) logsRef.current.scrollTop = logsRef.current.scrollHeight;
  }, [logs]);

  const state = vmInfo ? formatVmState(vmInfo.state) : (logState || '-');
  const stateClass = state !== '-' ? 'status-' + state : '';

  return html`
    <a href="/#vms" class="back-link">\u2190 Back to Dashboard</a>
    <h1>VM: ${vmId}</h1>

    ${error && html`<div class="error-message">${error}</div>`}

    <div class="info-grid">
      <${InfoCard} title="VM Info">
        <${InfoRow} label="ID" value=${vmId} />
        <${InfoRow} label="State" value=${state} valueClass=${stateClass} />
        <${InfoRow} label="Address" value=${vmInfo ? (vmInfo.address || '-') : '-'} />
        <${InfoRow} label="Worker" value=${vmInfo ? (vmInfo.workerId || '-') : '-'} />
        <${InfoRow} label="Init Phase" value=${vmInfo ? (vmInfo.initPhase || '-') : '-'} />
      <//>
      <${InfoCard} title="Scale Group">
        <${InfoRow} label="Group" value=${groupName} />
        <${InfoRow} label="Slice" value=${sliceId} />
      <//>
    </div>

    ${vmInfo && vmInfo.initError && html`
      <div class="error-message"><strong>Init Error:</strong> ${vmInfo.initError}</div>
    `}

    <h2>Bootstrap Logs</h2>
    <pre ref=${logsRef} style="background:white;padding:15px;border-radius:6px;box-shadow:0 1px 3px rgba(0,0,0,0.12);max-height:600px;overflow-y:auto;font-size:12px;white-space:pre-wrap">${logs}</pre>
  `;
}

render(html`<${VmDetailApp} />`, document.getElementById('root'));
