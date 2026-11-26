import { Fragment, useId, useLayoutEffect, useRef } from 'react';
import { nodes, nodeSettings } from './contracts.js';
import { number, recordedNodeSettings } from './display.js';

const batchPolicies = { full_batch: 'Full batch', rolling: 'Rolling' };

export function NodeInfo({ id, model, onClose }) {
  const dialog = useRef(null);
  const title = useId();
  const recorded = recordedNodeSettings(id, model);
  useLayoutEffect(() => { dialog.current.showModal(); }, []);

  return <dialog ref={dialog} className="node-info" aria-labelledby={title} onClose={onClose}>
    <header>
      <h2 id={title}>{nodes[id].label} settings</h2>
      <button type="button" className="dialog-close" onClick={() => dialog.current.close()}>Close</button>
    </header>
    <p>{nodeSettings[id].purpose}</p>
    <dl>
      {recorded.map(({ key, label, value }) => <Fragment key={key}>
        <dt>{label}</dt>
        <dd>{key === 'batchPolicy' ? batchPolicies[value] ?? value :
          typeof value === 'number' ? number.format(value) : value}</dd>
      </Fragment>)}
    </dl>
  </dialog>;
}
