import { PanelPlugin } from '@grafana/data';
import { InfraPanel } from './components/InfraPanel';
import { InfraPanelOptions } from './types';

export const plugin = new PanelPlugin<InfraPanelOptions>(InfraPanel).setPanelOptions((builder) => {
  return builder.addRadio({
    path: 'view',
    name: 'View',
    defaultValue: 'nightlies',
    settings: {
      options: [
        { value: 'nightlies', label: 'Nightly matrix' },
        { value: 'commits', label: 'Commit strip' },
        { value: 'wandb', label: 'W&B chart' },
      ],
    },
  });
});
